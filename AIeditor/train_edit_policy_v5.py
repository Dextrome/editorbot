"""
Edit Policy V5 - Improvements over V4:
1. Per-track threshold calibration (relative ranking instead of absolute threshold)
2. Context window (prev + current + next segment features)

Builds on V4's simple positive/negative approach.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import librosa
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUM_WORKERS = max(1, os.cpu_count() - 2)


# =============================================================================
# CONTEXT-AWARE CLASSIFIER
# =============================================================================

class ContextAwareClassifier(nn.Module):
    """
    MLP classifier that takes context window (prev + current + next segment).
    Input: [prev_features, current_features, next_features] concatenated
    """
    
    def __init__(
        self,
        base_feature_dim: int,  # Single segment feature dim
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Input is 3x features (prev + current + next)
        input_dim = base_feature_dim * 3
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.base_feature_dim = base_feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# =============================================================================
# FEATURE EXTRACTOR (same as V4)
# =============================================================================

class SegmentFeatureExtractor:
    """Extract audio features from a segment."""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from audio segment."""
        features = []
        
        # 1. MFCCs (13 coefficients, mean + std = 26)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        features.extend(mfcc.mean(axis=1))
        features.extend(mfcc.std(axis=1))
        
        # 2. Spectral features (5)
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        
        features.append(spec_cent.mean())
        features.append(spec_bw.mean())
        features.append(spec_flat.mean())
        features.append(spec_rolloff.mean())
        
        # 3. RMS energy (2)
        rms = librosa.feature.rms(y=audio)
        features.append(rms.mean())
        features.append(rms.std())
        
        # 4. Zero crossing rate (1)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(zcr.mean())
        
        # 5. Chroma (12 mean)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features.extend(chroma.mean(axis=1))
        
        # 6. Tempo (1)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features.append(float(tempo) / 200.0)
        
        # 7. Spectral contrast (7)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        features.extend(contrast.mean(axis=1))
        
        # 8. NEW: Onset strength (helps detect "tightness")
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        features.append(onset_env.mean())
        features.append(onset_env.std())
        
        # 9. NEW: Spectral flux (how much sound changes)
        spec = np.abs(librosa.stft(audio))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        features.append(flux.mean())
        features.append(flux.std())
        
        return np.array(features, dtype=np.float32)


def _extract_features_from_file(args) -> Dict:
    """Worker function to extract features from a single file."""
    import time
    start_time = time.time()
    
    file_path, segment_duration, hop_duration, is_positive = args
    file_path = Path(file_path)
    
    try:
        audio, sr = librosa.load(str(file_path), sr=22050, mono=True)
        duration = len(audio) / sr
        
        extractor = SegmentFeatureExtractor(sr=sr)
        
        segments = []
        start = 0.0
        while start + segment_duration <= duration:
            end = start + segment_duration
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            segment_audio = audio[start_sample:end_sample]
            features = extractor.extract(segment_audio)
            segments.append(features)
            
            start += hop_duration
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'file': file_path.stem,
            'features': segments,
            'is_positive': is_positive,
            'n_segments': len(segments),
            'elapsed': elapsed
        }
    except Exception as e:
        return {
            'success': False,
            'file': file_path.stem,
            'error': str(e),
            'elapsed': time.time() - start_time
        }


def create_context_windows(features: np.ndarray) -> np.ndarray:
    """
    Create context windows from sequential features.
    For each segment, concatenate [prev, current, next].
    For first/last segments, duplicate current for missing context.
    """
    n_segments = len(features)
    if n_segments == 0:
        return np.array([])
    
    feature_dim = features.shape[1]
    windowed = []
    
    for i in range(n_segments):
        prev_idx = max(0, i - 1)
        next_idx = min(n_segments - 1, i + 1)
        
        context = np.concatenate([
            features[prev_idx],
            features[i],
            features[next_idx]
        ])
        windowed.append(context)
    
    return np.array(windowed)


# =============================================================================
# TRAINER V5
# =============================================================================

class EditPolicyTrainerV5:
    """
    V5 improvements:
    - Context window (prev + current + next)
    - Per-track threshold calibration
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_dir: str = "./models",
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.classifier = None
        self.base_feature_dim = None
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching raw/edit pairs."""
        pairs = []
        
        # Check both .wav and .mp3 files
        for ext in ['*.wav', '*.mp3']:
            for raw_file in self.input_dir.glob(f"*_raw*{ext[1:]}"):
                stem = raw_file.stem
                if "_mastered" in stem:
                    prefix = stem.replace("_raw_mastered", "")
                else:
                    prefix = stem.replace("_raw", "")
                
                for pattern in [f"{prefix}_edit_mastered.wav", f"{prefix}_edit.wav",
                               f"{prefix}_edit_mastered.mp3", f"{prefix}_edit.mp3"]:
                    edit_file = self.output_dir / pattern
                    if edit_file.exists():
                        pairs.append((raw_file, edit_file))
                        break
        
        logger.info(f"Found {len(pairs)} raw/edit pairs")
        return pairs
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with context windows:
        - Extract features from all files
        - Create context windows (prev + current + next)
        - Positives: edit segments
        - Negatives: raw segments most different from edit
        """
        pairs = self.find_pairs()
        
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES (V5 - Context Windows)")
        print("=" * 70)
        
        # Collect all files to process
        edit_files = [(str(edit), self.segment_duration, self.hop_duration, True) 
                      for _, edit in pairs]
        raw_files = [(str(raw), self.segment_duration, self.hop_duration, False) 
                     for raw, _ in pairs]
        
        all_files = edit_files + raw_files
        n_workers = min(len(all_files), NUM_WORKERS)
        
        print(f"Processing {len(edit_files)} edit + {len(raw_files)} raw files")
        print(f"Using {n_workers} workers")
        
        # Store per-track features for context window creation
        track_edit_features = {}  # track_name -> list of features
        track_raw_features = {}
        
        import time
        total_start = time.time()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_extract_features_from_file, args): args 
                      for args in all_files}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                
                if result['success']:
                    track_name = result['file'].replace('_edit', '').replace('_raw', '')
                    track_name = track_name.replace('_mastered', '')
                    
                    if result['is_positive']:
                        track_edit_features[track_name] = np.array(result['features'])
                        print(f"[{completed}/{len(all_files)}] EDIT {result['file']}: {result['n_segments']} segments ({result['elapsed']:.1f}s)")
                    else:
                        track_raw_features[track_name] = np.array(result['features'])
                        print(f"[{completed}/{len(all_files)}] RAW  {result['file']}: {result['n_segments']} segments ({result['elapsed']:.1f}s)")
                else:
                    print(f"[{completed}/{len(all_files)}] FAILED {result['file']}: {result['error']}")
        
        total_elapsed = time.time() - total_start
        
        # Get base feature dim
        sample_features = list(track_edit_features.values())[0]
        self.base_feature_dim = sample_features.shape[1]
        
        print(f"\nBase feature dim: {self.base_feature_dim}")
        print(f"Context window dim: {self.base_feature_dim * 3}")
        
        # Create context-windowed features for each track
        all_positive_windowed = []
        all_raw_windowed = []
        
        print("\nCreating context windows per track...")
        for track_name in track_edit_features:
            edit_feats = track_edit_features[track_name]
            raw_feats = track_raw_features.get(track_name)
            
            if raw_feats is None:
                continue
            
            # Create context windows
            edit_windowed = create_context_windows(edit_feats)
            raw_windowed = create_context_windows(raw_feats)
            
            all_positive_windowed.append(edit_windowed)
            all_raw_windowed.append(raw_windowed)
            
            print(f"  {track_name}: {len(edit_windowed)} edit, {len(raw_windowed)} raw contexts")
        
        X_positive = np.vstack(all_positive_windowed)
        X_raw = np.vstack(all_raw_windowed)
        
        print(f"\nTotal positive (edit) contexts: {len(X_positive)}")
        print(f"Total raw contexts: {len(X_raw)}")
        
        # Find negative samples
        print("\nFinding negative samples (dissimilar to edit)...")
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_pos_scaled = scaler.fit_transform(X_positive)
        X_raw_scaled = scaler.transform(X_raw)
        
        # Compute distances (in batches)
        batch_size = 500
        min_distances = []
        
        for i in range(0, len(X_raw_scaled), batch_size):
            batch = X_raw_scaled[i:i+batch_size]
            dists = np.sqrt(((batch[:, None, :] - X_pos_scaled[None, :, :]) ** 2).sum(axis=2))
            min_dist = dists.min(axis=1)
            min_distances.extend(min_dist)
        
        min_distances = np.array(min_distances)
        
        # Select negatives - top 75% by distance
        target_negative_ratio = 0.75
        n_negatives = int(len(X_raw) * target_negative_ratio)
        neg_indices = np.argsort(min_distances)[-n_negatives:]
        X_negative = X_raw[neg_indices]
        
        print(f"Selected {len(X_negative)} negative samples")
        
        # Balance classes
        n_positive = len(X_positive)
        n_negative = len(X_negative)
        
        if n_negative > n_positive * 2:
            indices = np.random.choice(n_negative, n_positive * 2, replace=False)
            X_negative = X_negative[indices]
            print(f"Undersampled negatives to {len(X_negative)}")
        
        # Combine
        X = np.vstack([X_positive, X_negative])
        y = np.array([1.0] * len(X_positive) + [0.0] * len(X_negative))
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print("=" * 70)
        print(f"TOTAL: {len(X)} samples ({len(X_positive)} pos, {len(X_negative)} neg)")
        print(f"Context feature dimension: {X.shape[1]}")
        print(f"Class balance: {y.mean()*100:.1f}% positive")
        print(f"Total time: {total_elapsed:.1f}s")
        print("=" * 70 + "\n")
        
        return X, y
    
    def train(
        self,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_split: float = 0.15
    ):
        """Train with validation split."""
        
        X, y = self.prepare_training_data()
        
        # Split into train/val
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Create model
        self.classifier = ContextAwareClassifier(
            base_feature_dim=self.base_feature_dim,
            hidden_dim=256,
            dropout=0.3
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        bce_loss = nn.BCEWithLogitsLoss()
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        max_patience = 30
        
        logger.info("Training V5 (Context Windows)...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        logger.info("-" * 70)
        
        for epoch in range(epochs):
            # Training
            self.classifier.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.classifier(batch_x)
                loss = bce_loss(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.classifier.eval()
            with torch.no_grad():
                val_logits = self.classifier(X_val_t)
                val_loss = bce_loss(val_logits, y_val_t).item()
                
                val_probs = torch.sigmoid(val_logits).squeeze()
                val_preds = (val_probs > 0.5).float()
                val_labels = y_val_t.squeeze()
                
                val_acc = (val_preds == val_labels).float().mean().item()
                
                tp = ((val_preds == 1) & (val_labels == 1)).sum().item()
                fp = ((val_preds == 1) & (val_labels == 0)).sum().item()
                fn = ((val_preds == 0) & (val_labels == 1)).sum().item()
                
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
            
            scheduler.step(val_loss)
            
            # Check for improvement
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_state = self.classifier.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            best_marker = " *BEST*" if is_best else ""
            logger.info(
                f"Epoch {epoch+1:3d}: "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"acc={val_acc*100:.1f}% F1={f1:.3f}{best_marker}"
            )
            
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_state is not None:
            self.classifier.load_state_dict(best_state)
        
        # Save
        model_path = self.model_dir / "classifier_v5_best.pt"
        torch.save(self.classifier.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save feature dim
        np.save(self.model_dir / "feature_dim_v5.npy", self.base_feature_dim)
        
        return self.classifier


# =============================================================================
# INFERENCE WITH PER-TRACK CALIBRATION
# =============================================================================

class V5Editor:
    """
    V5 inference with per-track threshold calibration.
    Instead of absolute threshold, keeps top N% of segments per track.
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.base_feature_dim = int(np.load(self.model_dir / "feature_dim_v5.npy"))
        
        self.classifier = ContextAwareClassifier(
            base_feature_dim=self.base_feature_dim,
            hidden_dim=256,
            dropout=0.3
        ).to(self.device)
        
        self.classifier.load_state_dict(
            torch.load(self.model_dir / "classifier_v5_best.pt", weights_only=True)
        )
        self.classifier.eval()
        
        self.extractor = SegmentFeatureExtractor()
    
    def process_track(
        self,
        audio_path: str,
        output_path: str,
        keep_ratio: float = 0.35,  # Keep top 35% of segments (per-track calibration)
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        crossfade_duration: float = 0.3
    ) -> Dict:
        """
        Process a track using per-track threshold calibration.
        
        Args:
            audio_path: Input audio file
            output_path: Output audio file
            keep_ratio: Fraction of segments to keep (0.35 = top 35%)
            segment_duration: Segment length in seconds
            hop_duration: Hop between segments
            crossfade_duration: Crossfade duration for joining
        
        Returns:
            Dict with processing info
        """
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(audio) / sr
        
        # Extract features for all segments
        features = []
        times = []
        start = 0.0
        
        while start + segment_duration <= duration:
            end = start + segment_duration
            seg_audio = audio[int(start*sr):int(end*sr)]
            feat = self.extractor.extract(seg_audio)
            features.append(feat)
            times.append((start, end))
            start += hop_duration
        
        if len(features) == 0:
            return {'success': False, 'error': 'File too short'}
        
        features = np.array(features)
        
        # Create context windows
        windowed = create_context_windows(features)
        
        # Get probabilities
        X = torch.FloatTensor(windowed).to(self.device)
        with torch.no_grad():
            probs = self.classifier.predict_proba(X).squeeze().cpu().numpy()
        
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        
        # PER-TRACK CALIBRATION: Keep top N% by probability
        n_keep = max(1, int(len(probs) * keep_ratio))
        threshold_idx = np.argsort(probs)[-n_keep]
        adaptive_threshold = probs[threshold_idx]
        
        # Get segments above threshold
        results = [(s, e, float(p)) for (s, e), p in zip(times, probs)]
        kept = [(s, e, p) for s, e, p in results if p >= adaptive_threshold]
        
        # Merge adjacent kept segments
        if len(kept) == 0:
            return {'success': False, 'error': 'No segments kept'}
        
        merged = []
        cs, ce, _ = kept[0]
        for s, e, p in kept[1:]:
            if s <= ce + hop_duration:  # Adjacent or overlapping
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
        
        # Build output with crossfades
        crossfade_samples = int(crossfade_duration * sr)
        output = []
        
        for i, (s, e) in enumerate(merged):
            seg = audio[int(s*sr):int(e*sr)]
            
            if i > 0 and len(output) >= crossfade_samples and len(seg) > crossfade_samples:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                crossfaded = np.array(output[-crossfade_samples:]) * fade_out + seg[:crossfade_samples] * fade_in
                output = output[:-crossfade_samples]
                output.extend(crossfaded.tolist())
                output.extend(seg[crossfade_samples:].tolist())
            else:
                output.extend(seg.tolist())
        
        output = np.array(output)
        
        # Save
        import soundfile as sf
        sf.write(output_path, output, sr)
        
        return {
            'success': True,
            'input_duration': duration,
            'output_duration': len(output) / sr,
            'n_segments': len(features),
            'n_kept': len(kept),
            'n_regions': len(merged),
            'adaptive_threshold': adaptive_threshold,
            'prob_stats': {
                'min': float(probs.min()),
                'max': float(probs.max()),
                'mean': float(probs.mean()),
                'std': float(probs.std())
            },
            'keep_ratio_actual': len(kept) / len(features),
            'timeline': results
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    trainer = EditPolicyTrainerV5(
        input_dir="./training_data/input",
        output_dir="./training_data/desired_output",
        model_dir="./models"
    )
    
    trainer.train(epochs=200, batch_size=64, lr=1e-3)
