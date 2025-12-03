"""
Edit Policy V7 - Reference-Enhanced Training

Improvements over V6:
1. Per-track threshold calibration
2. Context window (prev + current + next segment features)
3. Beat-aligned cuts + phrase detection
4. NEW: Reference song inspiration - learns from songs you like

Training approach:
- Positive samples: your edits + segments from reference songs you like
- Negative samples: raw segments you cut out

This teaches the model both "what I keep" and "what good music sounds like".
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
# CONTEXT-AWARE CLASSIFIER (same as V5/V6)
# =============================================================================

class ContextAwareClassifier(nn.Module):
    """MLP classifier with context window."""
    
    def __init__(
        self,
        base_feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
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
# FEATURE EXTRACTOR (same as V5/V6)
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
        
        # 8. Onset strength (2)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        features.append(onset_env.mean())
        features.append(onset_env.std())
        
        # 9. Spectral flux (2)
        spec = np.abs(librosa.stft(audio))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        features.append(flux.mean())
        features.append(flux.std())
        
        return np.array(features, dtype=np.float32)


def _extract_features_from_file(args) -> Dict:
    """Worker function to extract features from a single file."""
    import time
    start_time = time.time()
    
    file_path, segment_duration, hop_duration, label, source_type = args
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
            'label': label,
            'source_type': source_type,
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
    """Create context windows from sequential features."""
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
# TRAINER V7 - WITH REFERENCE SONGS
# =============================================================================

class EditPolicyTrainerV7:
    """
    V7: Reference-enhanced training.
    
    Uses three sources of data:
    1. Edit files (positive) - what you kept
    2. Reference songs (positive) - songs you like  
    3. Raw rejects (negative) - what you cut
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        reference_dir: str = None,
        model_dir: str = "./models",
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        reference_weight: float = 0.5,  # How much to weight reference samples
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.reference_weight = reference_weight
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.classifier = None
        self.base_feature_dim = None
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching raw/edit pairs."""
        pairs = []
        
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
    
    def find_reference_files(self) -> List[Path]:
        """Find reference songs."""
        if not self.reference_dir or not self.reference_dir.exists():
            return []
        
        refs = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            refs.extend(self.reference_dir.glob(ext))
        
        logger.info(f"Found {len(refs)} reference songs")
        return refs
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with reference songs.
        
        Returns:
            X: features
            y: labels (1=positive, 0=negative)
            weights: sample weights (reference samples can be weighted differently)
        """
        pairs = self.find_pairs()
        reference_files = self.find_reference_files()
        
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES (V7 - Reference Enhanced)")
        print("=" * 70)
        
        # Collect all files to process
        # label: 1 = positive (edit/reference), 0 = negative (raw)
        # source_type: 'edit', 'raw', 'reference'
        
        all_files = []
        
        # Edit files (positive)
        for _, edit in pairs:
            all_files.append((str(edit), self.segment_duration, self.hop_duration, 1, 'edit'))
        
        # Raw files (will extract negatives later)
        for raw, _ in pairs:
            all_files.append((str(raw), self.segment_duration, self.hop_duration, 0, 'raw'))
        
        # Reference files (positive)
        for ref in reference_files:
            all_files.append((str(ref), self.segment_duration, self.hop_duration, 1, 'reference'))
        
        n_workers = min(len(all_files), NUM_WORKERS)
        
        print(f"Processing:")
        print(f"  - {len(pairs)} edit files (positive)")
        print(f"  - {len(pairs)} raw files (for negative mining)")
        print(f"  - {len(reference_files)} reference songs (positive)")
        print(f"Using {n_workers} workers")
        
        # Store features by source type
        edit_features = {}  # track_name -> features array
        raw_features = {}
        reference_features = []  # list of (filename, features array)
        
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
                    source = result['source_type']
                    
                    if source == 'edit':
                        track_name = result['file'].replace('_edit', '').replace('_mastered', '')
                        edit_features[track_name] = np.array(result['features'])
                        print(f"[{completed}/{len(all_files)}] EDIT {result['file']}: {result['n_segments']} seg ({result['elapsed']:.1f}s)")
                    
                    elif source == 'raw':
                        track_name = result['file'].replace('_raw', '').replace('_mastered', '')
                        raw_features[track_name] = np.array(result['features'])
                        print(f"[{completed}/{len(all_files)}] RAW  {result['file']}: {result['n_segments']} seg ({result['elapsed']:.1f}s)")
                    
                    elif source == 'reference':
                        reference_features.append((result['file'], np.array(result['features'])))
                        print(f"[{completed}/{len(all_files)}] REF  {result['file']}: {result['n_segments']} seg ({result['elapsed']:.1f}s)")
                else:
                    print(f"[{completed}/{len(all_files)}] FAILED {result['file']}: {result['error']}")
        
        total_elapsed = time.time() - total_start
        
        # Get feature dim from first successful extraction
        sample_features = list(edit_features.values())[0]
        self.base_feature_dim = sample_features.shape[1]
        
        print(f"\nBase feature dim: {self.base_feature_dim}")
        print(f"Context window dim: {self.base_feature_dim * 3}")
        
        # Create context windows and combine
        print("\nCreating context windows...")
        
        # === POSITIVE SAMPLES ===
        all_positive_windowed = []
        positive_weights = []
        
        # From edits (weight = 1.0)
        for track_name, feats in edit_features.items():
            windowed = create_context_windows(feats)
            all_positive_windowed.append(windowed)
            positive_weights.extend([1.0] * len(windowed))
            print(f"  Edit {track_name}: {len(windowed)} contexts")
        
        # From references (weight = reference_weight)
        for ref_name, feats in reference_features:
            windowed = create_context_windows(feats)
            all_positive_windowed.append(windowed)
            positive_weights.extend([self.reference_weight] * len(windowed))
            print(f"  Ref  {ref_name}: {len(windowed)} contexts (weight={self.reference_weight})")
        
        X_positive = np.vstack(all_positive_windowed)
        w_positive = np.array(positive_weights)
        
        # === NEGATIVE SAMPLES (from raw, dissimilar to positive) ===
        all_raw_windowed = []
        for track_name, feats in raw_features.items():
            windowed = create_context_windows(feats)
            all_raw_windowed.append(windowed)
        
        X_raw = np.vstack(all_raw_windowed)
        
        print(f"\nTotal positive contexts: {len(X_positive)}")
        print(f"  - From edits: {sum(1 for w in positive_weights if w == 1.0)}")
        print(f"  - From references: {sum(1 for w in positive_weights if w == self.reference_weight)}")
        print(f"Total raw contexts: {len(X_raw)}")
        
        # Find negatives by distance to positive samples
        print("\nFinding negative samples (dissimilar to positive)...")
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_pos_scaled = scaler.fit_transform(X_positive)
        X_raw_scaled = scaler.transform(X_raw)
        
        # Compute distances in batches
        batch_size = 500
        min_distances = []
        
        for i in range(0, len(X_raw_scaled), batch_size):
            batch = X_raw_scaled[i:i+batch_size]
            dists = np.sqrt(((batch[:, None, :] - X_pos_scaled[None, :, :]) ** 2).sum(axis=2))
            min_dist = dists.min(axis=1)
            min_distances.extend(min_dist)
        
        min_distances = np.array(min_distances)
        
        # Select top 75% most different as negatives
        target_negative_ratio = 0.75
        n_negatives = int(len(X_raw) * target_negative_ratio)
        neg_indices = np.argsort(min_distances)[-n_negatives:]
        X_negative = X_raw[neg_indices]
        w_negative = np.ones(len(X_negative))  # Weight = 1.0 for negatives
        
        print(f"Selected {len(X_negative)} negative samples")
        
        # Balance classes
        n_positive = len(X_positive)
        n_negative = len(X_negative)
        
        if n_negative > n_positive * 2:
            indices = np.random.choice(n_negative, n_positive * 2, replace=False)
            X_negative = X_negative[indices]
            w_negative = w_negative[indices]
            print(f"Undersampled negatives to {len(X_negative)}")
        
        # Combine
        X = np.vstack([X_positive, X_negative])
        y = np.array([1.0] * len(X_positive) + [0.0] * len(X_negative))
        weights = np.concatenate([w_positive, w_negative])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        weights = weights[indices]
        
        print("=" * 70)
        print(f"TOTAL: {len(X)} samples ({len(X_positive)} pos, {len(X_negative)} neg)")
        print(f"Context feature dimension: {X.shape[1]}")
        print(f"Class balance: {y.mean()*100:.1f}% positive")
        print(f"Total time: {total_elapsed:.1f}s")
        print("=" * 70 + "\n")
        
        return X, y, weights
    
    def train(
        self,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_split: float = 0.15
    ):
        """Train with sample weights for reference data."""
        
        X, y, weights = self.prepare_training_data()
        
        # Split into train/val
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, y_train, w_train = X[train_indices], y[train_indices], weights[train_indices]
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
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        w_train_t = torch.FloatTensor(w_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Custom weighted BCE loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t, w_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        max_patience = 30
        
        logger.info("Training V7 (Reference Enhanced)...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        logger.info(f"Reference weight: {self.reference_weight}")
        logger.info("-" * 70)
        
        for epoch in range(epochs):
            # Training
            self.classifier.train()
            train_loss = 0
            
            for batch_x, batch_y, batch_w in train_loader:
                optimizer.zero_grad()
                logits = self.classifier(batch_x)
                
                # Weighted loss
                loss_per_sample = bce_loss(logits, batch_y)
                loss = (loss_per_sample * batch_w.unsqueeze(1)).mean()
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation (unweighted)
            self.classifier.eval()
            with torch.no_grad():
                val_logits = self.classifier(X_val_t)
                val_loss = nn.BCEWithLogitsLoss()(val_logits, y_val_t).item()
                
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
        model_path = self.model_dir / "classifier_v7_best.pt"
        torch.save(self.classifier.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        np.save(self.model_dir / "feature_dim_v7.npy", self.base_feature_dim)
        
        return self.classifier


# =============================================================================
# V7 EDITOR (same as V6 but loads V7 model)
# =============================================================================

from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def detect_beats(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Detect beat times in audio."""
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times


def detect_phrase_boundaries(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Detect likely phrase boundaries."""
    phrase_candidates = []
    
    # RMS energy dips
    rms = librosa.feature.rms(y=audio)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    rms_smooth = uniform_filter1d(rms, size=10)
    
    inverted = -rms_smooth
    peaks, _ = find_peaks(inverted, distance=50, prominence=0.01)
    
    for idx in peaks:
        if idx < len(rms_times):
            phrase_candidates.append(rms_times[idx])
    
    # Spectral flux peaks
    spec = np.abs(librosa.stft(audio))
    flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
    flux_times = librosa.frames_to_time(np.arange(len(flux)), sr=sr)
    flux_smooth = uniform_filter1d(flux, size=5)
    
    flux_peaks, _ = find_peaks(flux_smooth, distance=50, prominence=np.std(flux_smooth) * 0.5)
    
    for idx in flux_peaks:
        if idx < len(flux_times):
            phrase_candidates.append(flux_times[idx])
    
    # Onset strength dips
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    onset_smooth = uniform_filter1d(onset_env, size=10)
    
    inverted_onset = -onset_smooth
    onset_peaks, _ = find_peaks(inverted_onset, distance=50, prominence=0.1)
    
    for idx in onset_peaks:
        if idx < len(onset_times):
            phrase_candidates.append(onset_times[idx])
    
    if len(phrase_candidates) == 0:
        return np.array([])
    
    phrase_candidates = np.sort(phrase_candidates)
    
    merged = [phrase_candidates[0]]
    for t in phrase_candidates[1:]:
        if t - merged[-1] > 0.5:
            merged.append(t)
    
    return np.array(merged)


def snap_to_beat(time: float, beat_times: np.ndarray, max_shift: float = 0.5) -> float:
    """Snap time to nearest beat."""
    if len(beat_times) == 0:
        return time
    
    distances = np.abs(beat_times - time)
    nearest_idx = np.argmin(distances)
    
    if distances[nearest_idx] <= max_shift:
        return beat_times[nearest_idx]
    return time


def find_best_cut_point(
    target_time: float,
    beat_times: np.ndarray,
    phrase_times: np.ndarray,
    search_window: float = 1.0
) -> float:
    """Find best cut point near target_time."""
    if len(phrase_times) > 0:
        phrase_in_window = phrase_times[
            (phrase_times >= target_time - search_window) &
            (phrase_times <= target_time + search_window)
        ]
        if len(phrase_in_window) > 0:
            closest_phrase = phrase_in_window[np.argmin(np.abs(phrase_in_window - target_time))]
            return snap_to_beat(closest_phrase, beat_times, max_shift=0.25)
    
    return snap_to_beat(target_time, beat_times, max_shift=0.5)


class V7Editor:
    """V7 editor with reference-enhanced model."""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load V7 model
        self.base_feature_dim = int(np.load(self.model_dir / "feature_dim_v7.npy"))
        
        self.classifier = ContextAwareClassifier(
            base_feature_dim=self.base_feature_dim,
            hidden_dim=256,
            dropout=0.3
        ).to(self.device)
        
        self.classifier.load_state_dict(
            torch.load(self.model_dir / "classifier_v7_best.pt", weights_only=True)
        )
        self.classifier.eval()
        
        self.extractor = SegmentFeatureExtractor()
    
    def process_track(
        self,
        audio_path: str,
        output_path: str,
        keep_ratio: float = 0.35,
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        crossfade_duration: float = 0.5,
        beat_align: bool = True,
        phrase_detect: bool = True
    ) -> Dict:
        """Process track with beat-aligned cuts."""
        
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(audio) / sr
        
        print("Detecting beats...")
        beat_times = detect_beats(audio, sr) if beat_align else np.array([])
        print(f"  Found {len(beat_times)} beats")
        
        print("Detecting phrase boundaries...")
        phrase_times = detect_phrase_boundaries(audio, sr) if phrase_detect else np.array([])
        print(f"  Found {len(phrase_times)} phrase boundaries")
        
        # Extract features
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
        windowed = create_context_windows(features)
        
        X = torch.FloatTensor(windowed).to(self.device)
        with torch.no_grad():
            probs = self.classifier.predict_proba(X).squeeze().cpu().numpy()
        
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        
        # Per-track calibration
        n_keep = max(1, int(len(probs) * keep_ratio))
        threshold_idx = np.argsort(probs)[-n_keep]
        adaptive_threshold = probs[threshold_idx]
        
        results = [(s, e, float(p)) for (s, e), p in zip(times, probs)]
        kept = [(s, e, p) for s, e, p in results if p >= adaptive_threshold]
        
        if len(kept) == 0:
            return {'success': False, 'error': 'No segments kept'}
        
        # Merge adjacent segments
        merged = []
        cs, ce, _ = kept[0]
        for s, e, p in kept[1:]:
            if s <= ce + hop_duration:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
        
        # Beat-align cuts
        print(f"Adjusting {len(merged)} regions to beat/phrase boundaries...")
        aligned_regions = []
        for i, (start, end) in enumerate(merged):
            if i == 0:
                new_start = max(0, find_best_cut_point(start, beat_times, phrase_times, search_window=0.5))
            else:
                new_start = find_best_cut_point(start, beat_times, phrase_times, search_window=1.0)
            
            if i == len(merged) - 1:
                new_end = min(duration, find_best_cut_point(end, beat_times, phrase_times, search_window=0.5))
            else:
                new_end = find_best_cut_point(end, beat_times, phrase_times, search_window=1.0)
            
            if new_end > new_start + 0.5:
                aligned_regions.append((new_start, new_end))
        
        merged = aligned_regions if aligned_regions else merged
        
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
            'n_beats': len(beat_times),
            'n_phrases': len(phrase_times),
            'timeline': results
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    trainer = EditPolicyTrainerV7(
        input_dir="./training_data/input",
        output_dir="./training_data/desired_output",
        reference_dir="./training_data/reference",
        model_dir="./models",
        reference_weight=0.5  # Reference samples count as half-weight
    )
    
    trainer.train(epochs=200, batch_size=64, lr=1e-3)
