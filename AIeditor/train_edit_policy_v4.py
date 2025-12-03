"""
Edit Policy V4 - Simpler approach:
- Positive samples: segments from the EDIT file (what user kept)
- Negative samples: segments from RAW that have LOW similarity to any edit segment

No alignment needed - just learn to distinguish "edited" audio from "cut" audio.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import librosa
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUM_WORKERS = max(1, os.cpu_count() - 2)


# =============================================================================
# SIMPLE MLP CLASSIFIER (no transformer overhead)
# =============================================================================

class SimpleEditClassifier(nn.Module):
    """Simple MLP classifier - less prone to overfitting than transformer."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# =============================================================================
# FEATURE EXTRACTOR (same as before but simpler)
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
        
        # 2. Spectral features (6)
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        
        features.append(spec_cent.mean())
        features.append(spec_cent.std())
        features.append(spec_bw.mean())
        features.append(spec_flat.mean())
        features.append(spec_rolloff.mean())
        
        # 3. RMS energy (2)
        rms = librosa.feature.rms(y=audio)
        features.append(rms.mean())
        features.append(rms.std())
        
        # 4. Zero crossing rate (2)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(zcr.mean())
        features.append(zcr.std())
        
        # 5. Chroma (12 mean)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features.extend(chroma.mean(axis=1))
        
        # 6. Tempo (1)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features.append(float(tempo) / 200.0)  # Normalize
        
        # 7. Spectral contrast (7)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        features.extend(contrast.mean(axis=1))
        
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


# =============================================================================
# TRAINER V4
# =============================================================================

class EditPolicyTrainerV4:
    """
    Simpler training approach:
    - Positives: All segments from edit files
    - Negatives: Segments from raw files that don't appear in edit
    
    Uses chromagram similarity to find which raw segments were cut.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_dir: str = "./models",
        segment_duration: float = 5.0,  # Shorter segments
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
        self.feature_dim = None
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching raw/edit pairs."""
        pairs = []
        
        for raw_file in self.input_dir.glob("*_raw*.wav"):
            stem = raw_file.stem
            if "_mastered" in stem:
                prefix = stem.replace("_raw_mastered", "")
            else:
                prefix = stem.replace("_raw", "")
            
            for pattern in [f"{prefix}_edit_mastered.wav", f"{prefix}_edit.wav"]:
                edit_file = self.output_dir / pattern
                if edit_file.exists():
                    pairs.append((raw_file, edit_file))
                    break
        
        logger.info(f"Found {len(pairs)} raw/edit pairs")
        return pairs
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data:
        - Extract features from all EDIT files (positives)
        - Extract features from all RAW files, find non-matching segments (negatives)
        """
        pairs = self.find_pairs()
        
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES (V4 - Simple Positive/Negative)")
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
        
        all_positive_features = []
        all_raw_features = []
        
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
                    if result['is_positive']:
                        all_positive_features.extend(result['features'])
                        print(f"[{completed}/{len(all_files)}] EDIT {result['file']}: {result['n_segments']} segments ({result['elapsed']:.1f}s)")
                    else:
                        all_raw_features.extend(result['features'])
                        print(f"[{completed}/{len(all_files)}] RAW  {result['file']}: {result['n_segments']} segments ({result['elapsed']:.1f}s)")
                else:
                    print(f"[{completed}/{len(all_files)}] FAILED {result['file']}: {result['error']}")
        
        total_elapsed = time.time() - total_start
        
        # Convert to arrays
        X_positive = np.array(all_positive_features)
        X_raw = np.array(all_raw_features)
        
        print(f"\nPositive (edit) segments: {len(X_positive)}")
        print(f"Raw segments: {len(X_raw)}")
        
        # Find negative samples: raw segments that are LEAST similar to any positive
        print("\nFinding negative samples (dissimilar to edit)...")
        
        # Use simple distance-based filtering
        # For each raw segment, find min distance to any positive
        # Keep segments with HIGH distance (they were cut)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_pos_scaled = scaler.fit_transform(X_positive)
        X_raw_scaled = scaler.transform(X_raw)
        
        # Compute distances (in batches to avoid memory issues)
        batch_size = 1000
        min_distances = []
        
        for i in range(0, len(X_raw_scaled), batch_size):
            batch = X_raw_scaled[i:i+batch_size]
            # Distance to nearest positive
            dists = np.sqrt(((batch[:, None, :] - X_pos_scaled[None, :, :]) ** 2).sum(axis=2))
            min_dist = dists.min(axis=1)
            min_distances.extend(min_dist)
        
        min_distances = np.array(min_distances)
        
        # Get target ratio based on actual edit ratios
        # Average keep rate is ~25%, so we want ~75% negatives from raw
        target_negative_ratio = 0.75
        n_negatives = int(len(X_raw) * target_negative_ratio)
        
        # Take segments with HIGHEST distance (most different from edit = likely cut)
        neg_indices = np.argsort(min_distances)[-n_negatives:]
        X_negative = X_raw[neg_indices]
        
        print(f"Selected {len(X_negative)} negative samples (top {target_negative_ratio*100:.0f}% by distance)")
        
        # Balance classes - undersample majority if needed
        n_positive = len(X_positive)
        n_negative = len(X_negative)
        
        if n_positive > n_negative * 2:
            # Undersample positives
            indices = np.random.choice(n_positive, n_negative * 2, replace=False)
            X_positive = X_positive[indices]
            print(f"Undersampled positives to {len(X_positive)}")
        elif n_negative > n_positive * 2:
            # Undersample negatives
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
        
        self.feature_dim = X.shape[1]
        
        print("=" * 70)
        print(f"TOTAL: {len(X)} samples ({len(X_positive)} pos, {len(X_negative)} neg)")
        print(f"Feature dimension: {self.feature_dim}")
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
        """Train with validation split to detect overfitting."""
        
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
        self.classifier = SimpleEditClassifier(
            input_dim=self.feature_dim,
            hidden_dim=256,
            dropout=0.3
        ).to(self.device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss with label smoothing
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
        
        logger.info("Training V4 (Simple Positive/Negative)...")
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
            
            # Track best
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.classifier.state_dict().copy()
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
            
            # Log
            best_marker = " *BEST*" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{epochs}: train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.3f} f1={f1:.3f}{best_marker}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best
        if best_state is not None:
            self.classifier.load_state_dict(best_state)
        
        # Final evaluation
        self.classifier.eval()
        with torch.no_grad():
            val_probs = self.classifier.predict_proba(X_val_t).squeeze().cpu().numpy()
        
        logger.info("-" * 70)
        logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
        
        # Save
        torch.save(self.classifier.state_dict(), self.model_dir / "classifier_v4_best.pt")
        np.save(self.model_dir / "feature_dim_v4.npy", self.feature_dim)
        
        logger.info(f"Models saved to {self.model_dir}")
        
        # Show probability distribution
        print(f"\nVal probability distribution:")
        print(f"  Min: {val_probs.min():.3f}")
        print(f"  Max: {val_probs.max():.3f}")
        print(f"  Mean: {val_probs.mean():.3f}")
        print(f"  Std: {val_probs.std():.3f}")


def main():
    logger.info("=" * 60)
    logger.info("Edit Policy Training V4")
    logger.info("  - Simple positive/negative approach")
    logger.info("  - Edit segments = positive")
    logger.info("  - Dissimilar raw segments = negative")
    logger.info("  - Validation split for early stopping")
    logger.info("=" * 60)
    
    trainer = EditPolicyTrainerV4(
        input_dir="training_data/input_mastered",
        output_dir="training_data/output_mastered",
        segment_duration=5.0,
        hop_duration=2.5
    )
    
    trainer.train(
        epochs=200,
        batch_size=64,
        lr=1e-3,
        val_split=0.15
    )
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
