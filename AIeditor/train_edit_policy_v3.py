"""
Edit Policy V3 - Improved training with:
1. Higher alignment threshold (0.55)
2. Longer context (10s segments + 3 neighboring segment features)
3. Transformer attention architecture
4. Contrastive learning from edit-only data

This version should produce better results by learning
what makes audio "edit-worthy" from multiple signals.
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
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from align_by_search import RobustAligner
from edit_policy import SegmentFeatureExtractor
from fast_features import CachedFeatureExtractor, GPUFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Number of parallel workers (per-track parallelism)
NUM_WORKERS = max(1, os.cpu_count() - 2)


# =============================================================================
# TRANSFORMER CLASSIFIER WITH CONTEXT
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEditClassifier(nn.Module):
    """
    Transformer-based classifier that sees context from neighboring segments.
    
    Takes a sequence of segment features and predicts keep/cut for each.
    Uses self-attention to learn patterns across time.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project input features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Contrastive projection head (for edit quality learning)
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)  # 128-dim embedding for contrastive
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - sequence of segment features
            return_embeddings: if True, also return contrastive embeddings
            
        Returns:
            logits: (batch, seq_len, 1) - keep/cut logits for each segment
            embeddings: (batch, seq_len, 128) - if return_embeddings=True
        """
        # Project to hidden dim
        h = self.input_proj(x)
        
        # Add positional encoding
        h = self.pos_encoding(h)
        
        # Transformer
        h = self.transformer(h)
        
        # Classification logits
        logits = self.output_head(h)
        
        if return_embeddings:
            embeddings = self.contrastive_head(h)
            embeddings = F.normalize(embeddings, dim=-1)
            return logits, embeddings
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities for keep class."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


# =============================================================================
# CONTEXT-AWARE FEATURE EXTRACTOR
# =============================================================================

class ContextFeatureExtractor:
    """
    Extract features with context from neighboring segments.
    
    For each segment, includes features from N neighbors on each side,
    giving the model temporal context.
    
    Uses GPU-accelerated extraction when available.
    """
    
    def __init__(
        self,
        segment_duration: float = 10.0,
        hop_duration: float = 5.0,
        num_neighbors: int = 3,
        sr: int = 22050,
        cache_dir: str = "./feature_cache",
        use_gpu: bool = True
    ):
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.num_neighbors = num_neighbors
        self.sr = sr
        self.cache_dir = cache_dir
        
        # Use GPU-accelerated extractor with caching
        self.cached_extractor = CachedFeatureExtractor(
            cache_dir=cache_dir,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            sr=sr,
            use_gpu=use_gpu
        )
        
        # Fallback for in-memory audio
        self.gpu_extractor = GPUFeatureExtractor(sr=sr)
    
    def extract_all_segments_from_file(
        self,
        audio_path: str
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract features from audio file (uses cache).
        """
        return self.cached_extractor.extract_all_segments(audio_path)
    
    def extract_all_segments(
        self,
        audio: np.ndarray
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract features for all segments in audio array.
        Uses GPU batch processing for speed.
        
        Returns:
            List of (start_time, end_time, features) tuples
        """
        duration = len(audio) / self.sr
        
        # Collect all segment audio first
        segments_audio = []
        segments_times = []
        
        start = 0.0
        while start + self.segment_duration <= duration:
            end = start + self.segment_duration
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            
            segments_audio.append(audio[start_sample:end_sample])
            segments_times.append((start, end))
            start += self.hop_duration
        
        # Batch extract on GPU
        if segments_audio:
            features_list = self.gpu_extractor.extract_batch(segments_audio)
        else:
            features_list = []
        
        # Combine
        segments = [
            (start, end, features)
            for (start, end), features in zip(segments_times, features_list)
        ]
        
        return segments
    
    def get_context_features(
        self,
        all_features: List[np.ndarray],
        center_idx: int
    ) -> np.ndarray:
        """
        Get features for a segment including its neighbors.
        
        Creates a feature vector that includes:
        - Center segment features
        - Mean of left neighbor features
        - Mean of right neighbor features
        - Delta features (change from neighbors)
        """
        n = len(all_features)
        center = all_features[center_idx]
        
        # Get left neighbors
        left_indices = [max(0, center_idx - i - 1) for i in range(self.num_neighbors)]
        left_features = [all_features[i] for i in left_indices]
        left_mean = np.mean(left_features, axis=0)
        
        # Get right neighbors
        right_indices = [min(n - 1, center_idx + i + 1) for i in range(self.num_neighbors)]
        right_features = [all_features[i] for i in right_indices]
        right_mean = np.mean(right_features, axis=0)
        
        # Compute deltas
        delta_left = center - left_mean
        delta_right = center - right_mean
        
        # Combine: center + left_mean + right_mean + deltas
        combined = np.concatenate([
            center,
            left_mean,
            right_mean,
            delta_left,
            delta_right
        ])
        
        return combined


# =============================================================================
# CONTRASTIVE LOSS
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning edit quality.
    
    Segments from edits (positive) should be similar to each other
    and different from segments likely cut (negative).
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim) normalized embeddings
            labels: (batch,) binary labels (1=from edit, 0=likely cut)
        
        Returns:
            Contrastive loss
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        
        # Remove diagonal
        batch_size = embeddings.size(0)
        pos_mask.fill_diagonal_(0)
        
        # Negative mask
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        
        # For each sample, compute log(pos / (pos + neg))
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)
        
        # Avoid division by zero
        pos_sum = pos_sum + 1e-8
        
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        
        return loss.mean()


# =============================================================================
# PARALLEL TRACK PROCESSING
# =============================================================================

def _process_track_worker(args: Tuple) -> Dict:
    """
    Worker function to process a single track pair in a separate process.
    This runs completely independently - no shared state.
    
    Args:
        args: (raw_path, edit_path, segment_duration, hop_duration, num_neighbors, similarity_threshold)
    
    Returns:
        Dict with track results or error info
    """
    import time
    start_time = time.time()
    
    raw_path, edit_path, segment_duration, hop_duration, num_neighbors, similarity_threshold = args
    raw_path = Path(raw_path)
    edit_path = Path(edit_path)
    
    try:
        # Create local instances (no shared state)
        aligner = RobustAligner()
        feature_extractor = ContextFeatureExtractor(
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            num_neighbors=num_neighbors
        )
        
        # Load audio
        raw_audio, sr = librosa.load(str(raw_path), sr=22050, mono=True)
        edit_audio, _ = librosa.load(str(edit_path), sr=22050, mono=True)
        
        raw_duration = len(raw_audio) / sr
        
        # Get alignment
        matches = aligner.align(
            str(raw_path),
            str(edit_path),
            similarity_threshold=similarity_threshold
        )
        
        # Get kept segments
        kept_segments = aligner.get_kept_raw_segments(
            matches, raw_duration, segment_duration
        )
        
        # Extract features for raw segments
        raw_segments = feature_extractor.extract_all_segments(raw_audio)
        
        # Build labels
        labels = []
        for start, end, _ in raw_segments:
            is_kept = False
            for k_start, k_end, kept in kept_segments:
                if kept and start < k_end and end > k_start:
                    is_kept = True
                    break
            labels.append(1.0 if is_kept else 0.0)
        
        # Extract edit segments
        edit_segments = feature_extractor.extract_all_segments(edit_audio)
        
        # Build context features
        raw_base_features = [f for _, _, f in raw_segments]
        edit_base_features = [f for _, _, f in edit_segments]
        
        track_features = []
        for i in range(len(raw_base_features)):
            context_features = feature_extractor.get_context_features(raw_base_features, i)
            track_features.append(context_features)
        
        track_edit_features = []
        for i in range(len(edit_base_features)):
            context_features = feature_extractor.get_context_features(edit_base_features, i)
            track_edit_features.append(context_features)
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'track_name': raw_path.stem,
            'features': track_features,
            'labels': labels,
            'edit_features': track_edit_features,
            'elapsed': elapsed,
            'n_raw': len(track_features),
            'n_edit': len(track_edit_features),
            'kept_pct': 100 * sum(labels) / len(labels) if labels else 0
        }
        
    except Exception as e:
        return {
            'success': False,
            'track_name': raw_path.stem,
            'error': str(e),
            'elapsed': time.time() - start_time
        }


# =============================================================================
# TRAINER V3
# =============================================================================

class EditPolicyTrainerV3:
    """
    Improved trainer with:
    1. Higher alignment threshold
    2. Context features
    3. Transformer architecture
    4. Contrastive learning
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_dir: str = "./models",
        segment_duration: float = 10.0,
        hop_duration: float = 5.0,
        num_neighbors: int = 3,
        similarity_threshold: float = 0.55,  # Higher threshold!
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.num_neighbors = num_neighbors
        self.similarity_threshold = similarity_threshold
        
        # Aligner with higher threshold
        self.aligner = RobustAligner(
            segment_duration=3.0,  # Short segments for alignment
            hop_duration=0.5
        )
        
        # Context-aware feature extractor
        self.feature_extractor = ContextFeatureExtractor(
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            num_neighbors=num_neighbors
        )
        
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
    
    def analyze_pair(
        self,
        raw_path: Path,
        edit_path: Path
    ) -> Dict:
        """
        Analyze a raw/edit pair with higher threshold alignment.
        
        Returns dict with:
        - raw_features: list of (start, end, features)
        - labels: list of 0/1 for each segment
        - edit_features: list of features from edit file (for contrastive)
        """
        logger.info(f"Analyzing: {raw_path.stem}")
        
        # Load audio
        raw_audio, sr = librosa.load(str(raw_path), sr=22050, mono=True)
        edit_audio, _ = librosa.load(str(edit_path), sr=22050, mono=True)
        
        raw_duration = len(raw_audio) / sr
        edit_duration = len(edit_audio) / sr
        
        # Get alignment with HIGHER threshold
        matches = self.aligner.align(
            str(raw_path),
            str(edit_path),
            similarity_threshold=self.similarity_threshold
        )
        
        # Get kept segments
        kept_segments = self.aligner.get_kept_raw_segments(
            matches, raw_duration, self.segment_duration
        )
        
        # Extract features for raw segments
        raw_segments = self.feature_extractor.extract_all_segments(raw_audio)
        
        # Build labels based on kept_segments
        labels = []
        for start, end, _ in raw_segments:
            # Check if this segment overlaps with any kept segment
            is_kept = False
            for k_start, k_end, kept in kept_segments:
                if kept and start < k_end and end > k_start:
                    is_kept = True
                    break
            labels.append(1.0 if is_kept else 0.0)
        
        # Extract features from edit file (all segments are "good" examples)
        edit_segments = self.feature_extractor.extract_all_segments(edit_audio)
        
        kept_count = sum(labels)
        logger.info(f"  Raw segments: {len(raw_segments)}, Kept: {int(kept_count)} ({100*kept_count/len(raw_segments):.1f}%)")
        logger.info(f"  Edit segments: {len(edit_segments)} (all positive for contrastive)")
        
        return {
            'raw_segments': raw_segments,
            'labels': labels,
            'edit_segments': edit_segments
        }
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with context features.
        Uses parallel processing - one worker per track.
        
        Returns:
            X: features with context
            y: labels
            X_edit: features from edit files (for contrastive)
        """
        import time
        pairs = self.find_pairs()
        
        # Limit workers to number of pairs or NUM_WORKERS
        n_workers = min(len(pairs), NUM_WORKERS)
        
        print("\n" + "=" * 70)
        print(f"PROCESSING {len(pairs)} TRAINING PAIRS ({n_workers} parallel workers)")
        print("=" * 70)
        
        # Prepare args for workers
        worker_args = [
            (
                str(raw_path),
                str(edit_path),
                self.segment_duration,
                self.hop_duration,
                self.num_neighbors,
                self.similarity_threshold
            )
            for raw_path, edit_path in pairs
        ]
        
        all_features = []
        all_labels = []
        all_edit_features = []
        
        total_start = time.time()
        
        # Process tracks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_track_worker, args): i 
                      for i, args in enumerate(worker_args)}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                
                if result['success']:
                    all_features.extend(result['features'])
                    all_labels.extend(result['labels'])
                    all_edit_features.extend(result['edit_features'])
                    
                    print(f"[{completed}/{len(pairs)}] {result['track_name']}")
                    print(f"      Segments: {result['n_raw']} raw, {result['n_edit']} edit")
                    print(f"      Kept: {int(sum(result['labels']))}/{result['n_raw']} ({result['kept_pct']:.1f}%)")
                    print(f"      Time: {result['elapsed']:.1f}s")
                else:
                    print(f"[{completed}/{len(pairs)}] {result['track_name']} - FAILED: {result['error']}")
        
        total_elapsed = time.time() - total_start
        
        X = np.array(all_features)
        y = np.array(all_labels)
        X_edit = np.array(all_edit_features) if all_edit_features else None
        
        self.feature_dim = X.shape[1]
        
        print("=" * 70)
        print(f"TOTAL: {len(X)} raw segments, {int(y.sum())} kept ({y.mean()*100:.1f}%)")
        print(f"       {len(X_edit) if X_edit is not None else 0} edit segments (positive examples)")
        print(f"       Feature dimension (with context): {self.feature_dim}")
        print(f"       Total time: {total_elapsed:.1f}s ({total_elapsed/len(pairs):.1f}s/track avg)")
        print("=" * 70 + "\n")
        
        return X, y, X_edit
    
    def train(
        self,
        epochs: int = 500,
        batch_size: int = 32,
        lr: float = 1e-4,
        contrastive_weight: float = 0.3,
        resume_from: str = None
    ):
        """Train with combined classification + contrastive loss."""
        
        # Prepare data
        X, y, X_edit = self.prepare_training_data()
        
        # Create model
        self.classifier = TransformerEditClassifier(
            input_dim=self.feature_dim,
            hidden_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1
        ).to(self.device)
        
        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming from: {resume_from}")
            self.classifier.load_state_dict(
                torch.load(resume_from, weights_only=True)
            )
        
        # Losses
        pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]).to(self.device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        if X_edit is not None:
            X_edit_tensor = torch.FloatTensor(X_edit).to(self.device)
        
        # Create dataset - add sequence dimension for transformer
        # For now, treat each sample independently (seq_len=1)
        # TODO: Could batch consecutive segments together for better context
        X_tensor = X_tensor.unsqueeze(1)  # (N, 1, features)
        y_tensor = y_tensor.unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        best_loss = float('inf')
        best_acc = 0
        best_f1 = 0
        best_state = None
        
        logger.info("Training with Transformer + Contrastive learning...")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        logger.info("-" * 70)
        
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            total_bce = 0
            total_contrast = 0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                # Forward pass with embeddings
                logits, embeddings = self.classifier(batch_x, return_embeddings=True)
                
                # BCE loss
                loss_bce = bce_loss(logits, batch_y)
                
                # Contrastive loss (on embeddings)
                # Flatten for contrastive
                emb_flat = embeddings.squeeze(1)  # (batch, 128)
                labels_flat = batch_y.squeeze(1).squeeze(1)  # (batch,)
                
                if emb_flat.size(0) > 1:
                    loss_contrast = contrastive_loss(emb_flat, labels_flat)
                else:
                    loss_contrast = torch.tensor(0.0).to(self.device)
                
                # Combined loss
                loss = loss_bce + contrastive_weight * loss_contrast
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_bce += loss_bce.item()
                total_contrast += loss_contrast.item()
            
            scheduler.step()
            avg_loss = total_loss / len(loader)
            avg_bce = total_bce / len(loader)
            avg_contrast = total_contrast / len(loader)
            
            # Evaluate every epoch
            self.classifier.eval()
            with torch.no_grad():
                all_logits = self.classifier(X_tensor)
                probs = torch.sigmoid(all_logits).squeeze()
                preds = (probs > 0.5).float()
                labels = y_tensor.squeeze()
                
                acc = (preds == labels).float().mean().item()
                
                tp = ((preds == 1) & (labels == 1)).sum().item()
                fp = ((preds == 1) & (labels == 0)).sum().item()
                fn = ((preds == 0) & (labels == 1)).sum().item()
                
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
            
            # Track best
            is_best = False
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_acc = acc
                best_f1 = f1
                best_state = self.classifier.state_dict().copy()
                is_best = True
            
            # Log every epoch
            best_marker = " *BEST*" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f} acc={acc:.3f} f1={f1:.3f} prec={prec:.3f} rec={rec:.3f}{best_marker}")
        
        # Restore best
        if best_state is not None:
            self.classifier.load_state_dict(best_state)
        
        logger.info("-" * 70)
        logger.info(f"Training complete!")
        logger.info(f"Best: loss={best_loss:.4f}, acc={best_acc:.3f}, f1={best_f1:.3f}")
        self.save_models()
    
    def save_models(self):
        """Save trained models."""
        if self.classifier is not None:
            torch.save(
                self.classifier.state_dict(),
                self.model_dir / "classifier_v3_best.pt"
            )
        
        if self.feature_dim is not None:
            np.save(self.model_dir / "feature_dim_v3.npy", self.feature_dim)
        
        # Save config
        config = {
            'segment_duration': self.segment_duration,
            'hop_duration': self.hop_duration,
            'num_neighbors': self.num_neighbors,
            'similarity_threshold': self.similarity_threshold,
            'feature_dim': self.feature_dim
        }
        np.save(self.model_dir / "config_v3.npy", config)
        
        logger.info(f"Models saved to {self.model_dir}")


# =============================================================================
# INFERENCE
# =============================================================================

class EditPolicyV3:
    """Inference with V3 model."""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        config = np.load(self.model_dir / "config_v3.npy", allow_pickle=True).item()
        self.segment_duration = config['segment_duration']
        self.hop_duration = config['hop_duration']
        self.num_neighbors = config['num_neighbors']
        self.feature_dim = config['feature_dim']
        
        # Load model
        self.classifier = TransformerEditClassifier(
            input_dim=self.feature_dim,
            hidden_dim=256,
            num_heads=4,
            num_layers=3
        ).to(self.device)
        
        self.classifier.load_state_dict(
            torch.load(self.model_dir / "classifier_v3_best.pt", weights_only=True)
        )
        self.classifier.eval()
        
        # Feature extractor
        self.feature_extractor = ContextFeatureExtractor(
            segment_duration=self.segment_duration,
            hop_duration=self.hop_duration,
            num_neighbors=self.num_neighbors
        )
    
    def analyze(self, audio_path: str) -> List[Tuple[float, float, float]]:
        """Analyze audio and return (start, end, keep_prob) for each segment."""
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Extract all segments
        segments = self.feature_extractor.extract_all_segments(audio)
        base_features = [f for _, _, f in segments]
        
        # Add context
        results = []
        for i, (start, end, _) in enumerate(segments):
            context_features = self.feature_extractor.get_context_features(base_features, i)
            
            with torch.no_grad():
                x = torch.FloatTensor(context_features).unsqueeze(0).unsqueeze(0).to(self.device)
                prob = torch.sigmoid(self.classifier(x)).item()
            
            results.append((start, end, prob))
        
        return results
    
    def auto_edit(
        self,
        audio_path: str,
        output_path: str,
        threshold: float = 0.5,
        min_gap: float = 2.0,
        crossfade: float = 0.5
    ) -> str:
        """Create auto-edit of audio file."""
        import soundfile as sf
        
        results = self.analyze(audio_path)
        kept = [(s, e, p) for s, e, p in results if p >= threshold]
        
        logger.info(f"Keeping {len(kept)}/{len(results)} segments (threshold={threshold})")
        
        if not kept:
            logger.warning("No segments above threshold!")
            return None
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Merge adjacent
        merged = []
        current_start, current_end, _ = kept[0]
        for start, end, _ in kept[1:]:
            if start <= current_end + min_gap:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        
        # Build output with crossfade
        crossfade_samples = int(crossfade * sr)
        output_audio = []
        
        for i, (start, end) in enumerate(merged):
            segment = audio[int(start * sr):int(end * sr)]
            
            if i > 0 and len(output_audio) >= crossfade_samples and len(segment) > crossfade_samples:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                crossfaded = np.array(output_audio[-crossfade_samples:]) * fade_out + segment[:crossfade_samples] * fade_in
                output_audio = output_audio[:-crossfade_samples]
                output_audio.extend(crossfaded.tolist())
                output_audio.extend(segment[crossfade_samples:].tolist())
            else:
                output_audio.extend(segment.tolist())
        
        output_audio = np.array(output_audio)
        sf.write(output_path, output_audio, sr)
        
        logger.info(f"Saved: {output_path} ({len(output_audio)/sr/60:.1f} min)")
        return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Edit Policy V3")
    parser.add_argument("--input", "-i", default="./training_data/input_mastered")
    parser.add_argument("--output", "-o", default="./training_data/output_mastered")
    parser.add_argument("--model-dir", "-m", default="./models")
    parser.add_argument("--epochs", "-e", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", "-t", type=float, default=0.55)
    parser.add_argument("--segment-duration", type=float, default=10.0)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--contrastive-weight", type=float, default=0.3)
    parser.add_argument("--resume", "-r", default=None)
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Edit Policy Training V3")
    logger.info("  - Higher alignment threshold")
    logger.info("  - Longer context (10s + neighbors)")
    logger.info("  - Transformer attention")
    logger.info("  - Contrastive learning")
    logger.info("=" * 60)
    logger.info(f"Similarity threshold: {args.threshold}")
    logger.info(f"Segment duration: {args.segment_duration}s")
    logger.info(f"Num neighbors: {args.num_neighbors}")
    logger.info(f"Contrastive weight: {args.contrastive_weight}")
    logger.info("=" * 60)
    
    trainer = EditPolicyTrainerV3(
        input_dir=args.input,
        output_dir=args.output,
        model_dir=args.model_dir,
        segment_duration=args.segment_duration,
        num_neighbors=args.num_neighbors,
        similarity_threshold=args.threshold
    )
    
    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        contrastive_weight=args.contrastive_weight,
        resume_from=args.resume
    )
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
