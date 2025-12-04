"""
Edit Policy V9 - Contrastive Learning + Reference Similarity

Key innovations:
1. Contrastive learning: Train model to recognize "sounds like references" vs "doesn't"
2. Reference similarity scoring: At inference, boost segments similar to reference songs
3. Dual scoring: Combine edit quality score + reference similarity score

Architecture:
- Feature encoder that learns a "style embedding"
- Contrastive loss to pull reference-like segments together
- Similarity scoring against reference centroid at inference
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
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUM_WORKERS = max(1, os.cpu_count() - 2)


# =============================================================================
# FEATURE EXTRACTOR (same as previous versions)
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
# CONTRASTIVE MODEL - learns style embeddings
# =============================================================================

class StyleEncoder(nn.Module):
    """Encodes audio features into a style embedding space."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, embedding_dim)
        )
        
        # L2 normalize embeddings
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        # L2 normalize for cosine similarity
        return F.normalize(emb, p=2, dim=1)


class DualHeadModel(nn.Module):
    """
    Two-head model:
    1. Quality head: predicts if segment is "keep-worthy" (like V8)
    2. Style head: encodes into style embedding space
    
    Combined score = quality_score + reference_similarity_bonus
    """
    
    def __init__(self, base_feature_dim: int, embedding_dim: int = 64):
        super().__init__()
        
        input_dim = base_feature_dim * 3  # context window
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Quality prediction head
        self.quality_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Style embedding head
        self.style_head = nn.Sequential(
            nn.Linear(128, embedding_dim)
        )
        
        self.base_feature_dim = base_feature_dim
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (quality_logits, style_embeddings)"""
        features = self.backbone(x)
        
        quality = self.quality_head(features)
        style = self.style_head(features)
        style = F.normalize(style, p=2, dim=1)  # L2 normalize
        
        return quality, style
    
    def predict_quality(self, x: torch.Tensor) -> torch.Tensor:
        """Just predict quality score."""
        features = self.backbone(x)
        return torch.sigmoid(self.quality_head(features))
    
    def get_style_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Just get style embedding."""
        features = self.backbone(x)
        style = self.style_head(features)
        return F.normalize(style, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss.
    
    Pulls reference segments together, pushes non-reference apart.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor  # 1 = reference, 0 = non-reference
    ) -> torch.Tensor:
        """
        embeddings: (N, D) normalized embeddings
        labels: (N,) binary labels
        """
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Mask for positive pairs (both from reference)
        ref_mask = labels.unsqueeze(0) * labels.unsqueeze(1)  # both are reference
        
        # Mask for negative pairs (one reference, one not)
        non_ref_mask = (1 - labels.unsqueeze(0)) * labels.unsqueeze(1)
        non_ref_mask = non_ref_mask + non_ref_mask.t()  # symmetric
        
        # Self-mask (diagonal)
        self_mask = torch.eye(len(embeddings), device=embeddings.device)
        
        # For each reference sample, compute loss
        loss = 0
        n_ref = labels.sum().item()
        
        if n_ref < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        ref_indices = torch.where(labels == 1)[0]
        
        for i in ref_indices:
            # Positive: other reference samples
            pos_mask = ref_mask[i] * (1 - self_mask[i])
            
            if pos_mask.sum() == 0:
                continue
            
            # All non-self samples as denominator
            all_mask = 1 - self_mask[i]
            
            # Log-sum-exp trick for numerical stability
            pos_sim = sim_matrix[i][pos_mask.bool()]
            all_sim = sim_matrix[i][all_mask.bool()]
            
            # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
            loss_i = -torch.log(
                torch.exp(pos_sim).sum() / (torch.exp(all_sim).sum() + 1e-8)
            )
            loss += loss_i
        
        return loss / max(n_ref, 1)


# =============================================================================
# TRAINER V9 - Contrastive + Reference Similarity
# =============================================================================

class EditPolicyTrainerV9:
    """
    V9: Contrastive learning with reference similarity scoring.
    
    Training:
    1. Quality classification (edit vs raw) - BCE loss
    2. Style embedding (reference vs non-reference) - Contrastive loss
    
    Inference:
    Combined score = quality + λ * reference_similarity
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        reference_dir: str = None,
        model_dir: str = "./models",
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        embedding_dim: int = 64,
        similarity_weight: float = 0.3,  # How much reference similarity affects final score
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.embedding_dim = embedding_dim
        self.similarity_weight = similarity_weight
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.base_feature_dim = None
        self.reference_centroid = None  # Average embedding of all reference segments
        self.scaler = None
    
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
    
    def prepare_training_data(self) -> Dict:
        """
        Prepare training data with labels for both tasks.
        
        Returns dict with:
        - X: features
        - y_quality: 1=edit, 0=raw (for quality prediction)
        - y_style: 1=reference, 0=non-reference (for contrastive learning)
        """
        pairs = self.find_pairs()
        reference_files = self.find_reference_files()
        
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES (V9 - Contrastive Learning)")
        print("=" * 70)
        
        all_files = []
        
        # Edit files (quality=1, style=0)
        for _, edit in pairs:
            all_files.append((str(edit), self.segment_duration, self.hop_duration, 1, 'edit'))
        
        # Raw files (quality=0, style=0)
        for raw, _ in pairs:
            all_files.append((str(raw), self.segment_duration, self.hop_duration, 0, 'raw'))
        
        # Reference files (quality=1, style=1)
        for ref in reference_files:
            all_files.append((str(ref), self.segment_duration, self.hop_duration, 1, 'reference'))
        
        n_workers = min(len(all_files), NUM_WORKERS)
        
        print(f"Processing:")
        print(f"  - {len(pairs)} edit files")
        print(f"  - {len(pairs)} raw files")
        print(f"  - {len(reference_files)} reference songs")
        print(f"Using {n_workers} workers")
        
        # Store features by source type
        edit_features = []
        raw_features = []
        reference_features = []
        
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
                    feats = np.array(result['features'])
                    
                    if source == 'edit':
                        edit_features.append(feats)
                        print(f"[{completed}/{len(all_files)}] EDIT {result['file']}: {result['n_segments']} seg")
                    elif source == 'raw':
                        raw_features.append(feats)
                        print(f"[{completed}/{len(all_files)}] RAW  {result['file']}: {result['n_segments']} seg")
                    elif source == 'reference':
                        reference_features.append(feats)
                        print(f"[{completed}/{len(all_files)}] REF  {result['file']}: {result['n_segments']} seg")
                else:
                    print(f"[{completed}/{len(all_files)}] FAILED {result['file']}: {result['error']}")
        
        total_elapsed = time.time() - total_start
        
        # Get feature dim
        self.base_feature_dim = edit_features[0].shape[1]
        
        print(f"\nBase feature dim: {self.base_feature_dim}")
        print(f"Context window dim: {self.base_feature_dim * 3}")
        
        # Create context windows
        print("\nCreating context windows...")
        
        def process_features(feat_list, name):
            all_windowed = []
            for feats in feat_list:
                windowed = create_context_windows(feats)
                all_windowed.append(windowed)
            combined = np.vstack(all_windowed) if all_windowed else np.array([])
            print(f"  {name}: {len(combined)} contexts")
            return combined
        
        X_edit = process_features(edit_features, "Edit")
        X_raw = process_features(raw_features, "Raw")
        X_ref = process_features(reference_features, "Reference")
        
        # Create labels
        # Quality: edit=1, reference=1 (both are "good"), raw=0
        # Style: reference=1, edit=0, raw=0
        
        n_edit = len(X_edit)
        n_raw = len(X_raw)
        n_ref = len(X_ref)
        
        # Find dissimilar raw samples (negatives)
        print("\nFinding negative samples...")
        X_positive = np.vstack([X_edit, X_ref])
        
        self.scaler = StandardScaler()
        X_pos_scaled = self.scaler.fit_transform(X_positive)
        X_raw_scaled = self.scaler.transform(X_raw)
        
        # Compute distances
        batch_size = 500
        min_distances = []
        for i in range(0, len(X_raw_scaled), batch_size):
            batch = X_raw_scaled[i:i+batch_size]
            dists = np.sqrt(((batch[:, None, :] - X_pos_scaled[None, :, :]) ** 2).sum(axis=2))
            min_dist = dists.min(axis=1)
            min_distances.extend(min_dist)
        min_distances = np.array(min_distances)
        
        # Select top 75% most different
        n_negatives = int(len(X_raw) * 0.75)
        neg_indices = np.argsort(min_distances)[-n_negatives:]
        X_raw_selected = X_raw[neg_indices]
        n_raw_selected = len(X_raw_selected)
        
        print(f"Selected {n_raw_selected} negative samples from raw")
        
        # Balance classes for quality task
        n_positive_quality = n_edit + n_ref
        if n_raw_selected > n_positive_quality * 2:
            indices = np.random.choice(n_raw_selected, n_positive_quality * 2, replace=False)
            X_raw_selected = X_raw_selected[indices]
            n_raw_selected = len(X_raw_selected)
            print(f"Balanced to {n_raw_selected} negatives")
        
        # Combine all data
        X = np.vstack([X_edit, X_ref, X_raw_selected])
        
        # Quality labels: edit=1, ref=1, raw=0
        y_quality = np.array([1.0] * n_edit + [1.0] * n_ref + [0.0] * n_raw_selected)
        
        # Style labels: ref=1, edit=0, raw=0
        y_style = np.array([0.0] * n_edit + [1.0] * n_ref + [0.0] * n_raw_selected)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y_quality = y_quality[indices]
        y_style = y_style[indices]
        
        print("=" * 70)
        print(f"TOTAL: {len(X)} samples")
        print(f"  Quality: {y_quality.sum():.0f} positive, {len(y_quality) - y_quality.sum():.0f} negative")
        print(f"  Style: {y_style.sum():.0f} reference, {len(y_style) - y_style.sum():.0f} non-reference")
        print(f"Total time: {total_elapsed:.1f}s")
        print("=" * 70 + "\n")
        
        return {
            'X': X,
            'y_quality': y_quality,
            'y_style': y_style,
            'X_ref_raw': X_ref  # Keep raw reference features for centroid
        }
    
    def train(
        self,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_split: float = 0.15,
        contrastive_weight: float = 0.5  # Balance between quality and contrastive loss
    ):
        """Train with dual objectives."""
        
        data = self.prepare_training_data()
        X = data['X']
        y_quality = data['y_quality']
        y_style = data['y_style']
        X_ref = data['X_ref_raw']
        
        # Debug: check shapes
        print(f"DEBUG: X shape: {X.shape}")
        print(f"DEBUG: X_ref shape: {X_ref.shape}")
        
        # Split
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        X_train, y_q_train, y_s_train = X[train_idx], y_quality[train_idx], y_style[train_idx]
        X_val, y_q_val, y_s_val = X[val_idx], y_quality[val_idx], y_style[val_idx]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"Train references: {y_s_train.sum():.0f}")
        
        # Create model
        self.model = DualHeadModel(
            base_feature_dim=self.base_feature_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss functions
        quality_loss_fn = nn.BCEWithLogitsLoss()
        contrastive_loss_fn = ContrastiveLoss(temperature=0.1)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_q_train_t = torch.FloatTensor(y_q_train).unsqueeze(1).to(self.device)
        y_s_train_t = torch.FloatTensor(y_s_train).to(self.device)
        
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_q_val_t = torch.FloatTensor(y_q_val).unsqueeze(1).to(self.device)
        
        # Create dataloader
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_q_train_t, y_s_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        max_patience = 30
        
        logger.info("Training V9 (Contrastive Learning)...")
        logger.info(f"Epochs: {epochs}, Contrastive weight: {contrastive_weight}")
        logger.info("-" * 70)
        
        for epoch in range(epochs):
            self.model.train()
            train_q_loss = 0
            train_c_loss = 0
            
            for batch_x, batch_y_q, batch_y_s in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                quality_logits, style_emb = self.model(batch_x)
                
                # Quality loss
                q_loss = quality_loss_fn(quality_logits, batch_y_q)
                
                # Contrastive loss (only if we have reference samples in batch)
                if batch_y_s.sum() > 1:
                    c_loss = contrastive_loss_fn(style_emb, batch_y_s)
                else:
                    c_loss = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                loss = q_loss + contrastive_weight * c_loss
                
                loss.backward()
                optimizer.step()
                
                train_q_loss += q_loss.item()
                train_c_loss += c_loss.item()
            
            train_q_loss /= len(train_loader)
            train_c_loss /= len(train_loader)
            
            # Validation (quality only for simplicity)
            self.model.eval()
            with torch.no_grad():
                val_q_logits, _ = self.model(X_val_t)
                val_loss = quality_loss_fn(val_q_logits, y_q_val_t).item()
                
                val_probs = torch.sigmoid(val_q_logits).squeeze()
                val_preds = (val_probs > 0.5).float()
                val_labels = y_q_val_t.squeeze()
                
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
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            best_marker = " *BEST*" if is_best else ""
            if (epoch + 1) % 5 == 0 or is_best:
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"q_loss={train_q_loss:.4f} c_loss={train_c_loss:.4f} "
                    f"val={val_loss:.4f} acc={val_acc*100:.1f}% F1={f1:.3f}{best_marker}"
                )
            
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # SAVE MODEL FIRST (before centroid computation, in case it fails)
        model_path = self.model_dir / "classifier_v9_best.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        np.save(self.model_dir / "feature_dim_v9.npy", self.base_feature_dim)
        np.save(self.model_dir / "similarity_weight_v9.npy", self.similarity_weight)
        
        # Compute reference centroid (average style embedding of all reference segments)
        print("\nComputing reference centroid...")
        print(f"X_ref shape: {X_ref.shape}")  # Debug
        self.model.eval()
        
        # Filter reference samples from the training data using y_style labels
        # X_ref from prepare_training_data is already context-windowed
        X_ref_t = torch.FloatTensor(X_ref).to(self.device)
        
        with torch.no_grad():
            # Process in batches if needed
            batch_size = 512
            all_embeddings = []
            for i in range(0, len(X_ref_t), batch_size):
                batch = X_ref_t[i:i+batch_size]
                emb = self.model.get_style_embedding(batch)
                all_embeddings.append(emb)
            
            ref_embeddings = torch.cat(all_embeddings, dim=0)
            self.reference_centroid = ref_embeddings.mean(dim=0).cpu().numpy()
        
        print(f"Reference centroid shape: {self.reference_centroid.shape}")
        
        # Save centroid
        np.save(self.model_dir / "reference_centroid_v9.npy", self.reference_centroid)
        
        # Save scaler
        import pickle
        with open(self.model_dir / "scaler_v9.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return self.model


# =============================================================================
# V9 EDITOR
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
    
    if len(phrase_candidates) == 0:
        return np.array([])
    
    phrase_candidates = np.sort(phrase_candidates)
    
    merged = [phrase_candidates[0]]
    for t in phrase_candidates[1:]:
        if t - merged[-1] > 0.5:
            merged.append(t)
    
    return np.array(merged)


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
            if len(beat_times) > 0:
                distances = np.abs(beat_times - closest_phrase)
                if distances.min() <= 0.25:
                    return beat_times[np.argmin(distances)]
            return closest_phrase
    
    if len(beat_times) > 0:
        distances = np.abs(beat_times - target_time)
        if distances.min() <= 0.5:
            return beat_times[np.argmin(distances)]
    
    return target_time


class V9Editor:
    """V9 editor with contrastive learning + reference similarity."""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.base_feature_dim = int(np.load(self.model_dir / "feature_dim_v9.npy"))
        self.reference_centroid = np.load(self.model_dir / "reference_centroid_v9.npy")
        self.similarity_weight = float(np.load(self.model_dir / "similarity_weight_v9.npy"))
        
        self.model = DualHeadModel(
            base_feature_dim=self.base_feature_dim,
            embedding_dim=len(self.reference_centroid)
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(self.model_dir / "classifier_v9_best.pt", weights_only=True)
        )
        self.model.eval()
        
        self.extractor = SegmentFeatureExtractor()
        
        # Convert centroid to tensor
        self.ref_centroid_t = torch.FloatTensor(self.reference_centroid).to(self.device)
        
        logger.info(f"Loaded V9 model with similarity_weight={self.similarity_weight}")
    
    def compute_combined_score(
        self,
        features: torch.Tensor,
        similarity_weight: float = None
    ) -> np.ndarray:
        """
        Compute combined score: quality + similarity_weight * reference_similarity
        
        Args:
            features: (N, D) tensor of context-windowed features
            similarity_weight: override for self.similarity_weight
        
        Returns:
            (N,) array of combined scores
        """
        if similarity_weight is None:
            similarity_weight = self.similarity_weight
        
        with torch.no_grad():
            # Get quality score and style embedding
            quality_logits, style_emb = self.model(features)
            quality_score = torch.sigmoid(quality_logits).squeeze()
            
            # Compute cosine similarity to reference centroid
            # style_emb is already L2 normalized
            ref_sim = torch.mm(style_emb, self.ref_centroid_t.unsqueeze(1)).squeeze()
            
            # ref_sim is in [-1, 1], shift to [0, 1]
            ref_sim_normalized = (ref_sim + 1) / 2
            
            # Combined score
            combined = quality_score + similarity_weight * ref_sim_normalized
            
            # Normalize to [0, 1] for threshold compatibility
            combined = combined / (1 + similarity_weight)
        
        return combined.cpu().numpy(), quality_score.cpu().numpy(), ref_sim_normalized.cpu().numpy()
    
    def process_track(
        self,
        audio_path: str,
        output_path: str,
        keep_ratio: float = 0.35,
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        crossfade_duration: float = 0.5,
        beat_align: bool = True,
        phrase_detect: bool = True,
        similarity_weight: float = None,  # Override default
    ) -> Dict:
        """Process track with combined quality + reference similarity scoring."""
        
        if similarity_weight is None:
            similarity_weight = self.similarity_weight
        
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
        
        # Get combined scores
        combined_scores, quality_scores, ref_similarity = self.compute_combined_score(
            X, similarity_weight
        )
        
        if combined_scores.ndim == 0:
            combined_scores = np.array([float(combined_scores)])
            quality_scores = np.array([float(quality_scores)])
            ref_similarity = np.array([float(ref_similarity)])
        
        # Per-track calibration on combined score
        n_keep = max(1, int(len(combined_scores) * keep_ratio))
        threshold_idx = np.argsort(combined_scores)[-n_keep]
        adaptive_threshold = combined_scores[threshold_idx]
        
        results = [(s, e, float(c), float(q), float(r)) 
                   for (s, e), c, q, r in zip(times, combined_scores, quality_scores, ref_similarity)]
        kept = [(s, e, c, q, r) for s, e, c, q, r in results if c >= adaptive_threshold]
        
        if len(kept) == 0:
            return {'success': False, 'error': 'No segments kept'}
        
        # Merge adjacent segments
        merged = []
        cs, ce, _, _, _ = kept[0]
        for s, e, c, q, r in kept[1:]:
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
            'similarity_weight': similarity_weight,
            'score_stats': {
                'combined': {'min': float(combined_scores.min()), 'max': float(combined_scores.max()), 'mean': float(combined_scores.mean())},
                'quality': {'min': float(quality_scores.min()), 'max': float(quality_scores.max()), 'mean': float(quality_scores.mean())},
                'ref_similarity': {'min': float(ref_similarity.min()), 'max': float(ref_similarity.max()), 'mean': float(ref_similarity.mean())},
            },
            'keep_ratio_actual': len(kept) / len(features),
            'n_beats': len(beat_times),
            'n_phrases': len(phrase_times),
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    trainer = EditPolicyTrainerV9(
        input_dir="./training_data/input",
        output_dir="./training_data/desired_output",
        reference_dir="./training_data/reference",
        model_dir="./models",
        embedding_dim=64,
        similarity_weight=0.3,  # Reference similarity adds 30% boost to score
    )
    
    trainer.train(
        epochs=200,
        batch_size=64,
        lr=1e-3,
        contrastive_weight=0.5  # Balance quality vs contrastive loss
    )
