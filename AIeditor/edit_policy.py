"""
Edit Policy Model - Learns YOUR editing style from raw/edit pairs.

This module:
1. Aligns raw recordings with their edited versions (finds which segments were kept)
2. Trains a classifier to predict keep/cut for each segment
3. Learns sequencing patterns (what follows what)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import librosa
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A segment of audio with metadata."""
    start_time: float  # seconds
    end_time: float
    features: Optional[np.ndarray] = None
    kept: bool = False  # Was this segment kept in the edit?
    edit_position: int = -1  # Position in the edit (-1 if cut)


@dataclass 
class AlignmentResult:
    """Result of aligning raw audio with its edit."""
    raw_path: str
    edit_path: str
    segments: List[Segment]
    kept_indices: List[int]  # Which raw segments appear in edit
    edit_order: List[int]  # Order of kept segments in edit
    kept_ratio: float


class AudioFingerprinter:
    """
    Finds which segments from raw audio appear in the edited version.
    Uses chromagram correlation for robust matching.
    """
    
    def __init__(
        self,
        segment_duration: float = 5.0,  # seconds per segment
        hop_duration: float = 2.5,  # overlap between segments
        sr: int = 22050,
        match_threshold: float = 0.7
    ):
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sr = sr
        self.match_threshold = match_threshold
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_fingerprint(self, audio: np.ndarray) -> np.ndarray:
        """Extract chromagram fingerprint from audio segment."""
        # Chromagram is robust to volume changes and slight tempo variations
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        # Also get MFCCs for timbral matching
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=13, hop_length=self.hop_length
        )
        # Combine into single fingerprint (mean over time)
        chroma_mean = np.mean(chroma, axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)
        return np.concatenate([chroma_mean, mfcc_mean])
    
    def segment_audio(self, audio: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
        """Split audio into overlapping segments with fingerprints."""
        segments = []
        duration = len(audio) / self.sr
        
        start = 0.0
        while start + self.segment_duration <= duration:
            end = start + self.segment_duration
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            
            segment_audio = audio[start_sample:end_sample]
            fingerprint = self.extract_fingerprint(segment_audio)
            
            segments.append((start, end, fingerprint))
            start += self.hop_duration
        
        return segments
    
    def find_matches(
        self,
        raw_audio: np.ndarray,
        edit_audio: np.ndarray
    ) -> AlignmentResult:
        """
        Find which raw segments appear in the edit.
        Returns alignment information.
        """
        # Segment both
        raw_segments = self.segment_audio(raw_audio)
        edit_segments = self.segment_audio(edit_audio)
        
        logger.info(f"Raw: {len(raw_segments)} segments, Edit: {len(edit_segments)} segments")
        
        # Build fingerprint matrices
        raw_fps = np.array([s[2] for s in raw_segments])
        edit_fps = np.array([s[2] for s in edit_segments])
        
        # Normalize
        raw_fps = raw_fps / (np.linalg.norm(raw_fps, axis=1, keepdims=True) + 1e-8)
        edit_fps = edit_fps / (np.linalg.norm(edit_fps, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix (raw x edit)
        similarity = np.dot(raw_fps, edit_fps.T)
        
        # Find best matches for each edit segment
        kept_indices = []
        edit_order = []
        matched_raw = set()
        
        for edit_idx in range(len(edit_segments)):
            # Find best matching raw segment
            similarities = similarity[:, edit_idx]
            best_raw_idx = np.argmax(similarities)
            best_score = similarities[best_raw_idx]
            
            if best_score >= self.match_threshold and best_raw_idx not in matched_raw:
                kept_indices.append(best_raw_idx)
                edit_order.append(best_raw_idx)
                matched_raw.add(best_raw_idx)
        
        # Remove duplicates while preserving order
        kept_indices = sorted(set(kept_indices))
        
        # Create Segment objects
        segments = []
        for i, (start, end, fp) in enumerate(raw_segments):
            seg = Segment(
                start_time=start,
                end_time=end,
                features=fp,
                kept=(i in kept_indices),
                edit_position=edit_order.index(i) if i in edit_order else -1
            )
            segments.append(seg)
        
        kept_ratio = len(kept_indices) / len(raw_segments) if raw_segments else 0
        
        return AlignmentResult(
            raw_path="",
            edit_path="",
            segments=segments,
            kept_indices=kept_indices,
            edit_order=edit_order,
            kept_ratio=kept_ratio
        )


class SegmentFeatureExtractor:
    """Extract rich features from audio segments for the classifier."""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract feature vector from audio segment."""
        features = []
        
        # 1. Energy statistics
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        features.extend([
            np.mean(rms), np.std(rms), np.max(rms), np.min(rms),
            np.percentile(rms, 25), np.percentile(rms, 75)
        ])
        
        # 2. Spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr, hop_length=self.hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr, hop_length=self.hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr, hop_length=self.hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        
        for feat in [centroid, bandwidth, rolloff, flatness]:
            features.extend([np.mean(feat), np.std(feat)])
        
        # 3. MFCCs (timbral)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13, hop_length=self.hop_length)
        for i in range(13):
            features.extend([np.mean(mfcc[i]), np.std(mfcc[i])])
        
        # 4. Rhythm features
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr, hop_length=self.hop_length)
        features.extend([
            np.mean(onset_env), np.std(onset_env), np.max(onset_env)
        ])
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features.append(float(tempo) / 200.0)  # Normalize
        
        # 5. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # 6. Chroma (harmonic content)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, hop_length=self.hop_length)
        features.extend([np.mean(chroma[i]) for i in range(12)])
        
        # 7. Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr, hop_length=self.hop_length)
        for i in range(contrast.shape[0]):
            features.append(np.mean(contrast[i]))
        
        return np.array(features, dtype=np.float32)


class EditClassifier(nn.Module):
    """
    Neural network that predicts whether a segment should be kept or cut.
    """
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) segment features
        Returns:
            logits: (batch, 1) keep probability logits
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get keep probability."""
        return torch.sigmoid(self.forward(x))


class SequenceModel(nn.Module):
    """
    Learns the ordering/flow of kept segments.
    Given context of previous segments, predicts best next segment.
    """
    
    def __init__(self, feature_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Predict compatibility score between sequence and candidate
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_sequence(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of segments.
        Args:
            segments: (batch, seq_len, feature_dim)
        Returns:
            encoding: (batch, hidden_dim)
        """
        x = self.feature_proj(segments)  # (batch, seq_len, hidden_dim)
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)
        return x[:, -1, :]  # Use last position as summary
    
    def score_candidate(
        self,
        context: torch.Tensor,  # (batch, seq_len, feature_dim)
        candidate: torch.Tensor  # (batch, feature_dim)
    ) -> torch.Tensor:
        """Score how well a candidate follows the context."""
        context_enc = self.encode_sequence(context)  # (batch, hidden_dim)
        candidate_enc = self.feature_proj(candidate)  # (batch, hidden_dim)
        
        combined = torch.cat([context_enc, candidate_enc], dim=-1)
        return self.scorer(combined)


class EditPolicyTrainer:
    """
    Trains the edit policy model from raw/edit pairs.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_dir: str,
        segment_duration: float = 5.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.device = torch.device(device)
        
        self.fingerprinter = AudioFingerprinter(segment_duration=segment_duration)
        self.feature_extractor = SegmentFeatureExtractor()
        
        # Will be set after analyzing data
        self.feature_dim = None
        self.classifier = None
        self.sequence_model = None
    
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching raw/edit pairs."""
        pairs = []
        
        for raw_file in self.input_dir.glob("*_raw.*"):
            prefix = raw_file.stem.replace("_raw", "")
            
            # Look for matching edit
            for ext in [".wav", ".mp3", ".flac"]:
                edit_file = self.output_dir / f"{prefix}_edit{ext}"
                if edit_file.exists():
                    pairs.append((raw_file, edit_file))
                    break
        
        logger.info(f"Found {len(pairs)} raw/edit pairs")
        return pairs
    
    def analyze_pair(self, raw_path: Path, edit_path: Path) -> AlignmentResult:
        """Analyze a single raw/edit pair."""
        logger.info(f"Analyzing: {raw_path.stem}")
        
        # Load audio
        raw_audio, sr = librosa.load(str(raw_path), sr=self.fingerprinter.sr, mono=True)
        edit_audio, _ = librosa.load(str(edit_path), sr=self.fingerprinter.sr, mono=True)
        
        # Find alignment
        alignment = self.fingerprinter.find_matches(raw_audio, edit_audio)
        alignment.raw_path = str(raw_path)
        alignment.edit_path = str(edit_path)
        
        # Extract rich features for each segment
        for seg in alignment.segments:
            start_sample = int(seg.start_time * self.fingerprinter.sr)
            end_sample = int(seg.end_time * self.fingerprinter.sr)
            segment_audio = raw_audio[start_sample:end_sample]
            seg.features = self.feature_extractor.extract(segment_audio)
        
        logger.info(f"  Kept {len(alignment.kept_indices)}/{len(alignment.segments)} segments ({alignment.kept_ratio*100:.1f}%)")
        
        return alignment
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from all pairs."""
        pairs = self.find_pairs()
        
        all_features = []
        all_labels = []
        
        for raw_path, edit_path in tqdm(pairs, desc="Processing pairs"):
            alignment = self.analyze_pair(raw_path, edit_path)
            
            for seg in alignment.segments:
                if seg.features is not None:
                    all_features.append(seg.features)
                    all_labels.append(1.0 if seg.kept else 0.0)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        self.feature_dim = X.shape[1]
        logger.info(f"Prepared {len(X)} segments, {y.sum():.0f} kept ({y.mean()*100:.1f}%)")
        logger.info(f"Feature dimension: {self.feature_dim}")
        
        return X, y
    
    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3
    ):
        """Train the keep/cut classifier."""
        logger.info("Training classifier...")
        
        self.classifier = EditClassifier(
            input_dim=self.feature_dim,
            hidden_dim=256
        ).to(self.device)
        
        # Handle class imbalance
        pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.classifier(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.classifier.state_dict(), self.model_dir / "classifier_best.pt")
            
            if (epoch + 1) % 10 == 0:
                # Evaluate
                self.classifier.eval()
                with torch.no_grad():
                    probs = self.classifier.predict_proba(X_tensor).cpu().numpy()
                    preds = (probs > 0.5).astype(float)
                    acc = (preds.squeeze() == y).mean()
                    
                    # Precision/recall for "keep"
                    tp = ((preds.squeeze() == 1) & (y == 1)).sum()
                    fp = ((preds.squeeze() == 1) & (y == 0)).sum()
                    fn = ((preds.squeeze() == 0) & (y == 1)).sum()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.3f}, prec={precision:.3f}, rec={recall:.3f}")
        
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        torch.save(self.classifier.state_dict(), self.model_dir / "classifier_final.pt")
    
    def train(self, epochs: int = 100):
        """Full training pipeline."""
        X, y = self.prepare_training_data()
        self.train_classifier(X, y, epochs=epochs)
        
        # Save feature dim for inference
        np.save(self.model_dir / "feature_dim.npy", np.array([self.feature_dim]))
        logger.info(f"Models saved to {self.model_dir}")


class EditPolicy:
    """
    Inference class - uses trained model to edit new recordings.
    """
    
    def __init__(
        self,
        model_dir: str,
        segment_duration: float = 5.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.segment_duration = segment_duration
        
        # Load feature dim
        self.feature_dim = int(np.load(self.model_dir / "feature_dim.npy")[0])
        
        # Load classifier
        self.classifier = EditClassifier(input_dim=self.feature_dim).to(self.device)
        self.classifier.load_state_dict(
            torch.load(self.model_dir / "classifier_best.pt", map_location=self.device, weights_only=True)
        )
        self.classifier.eval()
        
        self.feature_extractor = SegmentFeatureExtractor()
        self.sr = 22050
        self.hop_length = 512
    
    def analyze(
        self,
        audio_path: str
    ) -> List[Tuple[float, float, float]]:
        """
        Analyze a raw recording and return segments with keep scores.
        
        Returns:
            List of (start_time, end_time, keep_probability)
        """
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = len(audio) / self.sr
        
        results = []
        hop_duration = self.segment_duration / 2
        
        start = 0.0
        while start + self.segment_duration <= duration:
            end = start + self.segment_duration
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)
            
            segment_audio = audio[start_sample:end_sample]
            features = self.feature_extractor.extract(segment_audio)
            
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                prob = self.classifier.predict_proba(x).item()
            
            results.append((start, end, prob))
            start += hop_duration
        
        return results
    
    def auto_edit(
        self,
        audio_path: str,
        output_path: str,
        threshold: float = 0.5,
        min_gap: float = 2.0,
        crossfade: float = 0.5
    ) -> str:
        """
        Automatically edit a raw recording based on the learned policy.
        
        Args:
            audio_path: Path to raw recording
            output_path: Where to save the edit
            threshold: Keep segments above this probability
            min_gap: Minimum gap between kept segments to merge
            crossfade: Crossfade duration in seconds
            
        Returns:
            Path to edited file
        """
        import soundfile as sf
        
        # Analyze
        segments = self.analyze(audio_path)
        
        # Filter by threshold
        kept = [(s, e, p) for s, e, p in segments if p >= threshold]
        logger.info(f"Keeping {len(kept)}/{len(segments)} segments (threshold={threshold})")
        
        if not kept:
            logger.warning("No segments above threshold!")
            return None
        
        # Merge overlapping/adjacent segments
        merged = []
        current_start, current_end, _ = kept[0]
        
        for start, end, prob in kept[1:]:
            if start <= current_end + min_gap:
                # Extend current segment
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        
        logger.info(f"Merged into {len(merged)} continuous sections")
        
        # Load audio and extract segments
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Build output
        output_parts = []
        crossfade_samples = int(crossfade * sr)
        
        for i, (start, end) in enumerate(merged):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = audio[:, start_sample:end_sample]
            
            # Apply crossfade
            if i > 0 and crossfade_samples > 0 and len(output_parts) > 0:
                # Fade out previous
                fade_len = min(crossfade_samples, output_parts[-1].shape[1])
                fade_out = np.linspace(1, 0, fade_len)
                output_parts[-1][:, -fade_len:] *= fade_out
                
                # Fade in current
                fade_len = min(crossfade_samples, segment.shape[1])
                fade_in = np.linspace(0, 1, fade_len)
                segment[:, :fade_len] *= fade_in
            
            output_parts.append(segment)
        
        # Concatenate
        output = np.concatenate(output_parts, axis=1)
        
        # Save
        sf.write(output_path, output.T, sr)
        
        output_duration = output.shape[1] / sr
        logger.info(f"Saved edit: {output_duration/60:.1f} min -> {output_path}")
        
        return output_path


def main():
    """CLI interface."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Train edit policy from raw/edit pairs")
    parser.add_argument("--input", type=str, required=True, help="Directory with raw recordings")
    parser.add_argument("--output", type=str, required=True, help="Directory with edited versions")
    parser.add_argument("--model-dir", type=str, default="./models", help="Where to save models")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--segment-duration", type=float, default=5.0, help="Segment duration in seconds")
    
    args = parser.parse_args()
    
    trainer = EditPolicyTrainer(
        input_dir=args.input,
        output_dir=args.output,
        model_dir=args.model_dir,
        segment_duration=args.segment_duration
    )
    
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
