"""
Train Edit Policy using the new RobustAligner (exhaustive search).
Works with remastered/reordered audio.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import librosa
from tqdm import tqdm
import logging
from typing import List, Tuple

from align_by_search import RobustAligner, MatchResult
from edit_policy import SegmentFeatureExtractor, EditClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EditPolicyTrainerV2:
    """
    Train edit policy using RobustAligner for better matching
    with remastered/reordered audio.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_dir: str = "./models",
        segment_duration: float = 5.0,
        align_segment_duration: float = 3.0,
        similarity_threshold: float = 0.4,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.align_segment_duration = align_segment_duration
        self.similarity_threshold = similarity_threshold
        
        # Use RobustAligner for alignment
        self.aligner = RobustAligner(
            segment_duration=align_segment_duration,
            hop_duration=0.5
        )
        
        # Feature extractor for training
        self.feature_extractor = SegmentFeatureExtractor()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.classifier = None
        self.feature_dim = None
        
    def find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching raw/edit pairs (supports both original and mastered)."""
        pairs = []
        
        # Try mastered versions first
        for raw_file in self.input_dir.glob("*_raw*.wav"):
            # Get base name (handle both _raw.wav and _raw_mastered.wav)
            stem = raw_file.stem
            if "_mastered" in stem:
                prefix = stem.replace("_raw_mastered", "")
                edit_patterns = [
                    f"{prefix}_edit_mastered.wav",
                    f"{prefix}_edit.wav"
                ]
            else:
                prefix = stem.replace("_raw", "")
                edit_patterns = [
                    f"{prefix}_edit_mastered.wav",
                    f"{prefix}_edit.wav"
                ]
            
            # Look for matching edit
            for pattern in edit_patterns:
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
    ) -> List[Tuple[float, float, np.ndarray, bool]]:
        """
        Analyze a single raw/edit pair using RobustAligner.
        
        Returns:
            List of (start, end, features, kept) for each segment
        """
        logger.info(f"Analyzing: {raw_path.stem}")
        
        # Get alignment matches
        matches = self.aligner.align(
            str(raw_path),
            str(edit_path),
            similarity_threshold=self.similarity_threshold
        )
        
        # Load raw audio for feature extraction
        raw_audio, sr = librosa.load(str(raw_path), sr=22050, mono=True)
        raw_duration = len(raw_audio) / sr
        
        # Convert matches to kept segments
        kept_segments = self.aligner.get_kept_raw_segments(
            matches, raw_duration, self.segment_duration
        )
        
        # Extract features for each segment
        results = []
        for start, end, kept in kept_segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if end_sample > len(raw_audio):
                continue
                
            segment_audio = raw_audio[start_sample:end_sample]
            features = self.feature_extractor.extract(segment_audio)
            results.append((start, end, features, kept))
        
        kept_count = sum(1 for r in results if r[3])
        logger.info(f"  Segments: {len(results)}, Kept: {kept_count} ({100*kept_count/len(results):.1f}%)")
        
        return results
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from all pairs."""
        pairs = self.find_pairs()
        
        all_features = []
        all_labels = []
        
        for raw_path, edit_path in tqdm(pairs, desc="Processing pairs"):
            segments = self.analyze_pair(raw_path, edit_path)
            
            for start, end, features, kept in segments:
                if features is not None:
                    all_features.append(features)
                    all_labels.append(1.0 if kept else 0.0)
        
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
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        resume_from: str = None
    ):
        """Train the keep/cut classifier."""
        logger.info("Training classifier...")
        
        self.classifier = EditClassifier(
            input_dim=self.feature_dim,
            hidden_dim=256
        ).to(self.device)
        
        # Load existing weights if resuming
        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming from: {resume_from}")
            self.classifier.load_state_dict(
                torch.load(resume_from, weights_only=True)
            )
        
        # Handle class imbalance with weighted loss
        pos_ratio = y.mean()
        neg_ratio = 1 - pos_ratio
        pos_weight = torch.tensor([neg_ratio / pos_ratio]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        best_loss = float('inf')
        best_state = None
        
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
                best_state = self.classifier.state_dict().copy()
            
            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.classifier.eval()
                with torch.no_grad():
                    probs = self.classifier.predict_proba(X_tensor)
                    preds = (probs > 0.5).float()
                    
                    acc = (preds == y_tensor).float().mean().item()
                    
                    # Precision and recall for "keep" class
                    tp = ((preds == 1) & (y_tensor == 1)).sum().item()
                    fp = ((preds == 1) & (y_tensor == 0)).sum().item()
                    fn = ((preds == 0) & (y_tensor == 1)).sum().item()
                    
                    prec = tp / (tp + fp + 1e-8)
                    rec = tp / (tp + fn + 1e-8)
                    f1 = 2 * prec * rec / (prec + rec + 1e-8)
                
                logger.info(
                    f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.3f}, "
                    f"prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}"
                )
        
        # Restore best model
        if best_state is not None:
            self.classifier.load_state_dict(best_state)
        
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        
    def save_models(self):
        """Save trained models."""
        # Save classifier
        if self.classifier is not None:
            torch.save(
                self.classifier.state_dict(),
                self.model_dir / "classifier_v2_best.pt"
            )
        
        # Save feature dimension
        if self.feature_dim is not None:
            np.save(self.model_dir / "feature_dim_v2.npy", self.feature_dim)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def train(self, epochs: int = 200, resume_from: str = None, lr: float = 1e-3):
        """Full training pipeline."""
        # Prepare data
        X, y = self.prepare_training_data()
        
        # Train classifier
        self.train_classifier(X, y, epochs=epochs, resume_from=resume_from, lr=lr)
        
        # Save
        self.save_models()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train edit policy V2 (with robust alignment)"
    )
    parser.add_argument(
        "--input", "-i",
        default="./training_data/input_mastered",
        help="Directory with mastered raw files"
    )
    parser.add_argument(
        "--output", "-o", 
        default="./training_data/output_mastered",
        help="Directory with mastered edit files"
    )
    parser.add_argument(
        "--model-dir", "-m",
        default="./models",
        help="Where to save trained models"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.4,
        help="Similarity threshold for alignment"
    )
    parser.add_argument(
        "--resume", "-r",
        default=None,
        help="Path to existing model weights to resume from"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (use lower like 1e-4 when resuming)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Edit Policy Training V2 (Robust Alignment)")
    logger.info("=" * 60)
    logger.info(f"Input (mastered raw): {args.input}")
    logger.info(f"Output (mastered edit): {args.output}")
    logger.info(f"Model dir: {args.model_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Resume from: {args.resume}")
    logger.info(f"Similarity threshold: {args.threshold}")
    logger.info("=" * 60)
    
    trainer = EditPolicyTrainerV2(
        input_dir=args.input,
        output_dir=args.output,
        model_dir=args.model_dir,
        similarity_threshold=args.threshold
    )
    
    trainer.train(epochs=args.epochs, resume_from=args.resume, lr=args.lr)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
