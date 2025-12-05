"""Behavioral Cloning (Supervised Learning) for Audio Editor.

Pre-trains the policy by imitating human edits (KEEP/CUT decisions).
This provides a better initialization for subsequent RL fine-tuning.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from rl_editor.config import get_default_config, Config
from rl_editor.agent import Agent
from rl_editor.data import AudioDataset
from rl_editor.state import AudioState, StateRepresentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BehavioralCloningDataset(Dataset):
    """Dataset for behavioral cloning.
    
    Creates (state, action_label) pairs from human-edited audio.
    """
    
    def __init__(
        self, 
        data_dir: str,
        config: Config,
        max_beats: int = 500
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Path to training data
            config: Configuration
            max_beats: Maximum beats per track
        """
        self.config = config
        self.max_beats = max_beats
        
        # Load paired audio dataset (has edit labels from raw/edited pairs)
        from .data import PairedAudioDataset
        self.audio_dataset = PairedAudioDataset(
            data_dir, 
            config, 
            include_reference=False  # Only paired data has edit labels
        )
        
        # Build state representation - pass config (Config object)
        # We'll set the actual beat feature dim after loading first sample
        self.state_rep = StateRepresentation(config)
        
        # Pre-extract all (state, label) pairs
        self.samples = []
        self._build_samples()
        
        logger.info(f"Built {len(self.samples)} BC samples from {len(self.audio_dataset)} tracks")
    
    def _build_samples(self):
        """Extract all beat-level samples from dataset.
        
        PairedAudioDataset returns items with:
            - 'raw': dict with beat_times, beat_features, tempo
            - 'edit_labels': tensor of per-beat KEEP=1/CUT=0 labels
        """
        beat_feature_dim_set = False
        
        for idx in range(len(self.audio_dataset)):
            item = self.audio_dataset[idx]
            
            # Skip reference tracks (no edit labels)
            if "edit_labels" not in item or item["edit_labels"] is None:
                continue
            
            # Extract from raw data dict
            raw_data = item["raw"]
            beat_times = raw_data["beat_times"].numpy()
            beat_features = raw_data["beat_features"].numpy()
            tempo = raw_data["tempo"].item()
            edit_labels = item["edit_labels"].numpy()
            
            # Set actual beat feature dimension from data (first sample)
            if not beat_feature_dim_set and beat_features.ndim > 1:
                actual_dim = beat_features.shape[1]
                self.state_rep.set_beat_feature_dim(actual_dim)
                beat_feature_dim_set = True
                logger.info(f"Beat feature dimension: {actual_dim}")
            
            # Truncate to max_beats
            n_beats = min(len(beat_times), self.max_beats)
            beat_times = beat_times[:n_beats]
            beat_features = beat_features[:n_beats]
            edit_labels = edit_labels[:n_beats]
            
            # Build state for each beat
            total_duration = beat_times[-1] if len(beat_times) > 0 else 1.0
            
            for beat_idx in range(n_beats):
                # Create audio state
                audio_state = AudioState(
                    beat_index=beat_idx,
                    beat_times=beat_times,
                    beat_features=beat_features,
                    tempo=tempo
                )
                
                # Create dummy edit history (empty at start, simulating sequential processing)
                from rl_editor.state import EditHistory
                edit_history = EditHistory()
                
                # Add beats processed so far to edit history
                for prev_idx in range(beat_idx):
                    if edit_labels[prev_idx] > 0.5:
                        edit_history.kept_beats.append(prev_idx)
                    else:
                        edit_history.cut_beats.append(prev_idx)
                
                # Compute remaining duration
                kept_duration = sum(
                    beat_times[i+1] - beat_times[i] if i+1 < len(beat_times) else 0
                    for i in edit_history.kept_beats
                )
                target_duration = total_duration * self.config.reward.target_keep_ratio
                remaining_duration = max(0.0, target_duration - kept_duration)
                
                # Build observation
                obs = self.state_rep.construct_observation(
                    audio_state,
                    edit_history,
                    remaining_duration=remaining_duration,
                    total_duration=total_duration
                )
                
                # Label: 0 = KEEP, 1 = CUT (note: edit_labels has KEEP=1, CUT=0)
                # We convert to action space: action 0 = KEEP, action 1 = CUT
                label = 0 if edit_labels[beat_idx] > 0.5 else 1
                
                self.samples.append({
                    "observation": torch.from_numpy(obs).float(),
                    "label": torch.tensor(label).long()
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class BehavioralCloningTrainer:
    """Trainer for behavioral cloning."""
    
    def __init__(
        self,
        config: Config,
        agent: Agent,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        use_class_weights: bool = True
    ):
        """Initialize trainer.
        
        Args:
            config: Configuration
            agent: Agent with policy network
            lr: Learning rate
            weight_decay: Weight decay for regularization
            use_class_weights: Whether to weight classes by frequency
        """
        self.config = config
        self.agent = agent
        self.device = agent.device
        
        # Only train policy network (not value network)
        self.optimizer = torch.optim.AdamW(
            agent.policy_net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.use_class_weights = use_class_weights
        self.class_weights = None  # Computed from data
        
    def compute_class_weights(self, dataset: BehavioralCloningDataset):
        """Compute class weights from dataset."""
        labels = [s["label"].item() for s in dataset.samples]
        counts = np.bincount(labels, minlength=2)
        
        # Inverse frequency weighting with capping to prevent extreme values
        total = sum(counts)
        weights = total / (2 * counts + 1e-8)
        
        # Cap weights to prevent NaN - max weight is 5x
        weights = np.clip(weights, 0.2, 5.0)
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        logger.info(f"Class counts: KEEP={counts[0]}, CUT={counts[1]}")
        logger.info(f"Class weights: KEEP={weights[0]:.3f}, CUT={weights[1]:.3f}")
    
    def train_epoch(
        self, 
        dataloader: DataLoader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            scheduler: Optional LR scheduler
            
        Returns:
            Dictionary of metrics
        """
        self.agent.policy_net.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        nan_batches = 0
        
        for batch in dataloader:
            obs = batch["observation"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Check for NaN in observations
            if torch.isnan(obs).any():
                nan_batches += 1
                continue
            
            # Forward pass - get logits for KEEP and CUT actions (indices 0 and 1)
            logits, _ = self.agent.policy_net(obs)
            
            # Check for NaN in logits
            if torch.isnan(logits).any():
                nan_batches += 1
                continue
            
            # We only care about KEEP (0) vs CUT (1) actions
            # Extract only first 2 logits
            keep_cut_logits = logits[:, :2]
            
            # Compute loss
            if self.use_class_weights and self.class_weights is not None:
                loss = F.cross_entropy(keep_cut_logits, labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(keep_cut_logits, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                nan_batches += 1
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients and skip if found
            has_nan_grad = False
            for param in self.agent.policy_net.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                nan_batches += 1
                continue
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * len(labels)
            preds = keep_cut_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        if nan_batches > 0:
            logger.warning(f"Skipped {nan_batches} batches due to NaN values")
        
        if scheduler is not None:
            scheduler.step()
        
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "lr": self.optimizer.param_groups[0]["lr"]
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on dataset.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary of metrics
        """
        self.agent.policy_net.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class metrics
        class_correct = [0, 0]
        class_total = [0, 0]
        
        for batch in dataloader:
            obs = batch["observation"].to(self.device)
            labels = batch["label"].to(self.device)
            
            logits, _ = self.agent.policy_net(obs)
            keep_cut_logits = logits[:, :2]
            
            loss = F.cross_entropy(keep_cut_logits, labels)
            
            total_loss += loss.item() * len(labels)
            preds = keep_cut_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            # Per-class
            for c in [0, 1]:
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c] += mask.sum().item()
        
        keep_acc = class_correct[0] / max(class_total[0], 1)
        cut_acc = class_correct[1] / max(class_total[1], 1)
        
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "keep_accuracy": keep_acc,
            "cut_accuracy": cut_acc
        }
    
    def save_checkpoint(self, path: str, epoch: int, best_acc: float):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "best_accuracy": best_acc,
            "policy_state_dict": self.agent.policy_net.state_dict(),
            "value_state_dict": self.agent.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved BC checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        if "value_state_dict" in checkpoint:
            self.agent.value_net.load_state_dict(checkpoint["value_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint


def train_behavioral_cloning(
    config: Config,
    data_dir: str,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    lr_decay: str = "cosine",
    checkpoint_path: Optional[str] = None,
    output_dir: str = "models"
):
    """Train policy via behavioral cloning.
    
    Args:
        config: Configuration
        data_dir: Training data directory
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Initial learning rate
        lr_decay: LR decay type ("cosine", "plateau", or "none")
        checkpoint_path: Optional checkpoint to resume from
        output_dir: Output directory for checkpoints
    """
    device = torch.device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build dataset
    logger.info("Building BC dataset...")
    bc_dataset = BehavioralCloningDataset(data_dir, config)
    
    if len(bc_dataset) == 0:
        logger.error("No training samples! Check that training data has edit_labels.")
        return
    
    # Split into train/val
    n_samples = len(bc_dataset)
    n_val = max(int(0.1 * n_samples), 1)
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        bc_dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Train: {n_train} samples, Val: {n_val} samples")
    
    # Get observation dimension from dataset
    sample = bc_dataset[0]
    input_dim = sample["observation"].shape[0]
    logger.info(f"Observation dimension: {input_dim}")
    
    # Initialize agent
    agent = Agent(config, input_dim=input_dim, n_actions=30)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent.value_net.load_state_dict(ckpt["value_state_dict"])
    
    # Initialize trainer
    trainer = BehavioralCloningTrainer(config, agent, lr=lr)
    trainer.compute_class_weights(bc_dataset)
    
    # LR scheduler
    if lr_decay == "cosine":
        scheduler = CosineAnnealingLR(trainer.optimizer, T_max=n_epochs, eta_min=1e-6)
    elif lr_decay == "plateau":
        scheduler = ReduceLROnPlateau(trainer.optimizer, mode='max', factor=0.5, patience=10)
    else:
        scheduler = None
    
    # Training loop
    best_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting BC training for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(
            train_loader, 
            scheduler if lr_decay == "cosine" else None
        )
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Update plateau scheduler if used
        if lr_decay == "plateau" and scheduler is not None:
            scheduler.step(val_metrics["accuracy"])
        
        # Log
        logger.info(
            f"Epoch {epoch}/{n_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f} "
            f"(KEEP: {val_metrics['keep_accuracy']:.3f}, CUT: {val_metrics['cut_accuracy']:.3f}) | "
            f"LR: {train_metrics['lr']:.6f}"
        )
        
        # Save best
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            trainer.save_checkpoint(
                str(output_path / "bc_checkpoint_best.pt"),
                epoch, best_acc
            )
        
        # Save periodic checkpoints
        if epoch % 20 == 0:
            trainer.save_checkpoint(
                str(output_path / f"bc_checkpoint_epoch_{epoch}.pt"),
                epoch, val_metrics["accuracy"]
            )
    
    # Save final
    trainer.save_checkpoint(
        str(output_path / "bc_checkpoint_final.pt"),
        n_epochs, val_metrics["accuracy"]
    )
    
    logger.info(f"BC Training complete! Best accuracy: {best_acc:.4f}")
    logger.info(f"Best checkpoint: {output_path / 'bc_checkpoint_best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning Pretraining")
    parser.add_argument("--data_dir", type=str, default="./training_data",
                       help="Training data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_decay", type=str, default="cosine",
                       choices=["cosine", "plateau", "none"], help="LR decay type")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint to initialize from")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Output directory")
    
    args = parser.parse_args()
    
    config = get_default_config()
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_behavioral_cloning(
        config=config,
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
