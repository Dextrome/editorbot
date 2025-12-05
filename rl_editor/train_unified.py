"""Unified training workflow: Behavioral Cloning warmup + RL fine-tuning.

This combines BC pre-training and PPO fine-tuning into a single workflow:
1. Phase 1 (BC): Supervised learning on human edit labels to learn KEEP/CUT decisions
2. Phase 2 (RL): PPO fine-tuning to optimize edit quality with learned reward

Usage:
    python -m rl_editor.train_unified --data_dir ./training_data --bc_epochs 50 --rl_epochs 500
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .config import Config, get_default_config
from .data import PairedAudioDataset
from .agent import Agent
from .state import AudioState, EditHistory, StateRepresentation
from .train_parallel import ParallelPPOTrainer, train_parallel
from .logging_utils import TrainingLogger, create_logger

logger = logging.getLogger(__name__)


class BCDataset(torch.utils.data.Dataset):
    """Dataset for behavioral cloning - extracts (state, KEEP/CUT label) pairs."""
    
    def __init__(
        self,
        data_dir: str,
        config: Config,
        max_beats: int = 500,
    ):
        self.config = config
        self.max_beats = max_beats
        
        # Load paired audio dataset
        self.audio_dataset = PairedAudioDataset(
            data_dir, config, include_reference=False
        )
        
        # State representation - will set actual feature dim from data
        self.state_rep = StateRepresentation(config)
        
        # Extract samples
        self.samples = []
        self._build_samples()
        
        logger.info(f"Built {len(self.samples)} BC samples from {len(self.audio_dataset)} tracks")
    
    def _build_samples(self):
        """Extract per-beat samples from dataset."""
        beat_feature_dim_set = False
        
        for idx in range(len(self.audio_dataset)):
            item = self.audio_dataset[idx]
            
            if "edit_labels" not in item or item["edit_labels"] is None:
                continue
            
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
            
            # Truncate
            n_beats = min(len(beat_times), self.max_beats)
            beat_times = beat_times[:n_beats]
            beat_features = beat_features[:n_beats]
            edit_labels = edit_labels[:n_beats]
            
            total_duration = beat_times[-1] if len(beat_times) > 0 else 1.0
            
            for beat_idx in range(n_beats):
                audio_state = AudioState(
                    beat_index=beat_idx,
                    beat_times=beat_times,
                    beat_features=beat_features,
                    tempo=tempo
                )
                
                edit_history = EditHistory()
                for prev_idx in range(beat_idx):
                    if edit_labels[prev_idx] > 0.5:
                        edit_history.kept_beats.append(prev_idx)
                    else:
                        edit_history.cut_beats.append(prev_idx)
                
                kept_duration = sum(
                    beat_times[i+1] - beat_times[i] if i+1 < len(beat_times) else 0
                    for i in edit_history.kept_beats
                )
                target_duration = total_duration * self.config.reward.target_keep_ratio
                remaining_duration = max(0.0, target_duration - kept_duration)
                
                obs = self.state_rep.construct_observation(
                    audio_state, edit_history,
                    remaining_duration=remaining_duration,
                    total_duration=total_duration
                )
                
                # Label: 0=KEEP, 1=CUT (convert from edit_labels where KEEP=1)
                label = 0 if edit_labels[beat_idx] > 0.5 else 1
                
                self.samples.append({
                    "observation": torch.from_numpy(obs).float(),
                    "label": torch.tensor(label).long()
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def run_bc_phase(
    config: Config,
    agent: Agent,
    data_dir: str,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: torch.device = None,
) -> Dict[str, float]:
    """Run behavioral cloning phase.
    
    Args:
        config: Configuration
        agent: Agent to train
        data_dir: Training data directory
        n_epochs: Number of BC epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
        
    Returns:
        Training metrics
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Behavioral Cloning")
    logger.info("=" * 60)
    
    if device is None:
        device = agent.device
    
    # Build dataset
    logger.info("Building BC dataset...")
    bc_dataset = BCDataset(data_dir, config)
    
    if len(bc_dataset) == 0:
        logger.error("No BC samples found!")
        return {"bc_accuracy": 0.0}
    
    # Split train/val
    n_val = max(int(0.1 * len(bc_dataset)), 1)
    n_train = len(bc_dataset) - n_val
    train_dataset, val_dataset = random_split(bc_dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train: {n_train}, Val: {n_val} samples")
    
    # Compute class weights
    labels = [s["label"].item() for s in bc_dataset.samples]
    counts = np.bincount(labels, minlength=2)
    weights = len(labels) / (2 * counts + 1e-8)
    weights = np.clip(weights, 0.2, 5.0)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights: KEEP={weights[0]:.3f}, CUT={weights[1]:.3f}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(agent.policy_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_acc = 0.0
    
    for epoch in range(1, n_epochs + 1):
        # Train
        agent.policy_net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            obs = batch["observation"].to(device)
            labels = batch["label"].to(device)
            
            if torch.isnan(obs).any():
                continue
            
            logits, _ = agent.policy_net(obs)
            keep_cut_logits = logits[:, :2]
            
            if torch.isnan(logits).any():
                continue
            
            loss = F.cross_entropy(keep_cut_logits, labels, weight=class_weights)
            
            if torch.isnan(loss):
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = keep_cut_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
        
        # Validate
        agent.policy_net.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["observation"].to(device)
                labels = batch["label"].to(device)
                
                logits, _ = agent.policy_net(obs)
                keep_cut_logits = logits[:, :2]
                preds = keep_cut_logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        logger.info(
            f"BC Epoch {epoch}/{n_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
    
    logger.info(f"BC Phase complete. Best accuracy: {best_acc:.3f}")
    return {"bc_accuracy": best_acc}


def run_rl_phase(
    config: Config,
    agent: Agent,
    data_dir: str,
    n_epochs: int = 500,
    n_envs: int = 4,
    steps_per_epoch: int = 2048,
    save_dir: str = "./models",
) -> Dict[str, float]:
    """Run RL fine-tuning phase.
    
    Args:
        config: Configuration
        agent: Pre-trained agent from BC
        data_dir: Training data directory
        n_epochs: Number of RL epochs
        n_envs: Number of parallel environments
        steps_per_epoch: Steps per epoch
        save_dir: Directory to save checkpoints
        
    Returns:
        Training metrics
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: RL Fine-tuning (PPO)")
    logger.info("=" * 60)
    
    # Create trainer with the pre-trained agent
    # Use full action space so agent can learn LOOP/CROSSFADE/REORDER
    trainer = ParallelPPOTrainer(config, n_envs=n_envs, total_epochs=n_epochs, keep_cut_only=False)
    
    # Copy pre-trained weights to trainer's agent (once initialized)
    # We'll do this after the first rollout initializes the agent
    pretrained_policy_state = agent.policy_net.state_dict()
    pretrained_value_state = agent.value_net.state_dict()
    
    # Load dataset
    dataset = PairedAudioDataset(
        data_dir, config,
        cache_dir=config.data.cache_dir,
        include_reference=True,
    )
    
    if len(dataset) == 0:
        logger.error(f"No training data found in {data_dir}")
        return {"rl_reward": 0.0}
    
    logger.info(f"Loaded {len(dataset)} training samples")
    
    # Training loop
    start_time = time.time()
    best_reward = -np.inf
    
    # Reward collapse protection settings
    collapse_threshold = -20.0  # Reward below this = collapsed
    collapse_patience = 30  # Epochs of collapse before reverting
    revert_count = 0  # Track number of reverts
    max_reverts = 3  # Max times to revert before early stopping
    collapse_counter = 0  # Consecutive epochs below threshold
    recent_rewards = []  # Track recent rewards for smoothing
    reward_window = 10  # Window for reward averaging
    best_checkpoint_state = None  # Store best checkpoint in memory
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Sample audio states
        MAX_BEATS = 500
        audio_states = []
        for _ in range(n_envs):
            idx = np.random.randint(len(dataset))
            item = dataset[idx]
            
            raw_data = item["raw"]
            beat_times = raw_data["beat_times"].numpy()
            beat_features = raw_data["beat_features"].numpy()
            
            edit_labels = item.get("edit_labels")
            if edit_labels is not None:
                edit_labels = edit_labels.numpy() if hasattr(edit_labels, 'numpy') else np.array(edit_labels)
            
            if len(beat_times) > MAX_BEATS:
                start_idx = np.random.randint(0, len(beat_times) - MAX_BEATS)
                beat_times = beat_times[start_idx:start_idx + MAX_BEATS]
                beat_features = beat_features[start_idx:start_idx + MAX_BEATS]
                if edit_labels is not None and len(edit_labels) > MAX_BEATS:
                    edit_labels = edit_labels[start_idx:start_idx + MAX_BEATS]
                beat_times = beat_times - beat_times[0]
            
            audio_state = AudioState(
                beat_index=0,
                beat_times=beat_times,
                beat_features=beat_features,
                tempo=raw_data["tempo"].item(),
                target_labels=edit_labels,
            )
            audio_states.append(audio_state)
        
        # Collect rollouts
        rollout_data = trainer.collect_rollouts_parallel(audio_states, steps_per_epoch // n_envs)
        
        # Initialize trainer's agent with pre-trained weights on first epoch
        if epoch == 0 and trainer.agent is not None:
            logger.info("Loading BC pre-trained weights into RL agent...")
            trainer.agent.policy_net.load_state_dict(pretrained_policy_state)
            trainer.agent.value_net.load_state_dict(pretrained_value_state)
        
        # Update
        metrics = trainer.update(rollout_data)
        trainer.step_schedulers()
        
        epoch_time = time.time() - epoch_start
        current_lr = trainer.get_current_lr()
        current_reward = metrics['episode_reward']
        
        # Track recent rewards
        recent_rewards.append(current_reward)
        if len(recent_rewards) > reward_window:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards)
        
        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"RL Epoch {epoch + 1}/{n_epochs} | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"Reward: {current_reward:.2f} (avg: {avg_reward:.2f}) | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        # Save best and store checkpoint state in memory
        if current_reward > best_reward:
            best_reward = current_reward
            trainer.best_reward = best_reward
            best_file = Path(save_dir) / "checkpoint_best.pt"
            trainer.save_checkpoint(str(best_file))
            collapse_counter = 0  # Reset collapse counter on new best
            # Store best state in memory for fast revert
            best_checkpoint_state = {
                "policy_state_dict": {k: v.clone() for k, v in trainer.agent.policy_net.state_dict().items()},
                "value_state_dict": {k: v.clone() for k, v in trainer.agent.value_net.state_dict().items()},
            }
        
        # === REWARD COLLAPSE PROTECTION ===
        if avg_reward < collapse_threshold:
            collapse_counter += 1
            if collapse_counter >= collapse_patience:
                revert_count += 1
                if revert_count > max_reverts:
                    logger.warning(f"Max reverts ({max_reverts}) reached. Early stopping at epoch {epoch + 1}.")
                    break
                
                logger.warning(
                    f"REWARD COLLAPSE DETECTED! Avg reward {avg_reward:.2f} < {collapse_threshold} "
                    f"for {collapse_patience} epochs. Reverting to best checkpoint (revert {revert_count}/{max_reverts})..."
                )
                
                # Revert to best checkpoint
                if best_checkpoint_state is not None:
                    trainer.agent.policy_net.load_state_dict(best_checkpoint_state["policy_state_dict"])
                    trainer.agent.value_net.load_state_dict(best_checkpoint_state["value_state_dict"])
                    logger.info(f"Reverted to best checkpoint (reward {best_reward:.2f})")
                else:
                    # Load from file
                    best_file = Path(save_dir) / "checkpoint_best.pt"
                    if best_file.exists():
                        trainer.load_checkpoint(str(best_file))
                        logger.info(f"Loaded best checkpoint from {best_file}")
                
                collapse_counter = 0
                recent_rewards = []  # Reset reward tracking
        else:
            collapse_counter = max(0, collapse_counter - 1)  # Decay counter when doing well
        
        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint_file = Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_file))
    
    # Save final
    final_file = Path(save_dir) / "checkpoint_final.pt"
    trainer.save_checkpoint(str(final_file))
    
    total_time = time.time() - start_time
    logger.info(f"RL Phase complete in {total_time/60:.1f} minutes. Best reward: {best_reward:.2f}")
    
    return {"rl_reward": best_reward}


def train_unified(
    config: Config,
    data_dir: str,
    bc_epochs: int = 50,
    rl_epochs: int = 500,
    bc_lr: float = 1e-4,
    bc_batch_size: int = 64,
    n_envs: int = 4,
    steps_per_epoch: int = 2048,
    save_dir: str = "./models",
    skip_bc: bool = False,
    bc_checkpoint: Optional[str] = None,
):
    """Run unified BC + RL training workflow.
    
    Args:
        config: Configuration
        data_dir: Training data directory
        bc_epochs: Number of BC warmup epochs
        rl_epochs: Number of RL fine-tuning epochs
        bc_lr: BC learning rate
        bc_batch_size: BC batch size
        n_envs: Number of parallel RL environments
        steps_per_epoch: RL steps per epoch
        save_dir: Directory to save checkpoints
        skip_bc: Skip BC phase (use existing checkpoint)
        bc_checkpoint: Path to BC checkpoint to load
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get input dimension from a sample
    temp_dataset = PairedAudioDataset(data_dir, config, include_reference=False)
    if len(temp_dataset) == 0:
        logger.error("No training data found!")
        return
    
    # Get observation dimension
    item = temp_dataset[0]
    raw_data = item["raw"]
    beat_times = raw_data["beat_times"].numpy()[:500]
    beat_features = raw_data["beat_features"].numpy()[:500]
    
    state_rep = StateRepresentation(config)
    audio_state = AudioState(
        beat_index=0,
        beat_times=beat_times,
        beat_features=beat_features,
        tempo=raw_data["tempo"].item()
    )
    obs = state_rep.construct_observation(audio_state, EditHistory(), 100.0, 300.0)
    input_dim = len(obs)
    logger.info(f"Observation dimension: {input_dim}")
    
    # Initialize agent - use 9 actions (new simplified action space)
    n_actions = 9  # KEEP(1) + CUT(1) + LOOP(3) + REORDER(4)
    agent = Agent(config, input_dim=input_dim, n_actions=n_actions)
    logger.info(f"Using {n_actions}-action space")
    
    # Load BC checkpoint if provided
    if bc_checkpoint and Path(bc_checkpoint).exists():
        logger.info(f"Loading BC checkpoint: {bc_checkpoint}")
        ckpt = torch.load(bc_checkpoint, map_location=device, weights_only=False)
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
        agent.value_net.load_state_dict(ckpt["value_state_dict"])
        skip_bc = True
    
    # Phase 1: Behavioral Cloning
    if not skip_bc and bc_epochs > 0:
        bc_metrics = run_bc_phase(
            config, agent, data_dir,
            n_epochs=bc_epochs,
            batch_size=bc_batch_size,
            lr=bc_lr,
            device=device,
        )
        
        # Save BC checkpoint
        bc_file = save_path / f"bc_checkpoint_{timestamp}.pt"
        torch.save({
            "policy_state_dict": agent.policy_net.state_dict(),
            "value_state_dict": agent.value_net.state_dict(),
            "bc_accuracy": bc_metrics["bc_accuracy"],
        }, bc_file)
        logger.info(f"Saved BC checkpoint: {bc_file}")
    else:
        logger.info("Skipping BC phase (using existing weights)")
    
    # Phase 2: RL Fine-tuning
    if rl_epochs > 0:
        rl_metrics = run_rl_phase(
            config, agent, data_dir,
            n_epochs=rl_epochs,
            n_envs=n_envs,
            steps_per_epoch=steps_per_epoch,
            save_dir=str(save_path),
        )
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Unified BC + RL Training")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Training data directory")
    parser.add_argument("--bc_epochs", type=int, default=100, help="BC warmup epochs")
    parser.add_argument("--rl_epochs", type=int, default=500, help="RL fine-tuning epochs")
    parser.add_argument("--bc_lr", type=float, default=1e-4, help="BC learning rate")
    parser.add_argument("--bc_batch_size", type=int, default=64, help="BC batch size")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel RL environments")
    parser.add_argument("--steps", type=int, default=2048, help="RL steps per epoch")
    parser.add_argument("--save_dir", type=str, default="./models", help="Save directory")
    parser.add_argument("--skip_bc", action="store_true", help="Skip BC phase")
    parser.add_argument("--bc_checkpoint", type=str, default=None, help="BC checkpoint to load")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = get_default_config()
    
    train_unified(
        config=config,
        data_dir=args.data_dir,
        bc_epochs=args.bc_epochs,
        rl_epochs=args.rl_epochs,
        bc_lr=args.bc_lr,
        bc_batch_size=args.bc_batch_size,
        n_envs=args.n_envs,
        steps_per_epoch=args.steps,
        save_dir=args.save_dir,
        skip_bc=args.skip_bc,
        bc_checkpoint=args.bc_checkpoint,
    )


if __name__ == "__main__":
    main()
