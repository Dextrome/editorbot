"""Main entry point for RL audio editor training and inference.

Usage:
    python -m rl_editor.main train --data_dir ./training_data
    python -m rl_editor.main evaluate --checkpoint ./models/policy_best.pt --audio ./test.wav
    python -m rl_editor.main edit --checkpoint ./models/policy_best.pt --audio ./raw.wav --output ./edited.wav
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config, get_default_config
from .trainer import PPOTrainer
from .agent import Agent
from .environment import AudioEditingEnv
from .data import AudioDataset, create_dataloader
from .evaluation import Evaluator
from .state import AudioState
from .utils import load_audio, save_audio, setup_logging


logger = logging.getLogger(__name__)


def train(
    config: Config,
    data_dir: str,
    checkpoint_path: Optional[str] = None,
    n_epochs: int = 100,
) -> None:
    """Train the RL agent.

    Args:
        config: Configuration object
        data_dir: Directory containing training data
        checkpoint_path: Optional path to resume from checkpoint
        n_epochs: Number of training epochs
    """
    logger.info("Starting training...")
    
    # Setup dataset
    dataset = AudioDataset(data_dir, config, split="train")
    if len(dataset) == 0:
        logger.error(f"No training data found in {data_dir}")
        return
    
    logger.info(f"Loaded {len(dataset)} training samples")
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # Need to initialize with first sample to create agent
        item = dataset[0]
        audio_state = AudioState(
            beat_index=0,
            beat_times=item["beat_times"].numpy(),
            beat_features=item["beat_features"].numpy(),
            tempo=item["tempo"].item()
        )
        trainer.initialize_env_and_agent(audio_state)
        trainer.load_checkpoint(checkpoint_path)
    
    # Training loop
    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")
        
        # Sample random audio from dataset
        idx = np.random.randint(len(dataset))
        item = dataset[idx]
        
        # Create audio state
        audio_state = AudioState(
            beat_index=0,
            beat_times=item["beat_times"].numpy(),
            beat_features=item["beat_features"].numpy(),
            tempo=item["tempo"].item()
        )
        
        # Initialize or reinitialize environment
        trainer.initialize_env_and_agent(audio_state)
        
        # Collect rollouts and update
        rollout_data = trainer.collect_rollouts(n_steps=config.ppo.n_steps)
        metrics = trainer.update(rollout_data)
        
        logger.info(
            f"Policy Loss: {metrics['policy_loss']:.4f}, "
            f"Value Loss: {metrics['value_loss']:.4f}, "
            f"Episode Reward: {metrics['episode_reward']:.2f}"
        )
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_file = Path(config.training.save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_file))
            logger.info(f"Saved checkpoint to {checkpoint_file}")
    
    # Save final checkpoint
    final_checkpoint = Path(config.training.save_dir) / "checkpoint_final.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    logger.info(f"Training complete. Final checkpoint saved to {final_checkpoint}")


def evaluate(
    config: Config,
    checkpoint_path: str,
    data_dir: str,
    n_episodes: int = 10,
) -> None:
    """Evaluate trained agent.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing test data
        n_episodes: Number of evaluation episodes
    """
    logger.info(f"Evaluating agent from {checkpoint_path}...")
    
    # Load dataset
    dataset = AudioDataset(data_dir, config, split="test")
    if len(dataset) == 0:
        logger.error(f"No test data found in {data_dir}")
        return
    
    # Get sample to determine dimensions
    item = dataset[0]
    audio_state = AudioState(
        beat_index=0,
        beat_times=item["beat_times"].numpy(),
        beat_features=item["beat_features"].numpy(),
        tempo=item["tempo"].item()
    )
    
    # Create environment to get dimensions
    env = AudioEditingEnv(config, audio_state)
    obs, _ = env.reset()
    input_dim = obs.shape[0]
    n_actions = env.action_space.n_discrete_actions
    
    # Create and load agent
    agent = Agent(config, input_dim, n_actions)
    checkpoint = torch.load(checkpoint_path, map_location=config.training.device)
    agent.policy_net.load_state_dict(checkpoint["agent_state_dict"]["policy_net"])
    agent.value_net.load_state_dict(checkpoint["agent_state_dict"]["value_net"])
    
    # Evaluate
    evaluator = Evaluator(config, agent)
    metrics = evaluator.evaluate_dataset(dataset, n_episodes=n_episodes)
    
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")


def edit_audio(
    config: Config,
    checkpoint_path: str,
    input_audio: str,
    output_audio: str,
) -> None:
    """Edit audio file using trained agent.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        input_audio: Path to input audio file
        output_audio: Path to save edited audio
    """
    logger.info(f"Editing {input_audio} -> {output_audio}")
    
    # Load audio
    y, sr = load_audio(input_audio, sr=config.audio.sample_rate)
    
    # Extract features (simplified - use data module for full extraction)
    import librosa
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Compute beat features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    beat_features = []
    frames = librosa.time_to_frames(beat_times, sr=sr)
    frames = np.concatenate(([0], frames, [len(onset_env)]))
    
    for i in range(len(beats)):
        start = frames[i]
        end = frames[i+1]
        if start >= len(onset_env): break
        if end > len(onset_env): end = len(onset_env)
        if start == end: end = start + 1
        
        b_onset = np.mean(onset_env[start:end])
        b_centroid = np.mean(centroid[start:end])
        b_zcr = np.mean(zcr[start:end])
        beat_features.append([b_onset, b_centroid, b_zcr])
    
    beat_features = np.array(beat_features, dtype=np.float32)
    if len(beat_features) < len(beats):
        pad_width = len(beats) - len(beat_features)
        beat_features = np.pad(beat_features, ((0, pad_width), (0, 0)), mode='edge')
    
    # Create audio state
    audio_state = AudioState(
        beat_index=0,
        beat_times=beat_times,
        beat_features=beat_features,
        tempo=float(tempo)
    )
    
    # Create environment
    env = AudioEditingEnv(config, audio_state)
    obs, _ = env.reset()
    
    # Load agent
    input_dim = obs.shape[0]
    n_actions = env.action_space.n_discrete_actions
    agent = Agent(config, input_dim, n_actions)
    checkpoint = torch.load(checkpoint_path, map_location=config.training.device)
    agent.policy_net.load_state_dict(checkpoint["agent_state_dict"]["policy_net"])
    agent.eval()
    
    # Run agent to get edit decisions
    device = torch.device(config.training.device)
    edit_decisions = []
    
    done = False
    while not done:
        state_tensor = torch.from_numpy(obs).float().to(device)
        action, _ = agent.select_action(state_tensor, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        edit_decisions.append(action)
        done = terminated or truncated
    
    logger.info(f"Generated {len(edit_decisions)} edit decisions")
    
    # Apply edits to audio (simplified - concatenate kept beats)
    # In a full implementation, this would handle LOOP, CROSSFADE, REORDER
    kept_beats = env.edit_history.kept_beats
    
    if not kept_beats:
        logger.warning("No beats kept, outputting silence")
        edited_audio = np.zeros(sr)  # 1 second of silence
    else:
        # Concatenate kept beat segments
        segments = []
        for beat_idx in sorted(kept_beats):
            if beat_idx < len(beat_times) - 1:
                start_sample = int(beat_times[beat_idx] * sr)
                end_sample = int(beat_times[beat_idx + 1] * sr)
                segments.append(y[start_sample:end_sample])
        
        if segments:
            edited_audio = np.concatenate(segments)
        else:
            edited_audio = y  # Fallback to original
    
    # Save output
    save_audio(edited_audio, output_audio, sr=sr)
    logger.info(f"Saved edited audio to {output_audio}")
    logger.info(f"Original duration: {len(y)/sr:.2f}s, Edited duration: {len(edited_audio)/sr:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RL Audio Editor")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the RL agent")
    train_parser.add_argument("--data_dir", type=str, default="./training_data/train",
                             help="Directory containing training data")
    train_parser.add_argument("--checkpoint", type=str, default=None,
                             help="Path to checkpoint to resume from")
    train_parser.add_argument("--epochs", type=int, default=100,
                             help="Number of training epochs")
    train_parser.add_argument("--device", type=str, default="cuda",
                             help="Device to use (cuda/cpu)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Path to model checkpoint")
    eval_parser.add_argument("--data_dir", type=str, default="./training_data/test",
                            help="Directory containing test data")
    eval_parser.add_argument("--n_episodes", type=int, default=10,
                            help="Number of evaluation episodes")
    eval_parser.add_argument("--device", type=str, default="cuda",
                            help="Device to use (cuda/cpu)")
    
    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Edit audio file")
    edit_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Path to model checkpoint")
    edit_parser.add_argument("--audio", type=str, required=True,
                            help="Path to input audio file")
    edit_parser.add_argument("--output", type=str, required=True,
                            help="Path to save edited audio")
    edit_parser.add_argument("--device", type=str, default="cuda",
                            help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging("./logs", "rl_editor")
    
    # Create config
    config = get_default_config()
    if hasattr(args, 'device'):
        config.training.device = args.device
    
    # Execute command
    if args.command == "train":
        train(config, args.data_dir, args.checkpoint, args.epochs)
    elif args.command == "evaluate":
        evaluate(config, args.checkpoint, args.data_dir, args.n_episodes)
    elif args.command == "edit":
        edit_audio(config, args.checkpoint, args.audio, args.output)


if __name__ == "__main__":
    main()
