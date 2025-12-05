"""Evaluation and visualization module."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf

from rl_editor.agent import Agent
from rl_editor.config import Config
from rl_editor.environment import AudioEditingEnv
from rl_editor.state import AudioState
from rl_editor.data import AudioDataset

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for RL agent."""

    def __init__(self, config: Config, agent: Agent, output_dir: str = "./output/eval"):
        """Initialize evaluator.

        Args:
            config: Configuration object
            agent: Trained agent
            output_dir: Directory to save evaluation outputs
        """
        self.config = config
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_dataset(self, dataset: AudioDataset, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent on a dataset.

        Args:
            dataset: Test dataset
            n_episodes: Number of episodes to run (if None, run all)

        Returns:
            Dictionary of metrics
        """
        self.agent.eval()
        metrics = {
            "rewards": [],
            "lengths": [],
            "keep_ratios": []
        }

        n = min(len(dataset), n_episodes) if n_episodes else len(dataset)
        
        for i in range(n):
            item = dataset[i]
            # Create AudioState from dataset item
            # Note: This assumes dataset returns processed features compatible with AudioState
            # We might need to adjust AudioState creation if dataset format differs
            audio_state = AudioState(
                beat_index=0,
                beat_times=item["beat_times"].numpy(),
                beat_features=item["beat_features"].numpy(),
                tempo=item["tempo"].item()
            )
            
            env = AudioEditingEnv(self.config, audio_state)
            
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            
            # Track actions for visualization
            actions_taken = []
            
            while not done:
                state_tensor = torch.from_numpy(obs).float().to(self.agent.device)
                action, _ = self.agent.select_action(state_tensor, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                actions_taken.append(action)
                
                done = terminated or truncated
            
            metrics["rewards"].append(episode_reward)
            metrics["lengths"].append(steps)
            
            # Calculate keep ratio (approximate based on actions)
            # Assuming action 0 is KEEP (need to verify with ActionSpace)
            # This is a simplification
            keep_actions = [a for a in actions_taken if a == 0] # 0 is usually KEEP
            metrics["keep_ratios"].append(len(keep_actions) / steps if steps > 0 else 0)
            
            # Visualize first few episodes
            if i < 3:
                self.visualize_episode(audio_state, actions_taken, f"eval_{i}")

        return {
            "mean_reward": float(np.mean(metrics["rewards"])),
            "std_reward": float(np.std(metrics["rewards"])),
            "mean_length": float(np.mean(metrics["lengths"])),
            "mean_keep_ratio": float(np.mean(metrics["keep_ratios"]))
        }

    def visualize_episode(self, audio_state: AudioState, actions: List[int], name: str):
        """Visualize an editing episode.

        Args:
            audio_state: Initial audio state
            actions: List of actions taken
            name: Name for the plot file
        """
        plt.figure(figsize=(12, 6))
        
        # Plot beat times
        times = audio_state.beat_times
        n_beats = len(times)
        
        # Plot actions (simplified)
        # We need to map actions to something visual
        # For now, just plot the action index
        plt.step(range(len(actions)), actions, where='post', label='Action')
        
        plt.title(f"Episode Visualization: {name}")
        plt.xlabel("Step")
        plt.ylabel("Action Index")
        plt.legend()
        plt.grid(True)
        
        save_path = self.output_dir / f"{name}.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved visualization to {save_path}")

