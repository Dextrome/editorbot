"""Gymnasium-compatible environment for RL-based audio editing."""

import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional, Tuple, Dict
import numpy as np
import logging

from .config import Config
from .actions import ActionSpace, Action
from .state import AudioState, EditHistory, StateRepresentation
from .reward import RewardCalculator, RewardComponents

logger = logging.getLogger(__name__)


class AudioEditingEnv(gym.Env):
    """RL environment for audio editing.

    Observation space: State representation (audio features + edit history)
    Action space: Discrete actions (KEEP, CUT, LOOP, REORDER) 
                  Crossfades are automatic at edit boundaries.
    Reward: Sparse (human feedback) + Dense (automatic metrics)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Config, audio_state: Optional[AudioState] = None) -> None:
        """Initialize environment.

        Args:
            config: Configuration object
            audio_state: Audio state (can be provided or set via reset())
        """
        super().__init__()
        self.config = config
        self.audio_state = audio_state
        self.action_space: Optional[ActionSpace] = None
        self.observation_space: Optional[spaces.Box] = None
        self.state_rep: Optional[StateRepresentation] = None
        self.reward_calc: Optional[RewardCalculator] = None
        self.edit_history = EditHistory()
        
        # Tracking
        self.current_step = 0
        self.max_steps = 1000
        self.episode_actions: list = []
        self.episode_rewards: list = []
        self.last_reward_components: Optional[RewardComponents] = None
        
        # Seed
        self.seed_value = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment.

        Args:
            seed: Random seed
            options: Additional options (can include 'audio_state')

        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
            self.seed_value = seed

        # Set up audio state if provided in options
        if options and "audio_state" in options:
            self.audio_state = options["audio_state"]

        if self.audio_state is None:
            raise ValueError("audio_state must be provided (either at init or in reset options)")

        # Initialize components
        if self.state_rep is None:
            self.state_rep = StateRepresentation(self.config)
            
        # Update state representation with actual beat feature dimension from audio state
        if self.audio_state.beat_features is not None and self.audio_state.beat_features.ndim > 1:
            actual_dim = self.audio_state.beat_features.shape[1]
            if actual_dim != self.state_rep.beat_feature_dim:
                self.state_rep.set_beat_feature_dim(actual_dim)
                # Reset observation space since dimension changed
                self.observation_space = None
                
        if self.reward_calc is None:
            self.reward_calc = RewardCalculator(self.config)

        # Initialize action space
        n_beats = len(self.audio_state.beat_times)
        if self.action_space is None or self.action_space.n_beats != n_beats:
            self.action_space = ActionSpace(
                n_beats=n_beats,
                max_loop_times=self.config.action_space.max_loop_times,
                default_crossfade_ms=getattr(self.config.action_space, 'default_crossfade_ms', 50),
            )

        # Initialize observation space
        if self.observation_space is None:
            obs_dim = self.state_rep.feature_dim
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

        # Reset tracking
        self.edit_history = EditHistory()
        self.current_step = 0
        self.episode_actions = []
        self.episode_rewards = []
        self.last_reward_components = None

        # Compute initial observation
        obs = self.state_rep.construct_observation(
            self.audio_state,
            self.edit_history,
            remaining_duration=self._get_remaining_duration(),
            total_duration=self.audio_state.beat_times[-1] if len(self.audio_state.beat_times) > 0 else 1.0,
        )

        info = {
            "step": self.current_step,
            "n_beats": n_beats,
            "beat_index": self.audio_state.beat_index,
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in environment.

        Args:
            action: Discrete action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.audio_state is None or self.action_space is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Decode action (pass current beat for relative action decoding)
        try:
            action_obj = self.action_space.decode_action(action, self.audio_state.beat_index)
        except ValueError as e:
            logger.error(f"Invalid action: {e}")
            return self._get_observation(), 0.0, False, True, {}

        # Update edit history based on action
        self._apply_action(action_obj)

        # Move to next beat
        total_beats = len(self.audio_state.beat_times)
        self.audio_state.beat_index = min(self.audio_state.beat_index + 1, total_beats - 1)

        # Compute reward (dense reward at each step)
        reward = self._compute_step_reward(action_obj)
        self.episode_rewards.append(reward)

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        # Get next observation
        obs = self._get_observation()

        # Info dict
        info = {
            "step": self.current_step,
            "action": action,
            "action_type": action_obj.action_type.name,
            "beat_index": self.audio_state.beat_index,
            "n_beats_edited": len(self.edit_history.get_edited_beats()),
            "episode_reward": sum(self.episode_rewards),
        }
        if self.last_reward_components:
            info["reward_components"] = self.last_reward_components.to_dict()

        self.current_step += 1

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: Action) -> None:
        """Apply action to edit history.

        Args:
            action: Action object
        """
        from .actions import (
            KeepAction,
            CutAction,
            LoopAction,
            ReorderAction,
        )

        if isinstance(action, KeepAction):
            self.edit_history.add_keep(action.beat_index)
        elif isinstance(action, CutAction):
            self.edit_history.add_cut(action.beat_index)
        elif isinstance(action, LoopAction):
            self.edit_history.add_loop(action.beat_index, action.n_times)
        elif isinstance(action, ReorderAction):
            self.edit_history.add_reorder(action.beat_index, action.target_position)

    def _compute_step_reward(self, action: Action) -> float:
        """Compute reward for a step - supervised by ground truth labels.

        The reward function enforces:
        1. If target says CUT (label < 0.5), the agent MUST cut - no creative action can save it
        2. If target says KEEP (label >= 0.5), agent can KEEP or use creative actions (LOOP/REORDER)
        3. Creative actions (LOOP) are encouraged for high-quality beats
        4. Duration penalty if projected output exceeds target max duration

        Args:
            action: Action taken

        Returns:
            Step reward
        """
        from .actions import KeepAction, CutAction, LoopAction, ReorderAction, ActionType
        
        reward = 0.0
        beat_idx = action.beat_index
        
        # Use supervised reward from ground truth labels if available
        if self.audio_state.target_labels is not None and 0 <= beat_idx < len(self.audio_state.target_labels):
            target = self.audio_state.target_labels[beat_idx]  # 1.0 = KEEP, 0.0 = CUT
            should_keep = target >= 0.5
            should_cut = target < 0.5
            
            if isinstance(action, KeepAction):
                # Reward +1 if target is KEEP, penalty -1 if target is CUT
                reward = 1.0 if should_keep else -1.0
                
            elif isinstance(action, CutAction):
                # Reward +1 if target is CUT, penalty -1 if target is KEEP
                reward = 1.0 if should_cut else -1.0
                
            elif isinstance(action, LoopAction):
                # LOOP is ONLY valid if beat should be kept
                # If beat should be cut, looping is WRONG - big penalty
                if should_cut:
                    reward = -1.5  # Worse than just keeping a bad beat
                else:
                    # Valid loop - bonus for creative choice on great beats
                    # Bigger bonus if it's a really high-confidence keep (target > 0.9)
                    if target > 0.9:
                        reward = 1.3  # Better than KEEP for looping great content
                    elif target > 0.7:
                        reward = 1.1  # Small bonus for good content
                    else:
                        reward = 0.9  # Slight penalty for looping mediocre content
                    
                    # Apply loop ratio penalty if we're over-looping
                    reward += self._compute_loop_ratio_penalty()
                    
            elif isinstance(action, ReorderAction):
                # REORDER: only valid if moving kept content
                if should_cut:
                    reward = -1.2  # Moving content that should be cut
                else:
                    # Small bonus for creative reordering
                    reward = 1.0 + 0.05
            else:
                # Unknown action - check if we're skipping decisions
                if should_cut:
                    reward = 0.0  # Neutral for advancing past cut content
                else:
                    reward = -0.5  # Penalty for skipping kept content without deciding
        else:
            # Fallback: small positive reward for making progress
            edited_beats = set(self.edit_history.get_edited_beats())
            
            reward = 0.01
            if beat_idx in edited_beats:
                reward -= 0.1
            else:
                reward += 0.05
        
        # Duration penalty: penalize if projected output exceeds target max
        reward += self._compute_duration_penalty(action)
        
        self.last_reward_components = RewardComponents(total_reward=reward)
        return reward
    
    def _compute_duration_penalty(self, action: Action) -> float:
        """Compute penalty based on projected output duration exceeding target.
        
        Args:
            action: Action taken
            
        Returns:
            Penalty (negative) or bonus (positive) for duration management
        """
        from .actions import KeepAction, CutAction, LoopAction
        
        target_max_s = self.config.reward.target_max_duration_s
        penalty_weight = self.config.reward.duration_penalty_weight
        
        # Estimate current projected output duration
        projected_duration = self._estimate_output_duration()
        
        # Calculate what this action contributes to duration
        beat_idx = action.beat_index
        if beat_idx < len(self.audio_state.beat_times) - 1:
            beat_duration = self.audio_state.beat_times[beat_idx + 1] - self.audio_state.beat_times[beat_idx]
        else:
            # Last beat - estimate from tempo
            beat_duration = 60.0 / 120.0  # Assume 120 BPM if unknown
        
        # Actions that add duration get penalized when over budget
        if projected_duration > target_max_s:
            overage_ratio = (projected_duration - target_max_s) / target_max_s
            
            if isinstance(action, CutAction):
                # Cutting helps reduce duration - reward!
                return penalty_weight * min(0.3, overage_ratio * 0.5)
            elif isinstance(action, LoopAction):
                # Looping adds duration - penalize when over budget
                loop_times = getattr(action, 'loop_times', 2)
                added_duration = beat_duration * (loop_times - 1)
                return -penalty_weight * overage_ratio * (added_duration / beat_duration)
            elif isinstance(action, KeepAction):
                # Keep adds duration - small penalty when over budget
                return -penalty_weight * overage_ratio * 0.1
        else:
            # Under budget - no penalty, slight bonus for cutting
            if isinstance(action, CutAction):
                # Small bonus for cutting when near target
                budget_usage = projected_duration / target_max_s
                if budget_usage > 0.8:  # Only if we're using >80% of budget
                    return 0.05
        
        return 0.0
    
    def _estimate_output_duration(self) -> float:
        """Estimate projected output duration based on current edit history.
        
        Returns:
            Estimated output duration in seconds
        """
        total_duration = 0.0
        beat_times = self.audio_state.beat_times
        n_beats = len(beat_times)
        
        for beat_idx in self.edit_history.kept_beats:
            if beat_idx < n_beats - 1:
                beat_duration = beat_times[beat_idx + 1] - beat_times[beat_idx]
            else:
                # Last beat - estimate
                if n_beats >= 2:
                    beat_duration = beat_times[-1] - beat_times[-2]
                else:
                    beat_duration = 0.5  # Fallback
            total_duration += beat_duration
        
        # Account for loops (they multiply beat duration)
        for beat_idx, loop_times in self.edit_history.looped_beats.items():
            if beat_idx < n_beats - 1:
                beat_duration = beat_times[beat_idx + 1] - beat_times[beat_idx]
            else:
                if n_beats >= 2:
                    beat_duration = beat_times[-1] - beat_times[-2]
                else:
                    beat_duration = 0.5
            # Add extra duration from loops (loop_times - 1 extra copies)
            total_duration += beat_duration * (loop_times - 1)
        
        return total_duration

    def _compute_loop_ratio_penalty(self) -> float:
        """Compute penalty for excessive loop usage.
        
        Returns:
            Penalty (negative) if loop ratio exceeds max_loop_ratio, else 0.0
        """
        max_ratio = self.config.reward.max_loop_ratio
        penalty_weight = self.config.reward.loop_penalty_weight
        
        # Count current loops vs total beats processed
        n_loops = len(self.edit_history.looped_beats)
        n_processed = len(self.edit_history.get_edited_beats())
        
        if n_processed == 0:
            return 0.0
        
        current_ratio = n_loops / n_processed
        
        # No penalty if under the limit
        if current_ratio <= max_ratio:
            return 0.0
        
        # Progressive penalty as we exceed the limit
        # overage_ratio: how much we're over (e.g., 0.3 ratio vs 0.15 max = 1.0 overage)
        overage_ratio = (current_ratio - max_ratio) / max_ratio
        
        # Penalty scales with how far over we are
        # At 2x the max (e.g., 30% loops when max is 15%), penalty = -penalty_weight
        penalty = -penalty_weight * min(overage_ratio, 2.0)
        
        return penalty

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation vector
        """
        if self.audio_state is None or self.state_rep is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        total_duration = (
            self.audio_state.beat_times[-1]
            if len(self.audio_state.beat_times) > 0
            else 1.0
        )

        return self.state_rep.construct_observation(
            self.audio_state,
            self.edit_history,
            remaining_duration=self._get_remaining_duration(),
            total_duration=total_duration,
        )

    def _get_remaining_duration(self) -> float:
        """Compute remaining duration budget.

        Returns:
            Remaining duration
        """
        total_duration = (
            self.audio_state.beat_times[-1]
            if len(self.audio_state.beat_times) > 0
            else 1.0
        )
        target_duration = total_duration * self.config.reward.target_keep_ratio
        edited_duration = sum(
            self.audio_state.beat_times[i] if i < len(self.audio_state.beat_times) else 0.0
            for i in self.edit_history.kept_beats
        )
        return max(0.0, target_duration - edited_duration)

    def _is_terminated(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode should end
        """
        if self.audio_state is None:
            return False

        # Terminate if all beats are edited
        n_beats_edited = len(self.edit_history.get_edited_beats())
        total_beats = len(self.audio_state.beat_times)

        return n_beats_edited >= total_beats

    def render(self) -> None:
        """Render environment state (logging)."""
        if self.audio_state is None:
            return

        logger.info(
            f"Step {self.current_step}: "
            f"Beat {self.audio_state.beat_index}/{len(self.audio_state.beat_times)}, "
            f"Kept: {len(self.edit_history.kept_beats)}, "
            f"Cut: {len(self.edit_history.cut_beats)}, "
            f"Episode Reward: {sum(self.episode_rewards):.3f}"
        )

    def close(self) -> None:
        """Close environment."""
        pass

    def set_max_steps(self, max_steps: int) -> None:
        """Set maximum steps per episode.

        Args:
            max_steps: Maximum steps
        """
        self.max_steps = max_steps
