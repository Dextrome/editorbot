"""Gymnasium-compatible environment for RL-based audio editing."""

import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional, Tuple, Dict, List
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

    def __init__(self, config: Config, audio_state: Optional[AudioState] = None, learned_reward_model: Optional[Any] = None) -> None:
        """Initialize environment.

        Args:
            config: Configuration object
            audio_state: Audio state (can be provided or set via reset())
            learned_reward_model: Optional learned reward model for RLHF
        """
        super().__init__()
        self.config = config
        self.audio_state = audio_state
        self.learned_reward_model = learned_reward_model
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
        """Compute reward for a step.

        When trajectory rewards are enabled (use_trajectory_rewards=True):
        - Returns minimal shaping rewards (scaled down)
        - Main reward comes at episode end from trajectory_reward
        
        When trajectory rewards are disabled:
        - Uses traditional supervised reward from ground truth labels

        Args:
            action: Action taken

        Returns:
            Step reward
        """
        from .actions import KeepAction, CutAction, LoopAction, ReorderAction, ActionType
        
        reward_config = self.config.reward
        
        # If trajectory rewards enabled, use minimal step shaping
        if reward_config.use_trajectory_rewards:
            return self._compute_minimal_step_reward(action)
        
        # Otherwise use traditional supervised reward
        return self._compute_supervised_step_reward(action)
    
    def _compute_minimal_step_reward(self, action: Action) -> float:
        """Compute enhanced shaping reward for trajectory-based training (Option A).
        
        Provides richer signals to guide exploration:
        - Progress reward for making decisions
        - Ratio tracking (stay near target keep ratio)
        - NEW: Phrase boundary bonus (encourage musical cuts)
        - NEW: Section coherence bonus (keep adjacent beats together)
        - NEW: Energy continuity (avoid sudden jumps)
        
        Args:
            action: Action taken
            
        Returns:
            Shaping reward (scaled by step_reward_scale)
        """
        from .actions import KeepAction, CutAction, LoopAction, ReorderAction
        
        reward_config = self.config.reward
        scale = reward_config.step_reward_scale  # Default 0.01
        
        reward = 0.0
        beat_idx = action.beat_index
        n_beats = len(self.audio_state.beat_times)
        
        # Small reward for making progress (any decision is good)
        reward += 0.1
        
        # Encourage keeping a reasonable ratio (not all keep or all cut)
        n_kept = len(self.edit_history.kept_beats)
        n_processed = len(self.edit_history.get_edited_beats())
        
        if n_processed > 10:  # Wait until we have some data
            current_ratio = n_kept / n_processed
            target_ratio = reward_config.target_keep_ratio
            
            # Small bonus for staying near target ratio
            ratio_diff = abs(current_ratio - target_ratio)
            if ratio_diff < 0.1:
                reward += 0.2
            elif ratio_diff > 0.3:
                reward -= 0.1
        
        # Penalties for excessive/repetitive looping
        if isinstance(action, LoopAction):
            # Penalty for exceeding max loop ratio
            loop_penalty = self._compute_loop_ratio_penalty()
            reward += loop_penalty
            
            # Penalty for looping beats near other looped beats (repetition)
            repetition_penalty = self._compute_loop_repetition_penalty(beat_idx)
            reward += repetition_penalty
        
        # === NEW ENHANCED STEP REWARDS (Option A) ===
        
        # 1. Phrase boundary bonus for cuts
        if isinstance(action, CutAction):
            phrase_bonus = self._compute_step_phrase_boundary_bonus(beat_idx)
            reward += reward_config.step_phrase_boundary_bonus * phrase_bonus
        
        # 2. Section coherence bonus for keeps
        if isinstance(action, KeepAction):
            coherence_bonus = self._compute_step_coherence_bonus(beat_idx)
            reward += reward_config.step_coherence_bonus * coherence_bonus
        
        # 3. Energy continuity check (penalty for sudden jumps)
        if isinstance(action, KeepAction) and len(self.edit_history.kept_beats) > 0:
            energy_penalty = self._compute_step_energy_penalty(beat_idx)
            reward -= reward_config.step_energy_continuity_weight * energy_penalty
        
        # Scale down all rewards
        return reward * scale
    
    def _compute_step_phrase_boundary_bonus(self, beat_idx: int) -> float:
        """Compute bonus for cutting at musical phrase boundaries.
        
        Args:
            beat_idx: Beat index being cut
            
        Returns:
            Bonus value [0, 1] - higher for phrase boundaries
        """
        # Common phrase boundary positions (modulo)
        # Best: end of 8-beat phrase, also good: end of 4-beat bar
        
        # Check if we're at end of a phrase (beat 7, 15, 23, etc. -> before beat 0, 8, 16...)
        # Or at the start of a new phrase
        
        position_in_phrase = beat_idx % 8
        
        if position_in_phrase == 7 or position_in_phrase == 0:
            # At phrase boundary - great place to cut
            return 1.0
        elif position_in_phrase == 3 or position_in_phrase == 4:
            # At bar boundary - good place to cut
            return 0.6
        elif position_in_phrase in [1, 2, 5, 6]:
            # Mid-bar - less ideal but okay
            return 0.2
        
        return 0.0
    
    def _compute_step_coherence_bonus(self, beat_idx: int) -> float:
        """Compute bonus for keeping beats adjacent to other kept beats.
        
        Encourages keeping coherent sections rather than scattered beats.
        
        Args:
            beat_idx: Beat index being kept
            
        Returns:
            Bonus value [0, 1] - higher if adjacent to other kept beats
        """
        kept_beats = self.edit_history.kept_beats
        
        if not kept_beats:
            return 0.5  # First beat - neutral
        
        # Check if adjacent beats are kept
        prev_kept = (beat_idx - 1) in kept_beats
        next_kept = (beat_idx + 1) in kept_beats
        
        if prev_kept and next_kept:
            # Filling a gap - great for coherence
            return 1.0
        elif prev_kept:
            # Extending a section forward - good
            return 0.8
        elif next_kept:
            # Will connect to next kept beat - good
            return 0.7
        else:
            # Isolated beat (so far) - less ideal but might be fine
            # Check if we're near other kept beats
            nearby_kept = any(
                abs(beat_idx - k) <= 3 
                for k in kept_beats
            )
            if nearby_kept:
                return 0.4
            else:
                return 0.1  # Very isolated
    
    def _compute_step_energy_penalty(self, beat_idx: int) -> float:
        """Compute penalty for sudden energy discontinuity.
        
        If we keep a beat that has very different energy from the previous
        kept beat, that could create a jarring transition.
        
        Args:
            beat_idx: Beat index being kept
            
        Returns:
            Penalty value [0, 1] - higher for larger energy jumps
        """
        if self.audio_state.beat_features is None:
            return 0.0
        
        kept_beats = sorted(self.edit_history.kept_beats)
        
        if not kept_beats:
            return 0.0  # No previous beats to compare
        
        # Find most recent kept beat
        prev_kept_idx = kept_beats[-1]
        
        beat_features = self.audio_state.beat_features
        n_beats = len(beat_features)
        
        if prev_kept_idx >= n_beats or beat_idx >= n_beats:
            return 0.0
        
        # Use first few features (typically energy-related)
        # Assuming features include RMS energy in first positions
        try:
            prev_energy = np.mean(np.abs(beat_features[prev_kept_idx][:8]))
            curr_energy = np.mean(np.abs(beat_features[beat_idx][:8]))
            
            # Compute relative energy change
            avg_energy = (prev_energy + curr_energy) / 2 + 1e-6
            energy_diff = abs(prev_energy - curr_energy) / avg_energy
            
            # Penalty increases with energy difference
            # 50% change = 0.5 penalty, 100% change = 1.0 penalty
            penalty = min(1.0, energy_diff)
            
            return penalty
            
        except Exception:
            return 0.0
        return reward * scale
    
    def _compute_supervised_step_reward(self, action: Action) -> float:
        """Compute supervised reward from ground truth labels (legacy mode).
        
        Args:
            action: Action taken
            
        Returns:
            Step reward based on matching ground truth
        """
        from .actions import KeepAction, CutAction, LoopAction, ReorderAction
        
        reward = 0.0
        beat_idx = action.beat_index
        
        # Use supervised reward from ground truth labels if available
        if self.audio_state.target_labels is not None and 0 <= beat_idx < len(self.audio_state.target_labels):
            target = self.audio_state.target_labels[beat_idx]  # 1.0 = KEEP, 0.0 = CUT
            should_keep = target >= 0.5
            should_cut = target < 0.5
            
            if isinstance(action, KeepAction):
                reward = 1.0 if should_keep else -1.0
                
            elif isinstance(action, CutAction):
                reward = 1.0 if should_cut else -1.0
                
            elif isinstance(action, LoopAction):
                if should_cut:
                    reward = -1.5
                else:
                    if target > 0.9:
                        reward = 1.3
                    elif target > 0.7:
                        reward = 1.1
                    else:
                        reward = 0.9
                    reward += self._compute_loop_ratio_penalty()
                    
            elif isinstance(action, ReorderAction):
                if should_cut:
                    reward = -1.2
                else:
                    reward = 1.05
            else:
                reward = 0.0 if should_cut else -0.5
        else:
            # Fallback: small progress reward
            reward = 0.01
        
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

    def _compute_loop_repetition_penalty(self, beat_idx: int) -> float:
        """Compute penalty for looping beats near other looped beats.
        
        This discourages creating repetitive sections where multiple
        consecutive or nearby beats are all looped.
        
        Args:
            beat_idx: The beat index being looped
            
        Returns:
            Penalty (negative) based on nearby looped beats
        """
        penalty_weight = self.config.reward.loop_repetition_penalty
        window = self.config.reward.loop_proximity_window
        
        # Count how many nearby beats are already looped
        nearby_loops = 0
        for looped_beat in self.edit_history.looped_beats:
            if looped_beat != beat_idx and abs(looped_beat - beat_idx) <= window:
                nearby_loops += 1
        
        if nearby_loops == 0:
            return 0.0
        
        # Progressive penalty: more nearby loops = stronger penalty
        # 1 nearby: -0.3, 2 nearby: -0.6, 3+: -0.9 (capped)
        penalty = -penalty_weight * min(nearby_loops, 3)
        
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
    
    def set_learned_reward_model(self, model: Any) -> None:
        """Set a learned reward model for RLHF.
        
        Args:
            model: Learned reward model (must have forward method)
        """
        self.learned_reward_model = model
    
    def compute_learned_episode_reward(self) -> float:
        """Compute learned reward for the entire episode.
        
        Uses the learned reward model to evaluate the quality of the edit.
        Call this at episode end for sparse learned reward.
        
        Returns:
            Learned reward scalar, or 0.0 if no model is set
        """
        if self.learned_reward_model is None:
            return 0.0
        
        try:
            import torch
            
            # Get beat features and actions
            beat_features = self.audio_state.beat_features
            n_beats = len(beat_features)
            
            # Pad/truncate features to match expected input dimensions (125)
            expected_dim = 125
            actual_dim = beat_features.shape[1] if beat_features.ndim > 1 else beat_features.shape[0]
            if actual_dim < expected_dim:
                # Pad with zeros
                padding = np.zeros((n_beats, expected_dim - actual_dim))
                beat_features = np.concatenate([beat_features, padding], axis=1)
            elif actual_dim > expected_dim:
                # Truncate
                beat_features = beat_features[:, :expected_dim]
            
            # Build action sequence (0=KEEP, 1=CUT, etc.)
            actions = np.ones(n_beats, dtype=np.int64)  # Default: CUT
            for beat_idx in self.edit_history.kept_beats:
                if beat_idx < n_beats:
                    actions[beat_idx] = 0  # KEEP
            for beat_idx in self.edit_history.looped_beats:
                if beat_idx < n_beats:
                    actions[beat_idx] = 2  # LOOP
            
            # Prepare tensors
            device = next(self.learned_reward_model.parameters()).device
            feat_tensor = torch.from_numpy(beat_features).float().unsqueeze(0).to(device)
            action_tensor = torch.from_numpy(actions).long().unsqueeze(0).to(device)
            mask = torch.ones(1, n_beats, dtype=torch.bool).to(device)
            
            # Compute reward
            with torch.no_grad():
                reward = self.learned_reward_model(feat_tensor, action_tensor, mask)
            
            return reward.item()
        
        except Exception as e:
            logger.warning(f"Failed to compute learned reward: {e}")
            return 0.0

    def compute_trajectory_reward(self) -> float:
        """Compute trajectory-based reward at end of episode.
        
        Enhanced with perceptual quality metrics (Option B):
        1. Building the edited audio from edit history
        2. Computing audio quality metrics (transitions, energy, tempo)
        3. NEW: Spectral continuity at edit points
        4. NEW: Section coherence (consecutive beat groupings)
        5. NEW: Flow continuity (beat-to-beat transitions)
        6. Ground truth alignment (reduced weight - learn quality, not just copy labels)
        
        Returns:
            Trajectory reward (scaled by trajectory_reward_scale)
        """
        if self.audio_state is None or self.audio_state.raw_audio is None:
            logger.debug("No raw audio available for trajectory reward")
            return 0.0
        
        reward_config = self.config.reward
        
        # Build edited audio
        try:
            edited_audio = self._build_edited_audio()
            if edited_audio is None or len(edited_audio) == 0:
                return -reward_config.trajectory_reward_scale * 0.5  # Penalty for empty edit
        except Exception as e:
            logger.warning(f"Failed to build edited audio: {e}")
            return 0.0
        
        # Compute quality metrics
        quality_score = 0.0
        total_weight = 0.0
        
        # 1. Keep ratio score (how close to target)
        n_kept = len(self.edit_history.kept_beats)
        n_total = len(self.audio_state.beat_times)
        keep_ratio = 0.0
        if n_total > 0:
            keep_ratio = n_kept / n_total
            target_ratio = reward_config.target_keep_ratio
            # Score 1.0 at target, decreasing as we deviate
            ratio_diff = abs(keep_ratio - target_ratio)
            keep_ratio_score = max(0.0, 1.0 - ratio_diff / target_ratio)
            quality_score += reward_config.keep_ratio_weight * keep_ratio_score
            total_weight += reward_config.keep_ratio_weight
        
        # 2. Transition smoothness (detect clicks/pops)
        transition_score = self._compute_transition_smoothness(edited_audio)
        quality_score += reward_config.transition_smoothness_weight * transition_score
        total_weight += reward_config.transition_smoothness_weight
        
        # 3. Tempo consistency
        tempo_score = 0.5  # Default if estimation fails
        try:
            from .utils import estimate_tempo
            sr = self.audio_state.sample_rate
            original_tempo = self.audio_state.tempo or estimate_tempo(self.audio_state.raw_audio, sr)
            edited_tempo = estimate_tempo(edited_audio, sr)
            tempo_diff = abs(original_tempo - edited_tempo) / max(original_tempo, 1.0)
            tempo_score = max(0.0, 1.0 - tempo_diff * 2)  # 50% change = 0 score
            quality_score += reward_config.tempo_consistency_weight * tempo_score
            total_weight += reward_config.tempo_consistency_weight
        except Exception as e:
            logger.debug(f"Tempo estimation failed: {e}")
        
        # 4. Energy flow (smooth dynamics)
        energy_score = self._compute_energy_flow_score(edited_audio)
        quality_score += reward_config.energy_flow_weight * energy_score
        total_weight += reward_config.energy_flow_weight
        
        # 5. Phrase alignment bonus (cuts at phrase boundaries)
        phrase_score = self._compute_phrase_alignment_score()
        quality_score += reward_config.phrase_completeness_weight * phrase_score
        total_weight += reward_config.phrase_completeness_weight
        
        # === NEW ENHANCED METRICS (Option B) ===
        
        # 6. Spectral continuity at edit points
        spectral_score = self._compute_spectral_continuity_score(edited_audio)
        quality_score += reward_config.spectral_continuity_weight * spectral_score
        total_weight += reward_config.spectral_continuity_weight
        
        # 7. Section coherence (reward for keeping consecutive beats together)
        coherence_score = self._compute_section_coherence_score()
        quality_score += reward_config.section_coherence_weight * coherence_score
        total_weight += reward_config.section_coherence_weight
        
        # 8. Flow continuity (beat-to-beat transitions in kept sequence)
        flow_score = self._compute_flow_continuity_score()
        quality_score += reward_config.flow_continuity_weight * flow_score
        total_weight += reward_config.flow_continuity_weight
        
        # 9. Beat alignment quality at edit points
        beat_align_score = self._compute_beat_alignment_quality()
        quality_score += reward_config.beat_alignment_quality_weight * beat_align_score
        total_weight += reward_config.beat_alignment_quality_weight
        
        # 10. Ground truth alignment (REDUCED weight - learn quality, don't just copy)
        gt_alignment_score = 0.0
        if self.audio_state.target_labels is not None:
            gt_alignment_score = self._compute_ground_truth_alignment()
            # Use configurable weight (default 1.0, was hardcoded 3.0)
            gt_weight = reward_config.ground_truth_weight
            quality_score += gt_weight * gt_alignment_score
            total_weight += gt_weight
        
        # Normalize and scale
        if total_weight > 0:
            normalized_score = quality_score / total_weight
        else:
            normalized_score = 0.0
        
        # Scale to desired range
        trajectory_reward = normalized_score * reward_config.trajectory_reward_scale
        
        logger.debug(f"Trajectory reward: {trajectory_reward:.2f} (keep_ratio: {keep_ratio:.2f}, "
                    f"transition: {transition_score:.2f}, tempo: {tempo_score:.2f}, "
                    f"spectral: {spectral_score:.2f}, coherence: {coherence_score:.2f}, "
                    f"flow: {flow_score:.2f}, gt_align: {gt_alignment_score:.2f})")
        
        return trajectory_reward
    
    def _compute_ground_truth_alignment(self) -> float:
        """Compute how well the agent's edit matches the ground truth human edit.
        
        Returns:
            Alignment score 0.0-1.0 (F1 score of kept beats)
        """
        if self.audio_state.target_labels is None:
            return 0.0
        
        target_labels = self.audio_state.target_labels
        n_beats = len(target_labels)
        
        # Build predicted labels from edit history
        pred_labels = np.zeros(n_beats)
        for beat_idx in self.edit_history.kept_beats:
            if 0 <= beat_idx < n_beats:
                pred_labels[beat_idx] = 1.0
        
        # Compute F1 score (balances precision and recall)
        # True positives: beats we kept that should be kept
        tp = np.sum((pred_labels == 1) & (target_labels == 1))
        # False positives: beats we kept that should be cut
        fp = np.sum((pred_labels == 1) & (target_labels == 0))
        # False negatives: beats we cut that should be kept
        fn = np.sum((pred_labels == 0) & (target_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        return f1_score
    
    def _compute_spectral_continuity_score(self, edited_audio: np.ndarray) -> float:
        """Compute spectral continuity at edit points.
        
        Measures how smoothly the frequency content transitions at edit boundaries.
        High score = smooth spectral transitions, low score = jarring frequency jumps.
        
        Args:
            edited_audio: Edited audio array
            
        Returns:
            Spectral continuity score [0, 1]
        """
        sr = self.audio_state.sample_rate
        
        try:
            # Compute mel spectrogram
            import librosa
            n_fft = 2048
            hop_length = 512
            mel_spec = librosa.feature.melspectrogram(
                y=edited_audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            if mel_db.shape[1] < 3:
                return 0.5  # Too short
            
            # Compute frame-to-frame spectral flux
            spectral_flux = np.sqrt(np.sum(np.diff(mel_db, axis=1) ** 2, axis=0))
            
            # Find large jumps (potential edit points)
            mean_flux = np.mean(spectral_flux)
            std_flux = np.std(spectral_flux)
            threshold = mean_flux + 2 * std_flux
            
            n_large_jumps = np.sum(spectral_flux > threshold)
            jump_density = n_large_jumps / len(spectral_flux)
            
            # Score: 1.0 if no large jumps, decreasing with more jumps
            score = max(0.0, 1.0 - jump_density * 5)
            return score
            
        except Exception as e:
            logger.debug(f"Spectral continuity failed: {e}")
            return 0.5
    
    def _compute_section_coherence_score(self) -> float:
        """Compute section coherence score.
        
        Rewards keeping consecutive beats together (coherent sections)
        rather than keeping isolated scattered beats.
        
        Returns:
            Section coherence score [0, 1]
        """
        kept_sorted = sorted(self.edit_history.kept_beats)
        
        if len(kept_sorted) < 2:
            return 0.5  # Not enough data
        
        # Count consecutive pairs
        n_consecutive = 0
        for i in range(len(kept_sorted) - 1):
            if kept_sorted[i + 1] == kept_sorted[i] + 1:
                n_consecutive += 1
        
        # Score based on ratio of consecutive to total kept
        max_possible_consecutive = len(kept_sorted) - 1
        if max_possible_consecutive == 0:
            return 1.0
        
        coherence_ratio = n_consecutive / max_possible_consecutive
        
        # Reward higher coherence (keeping sections together)
        # But don't penalize too much - some gaps are okay
        return 0.3 + 0.7 * coherence_ratio
    
    def _compute_flow_continuity_score(self) -> float:
        """Compute flow continuity based on beat-level features.
        
        Measures how smoothly the audio flows between kept beats,
        using the audio features to detect jarring transitions.
        
        Returns:
            Flow continuity score [0, 1]
        """
        if self.audio_state.beat_features is None:
            return 0.5
        
        kept_sorted = sorted(self.edit_history.kept_beats)
        
        if len(kept_sorted) < 2:
            return 0.5
        
        beat_features = self.audio_state.beat_features
        n_beats = len(beat_features)
        
        # Compute feature distances between consecutive kept beats
        distances = []
        for i in range(len(kept_sorted) - 1):
            curr_idx = kept_sorted[i]
            next_idx = kept_sorted[i + 1]
            
            if curr_idx < n_beats and next_idx < n_beats:
                # Euclidean distance between feature vectors
                dist = np.linalg.norm(beat_features[curr_idx] - beat_features[next_idx])
                distances.append(dist)
        
        if not distances:
            return 0.5
        
        # Compare to distances between originally adjacent beats
        original_distances = []
        for i in range(min(n_beats - 1, 100)):  # Sample first 100 beats
            dist = np.linalg.norm(beat_features[i] - beat_features[i + 1])
            original_distances.append(dist)
        
        if not original_distances:
            return 0.5
        
        mean_edit_dist = np.mean(distances)
        mean_orig_dist = np.mean(original_distances)
        
        # Score: 1.0 if edit distances are similar to original, lower if larger
        if mean_orig_dist > 0:
            ratio = mean_edit_dist / (mean_orig_dist + 1e-6)
            score = max(0.0, 1.0 - (ratio - 1.0) * 0.5)  # Allow some slack
        else:
            score = 0.5
        
        return min(1.0, score)
    
    def _compute_beat_alignment_quality(self) -> float:
        """Compute beat alignment quality at edit boundaries.
        
        Checks if cuts happen at beat boundaries (rather than mid-beat)
        and if the timing alignment is clean.
        
        Returns:
            Beat alignment quality score [0, 1]
        """
        # Since we're always cutting at beat boundaries by design,
        # this metric focuses on the quality of those boundaries
        
        kept_sorted = sorted(self.edit_history.kept_beats)
        
        if len(kept_sorted) < 2:
            return 1.0  # Single section is always aligned
        
        # Find edit boundaries (where we jump between non-consecutive beats)
        n_edit_points = 0
        good_edit_points = 0
        
        beat_times = self.audio_state.beat_times
        n_beats = len(beat_times)
        
        for i in range(len(kept_sorted) - 1):
            curr_idx = kept_sorted[i]
            next_idx = kept_sorted[i + 1]
            
            # If not consecutive, this is an edit point
            if next_idx != curr_idx + 1:
                n_edit_points += 1
                
                # Check if both beats are on downbeats (every 4th beat)
                # This makes for cleaner musical transitions
                curr_is_downbeat = curr_idx % 4 == 0
                next_is_downbeat = next_idx % 4 == 0
                
                # Also check phrase boundaries (every 8th beat)
                curr_is_phrase = curr_idx % 8 == 0
                next_is_phrase = next_idx % 8 == 0
                
                # Score this edit point
                if curr_is_phrase and next_is_phrase:
                    good_edit_points += 1.0
                elif curr_is_downbeat or next_is_downbeat:
                    good_edit_points += 0.7
                elif (curr_idx + 1) % 4 == 0 or next_idx % 4 == 0:
                    # Cutting just before a downbeat is okay too
                    good_edit_points += 0.5
                else:
                    good_edit_points += 0.2
        
        if n_edit_points == 0:
            return 1.0  # No edit points needed
        
        return good_edit_points / n_edit_points

    def _build_edited_audio(self, crossfade_ms: float = 50.0) -> Optional[np.ndarray]:
        """Build edited audio from edit history.
        
        Args:
            crossfade_ms: Crossfade duration at edit boundaries
            
        Returns:
            Edited audio array or None if failed
        """
        if self.audio_state.raw_audio is None:
            return None
        
        audio = self.audio_state.raw_audio
        sr = self.audio_state.sample_rate
        beat_times = self.audio_state.beat_times
        
        # Convert beat times to samples
        beat_samples = (beat_times * sr).astype(int)
        beat_samples = np.append(beat_samples, len(audio))
        
        crossfade_samples = int(crossfade_ms * sr / 1000)
        
        # Collect segments to include
        segments = []
        
        # Sort kept beats
        kept_sorted = sorted(self.edit_history.kept_beats)
        
        for beat_idx in kept_sorted:
            if beat_idx >= len(beat_times):
                continue
            
            start = beat_samples[beat_idx]
            end = beat_samples[beat_idx + 1]
            segment = audio[start:end].copy()
            
            # Check if this beat is looped
            loop_times = self.edit_history.looped_beats.get(beat_idx, 1)
            if loop_times > 1:
                segment = np.tile(segment, loop_times)
            
            segments.append(segment)
        
        if not segments:
            return None
        
        # Concatenate with crossfades
        if len(segments) == 1:
            return segments[0]
        
        result = segments[0]
        for seg in segments[1:]:
            result = self._apply_crossfade(result, seg, crossfade_samples)
        
        return result
    
    def _apply_crossfade(self, seg1: np.ndarray, seg2: np.ndarray, crossfade_samples: int) -> np.ndarray:
        """Apply crossfade between two segments.
        
        Args:
            seg1: First segment
            seg2: Second segment
            crossfade_samples: Number of samples for crossfade
            
        Returns:
            Combined audio with crossfade
        """
        crossfade_samples = min(crossfade_samples, len(seg1), len(seg2))
        
        if crossfade_samples <= 0:
            return np.concatenate([seg1, seg2])
        
        # Create fade curves
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)
        
        # Apply crossfade
        result = np.zeros(len(seg1) + len(seg2) - crossfade_samples)
        result[:len(seg1) - crossfade_samples] = seg1[:-crossfade_samples]
        result[len(seg1) - crossfade_samples:len(seg1)] = (
            seg1[-crossfade_samples:] * fade_out + seg2[:crossfade_samples] * fade_in
        )
        result[len(seg1):] = seg2[crossfade_samples:]
        
        return result
    
    def _compute_transition_smoothness(self, audio: np.ndarray) -> float:
        """Compute transition smoothness score by detecting clicks/pops.
        
        High score = smooth transitions, low score = audible discontinuities.
        
        Args:
            audio: Audio array
            
        Returns:
            Smoothness score [0, 1]
        """
        if len(audio) < 1000:
            return 0.5  # Too short to evaluate
        
        # Compute sample-to-sample differences
        diff = np.abs(np.diff(audio))
        
        # Detect large jumps (potential clicks)
        threshold = np.std(diff) * 5  # 5 sigma threshold
        n_clicks = np.sum(diff > threshold)
        
        # Score based on click density
        click_density = n_clicks / len(audio) * 1000  # clicks per 1000 samples
        
        # Score: 1.0 if no clicks, decreasing with more clicks
        score = max(0.0, 1.0 - click_density * 10)
        
        return score
    
    def _compute_energy_flow_score(self, audio: np.ndarray) -> float:
        """Compute energy flow smoothness score.
        
        Checks for smooth dynamics without jarring drops.
        
        Args:
            audio: Audio array
            
        Returns:
            Energy flow score [0, 1]
        """
        sr = self.audio_state.sample_rate
        
        # Compute RMS energy in windows
        hop_length = sr // 10  # 100ms windows
        n_windows = len(audio) // hop_length
        
        if n_windows < 3:
            return 0.5
        
        rms = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * hop_length
            end = min(start + hop_length, len(audio))
            window = audio[start:end]
            rms[i] = np.sqrt(np.mean(window ** 2) + 1e-10)
        
        # Compute energy changes
        energy_diff = np.abs(np.diff(rms))
        mean_energy = np.mean(rms) + 1e-10
        
        # Normalize by mean energy
        relative_changes = energy_diff / mean_energy
        
        # Score based on variance of changes (lower = smoother)
        change_variance = np.var(relative_changes)
        
        # Score: 1.0 if very smooth, decreasing with variance
        score = max(0.0, 1.0 - change_variance * 5)
        
        return score
    
    def _compute_phrase_alignment_score(self) -> float:
        """Compute phrase alignment score.
        
        Rewards cuts that happen at phrase boundaries (every 4 or 8 beats).
        
        Returns:
            Phrase alignment score [0, 1]
        """
        if not self.edit_history.cut_beats:
            return 1.0  # No cuts = no alignment needed
        
        phrase_lengths = [4, 8, 16]  # Common phrase lengths
        n_aligned = 0
        n_cuts = len(self.edit_history.cut_beats)
        
        for cut_idx in self.edit_history.cut_beats:
            # Check if cut is at a phrase boundary
            for phrase_len in phrase_lengths:
                if cut_idx % phrase_len == 0 or cut_idx % phrase_len == phrase_len - 1:
                    n_aligned += 1
                    break
        
        score = n_aligned / max(n_cuts, 1)
        return score