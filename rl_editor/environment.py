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
        """Compute minimal shaping reward for trajectory-based training.
        
        Provides small signals to guide exploration without overwhelming
        the end-of-episode trajectory reward.
        
        Args:
            action: Action taken
            
        Returns:
            Small shaping reward
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
        
        # Small penalty for excessive looping
        if isinstance(action, LoopAction):
            loop_penalty = self._compute_loop_ratio_penalty()
            reward += loop_penalty * 0.5
        
        # Scale down all rewards
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
        
        This evaluates the quality of the entire edit by:
        1. Building the edited audio from edit history
        2. Computing audio quality metrics (transitions, energy, tempo)
        3. Checking keep ratio alignment with target
        
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
            # Use default tempo_score but don't add to quality_score
        
        # 4. Energy flow (smooth dynamics)
        energy_score = self._compute_energy_flow_score(edited_audio)
        quality_score += reward_config.energy_flow_weight * energy_score
        total_weight += reward_config.energy_flow_weight
        
        # 5. Phrase alignment bonus (cuts at phrase boundaries)
        phrase_score = self._compute_phrase_alignment_score()
        quality_score += reward_config.phrase_completeness_weight * phrase_score
        total_weight += reward_config.phrase_completeness_weight
        
        # 6. Ground truth alignment (how well does edit match human edit)
        gt_alignment_score = 0.0
        if self.audio_state.target_labels is not None:
            gt_alignment_score = self._compute_ground_truth_alignment()
            # Weight ground truth heavily - this is what we really want to learn
            gt_weight = 3.0  # Higher weight than other metrics
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
                    f"transition: {transition_score:.2f}, tempo: {tempo_score:.2f}, gt_align: {gt_alignment_score:.2f})")
        
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