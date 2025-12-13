"""Factored Environment for RL-based audio editing.

Uses factored action space: (action_type, action_size, action_amount)
instead of single discrete action index.

This environment expects actions as tuples of 3 integers.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

from .config import Config
from .state import AudioState, StateRepresentation
from .actions_factored import (
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace, EditHistoryFactored,
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
    apply_factored_action,
)

logger = logging.getLogger(__name__)


class AudioEditingEnvFactored(gym.Env):
    """Gymnasium environment for audio editing with factored actions.
    
    Action space: MultiDiscrete([18, 5, 5]) for (type, size, amount)
    Observation space: Box with state features
    """

    def __init__(
        self,
        config: Config,
        audio_state: Optional[AudioState] = None,
        learned_reward_model: Optional[Any] = None,
    ) -> None:
        """Initialize environment.
        
        Args:
            config: Configuration object
            audio_state: Initial audio state (can be set later via reset)
            learned_reward_model: Optional learned reward model for RLHF
        """
        super().__init__()
        self.config = config
        self.audio_state = audio_state
        self.learned_reward_model = learned_reward_model
        
        # Will be initialized on reset
        self.state_rep: Optional[StateRepresentation] = None
        self.action_space_factored: Optional[FactoredActionSpace] = None
        self.edit_history: Optional[EditHistoryFactored] = None
        self.current_beat: int = 0
        self.episode_actions: List[FactoredAction] = []
        self.step_rewards: List[float] = []
        self.episode_reward_breakdown: Dict[str, float] = {}  # Detailed reward breakdown
        self.max_steps: int = getattr(config.training, 'max_steps', 10000)  # Truncation limit
        
        # Define action space as MultiDiscrete
        self.action_space = spaces.MultiDiscrete([N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS])
        
        # Observation space will be set on first reset
        self.observation_space = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
        
        if options and "audio_state" in options:
            self.audio_state = options["audio_state"]
        
        if self.audio_state is None:
            raise ValueError("audio_state must be provided")
        
        # Initialize components
        if self.state_rep is None:
            self.state_rep = StateRepresentation(self.config)
        
        # Update state representation dimension if needed
        if self.audio_state.beat_features is not None and self.audio_state.beat_features.ndim > 1:
            actual_dim = self.audio_state.beat_features.shape[1]
            if actual_dim != self.state_rep.beat_feature_dim:
                self.state_rep.set_beat_feature_dim(actual_dim)
                self.observation_space = None
        
        # Initialize action space
        n_beats = len(self.audio_state.beat_times)
        self.action_space_factored = FactoredActionSpace(n_beats=n_beats)
        self.action_space_factored.reset()
        
        # Reset state (must happen BEFORE _get_observation)
        self.edit_history = EditHistoryFactored()
        self.current_beat = 0
        self.episode_actions = []
        self.step_rewards = []
        
        # Set observation space if not set
        if self.observation_space is None:
            obs = self._get_observation()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
            )
        
        obs = self._get_observation()
        info = {
            "n_beats": n_beats,
            "beat": self.current_beat,
        }
        
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take action in environment.
        
        Args:
            action: Array of [type, size, amount] integers
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.audio_state is None or self.action_space_factored is None:
            raise RuntimeError("Environment not properly initialized")
        
        # Parse action
        action_type = int(action[0])
        action_size = int(action[1])
        action_amount = int(action[2])
        
        # Create factored action
        factored_action = FactoredAction(
            action_type=ActionType(action_type),
            action_size=ActionSize(action_size),
            action_amount=ActionAmount(action_amount),
            beat_index=self.current_beat,
        )
        
        # Apply action
        n_beats = len(self.audio_state.beat_times)
        beats_advanced = apply_factored_action(
            factored_action,
            self.current_beat,
            self.edit_history,
            n_beats,
            self.action_space_factored,
        )
        
        # Update state
        self.current_beat += beats_advanced
        self.episode_actions.append(factored_action)
        
        # Compute step reward (minimal, most reward at episode end)
        step_reward = self._compute_step_reward(factored_action)
        self.step_rewards.append(step_reward)
        
        # Check termination
        terminated = self.current_beat >= n_beats
        truncated = len(self.episode_actions) >= self.max_steps
        
        # Episode reward at termination or truncation
        episode_reward = 0.0
        if terminated or truncated:
            episode_reward = self._compute_episode_reward()
        
        total_reward = step_reward + episode_reward
        
        # Get observation
        obs = self._get_observation()
        
        # Info
        info = {
            "beat": self.current_beat,
            "action_type": action_type,
            "action_size": action_size,
            "action_amount": action_amount,
            "step_reward": step_reward,
            "episode_reward": episode_reward if terminated else 0.0,
            "keep_ratio": self.edit_history.get_keep_ratio(),
            "n_section_decisions": len([a for a in self.episode_actions 
                                        if a.n_beats > 1]),
        }
        
        return obs, total_reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        if self.audio_state is None or self.state_rep is None:
            return np.zeros(100, dtype=np.float32)
        
        n_beats = len(self.audio_state.beat_times)
        current_beat = min(self.current_beat, n_beats - 1)
        
        # Create a copy of audio_state with updated beat_index
        audio_state_current = AudioState(
            beat_index=current_beat,
            beat_times=self.audio_state.beat_times,
            beat_features=self.audio_state.beat_features,
            mel_spectrogram=self.audio_state.mel_spectrogram,
            stem_features=self.audio_state.stem_features,
            global_features=self.audio_state.global_features,
            tempo=self.audio_state.tempo,
            energy_contour=self.audio_state.energy_contour,
            target_labels=self.audio_state.target_labels,
            raw_audio=self.audio_state.raw_audio,
            sample_rate=self.audio_state.sample_rate,
        )
        
        # Create EditHistory-compatible object from factored history
        from .state import EditHistory
        edit_hist = EditHistory()
        edit_hist.kept_beats = list(self.edit_history.kept_beats)
        edit_hist.cut_beats = list(self.edit_history.cut_beats)
        # Map looped_sections to looped_beats dict (beat_idx: times)
        edit_hist.looped_beats = {}
        for start_beat, n_beats, times in self.edit_history.looped_sections:
            for i in range(n_beats):
                edit_hist.looped_beats[start_beat + i] = times
        # Map effect lists to reordered_pairs for history flags
        n_effects = (
            len(self.edit_history.fade_markers) +
            len(self.edit_history.time_changes) +
            len(self.edit_history.reversed_sections) +
            len(self.edit_history.gain_changes) +
            len(self.edit_history.eq_changes) +
            len(self.edit_history.audio_effects)
        )
        if n_effects > 0:
            edit_hist.reordered_pairs = [(i, i) for i in range(n_effects)]
        
        # Get duration info
        avg_beat_duration = np.mean(np.diff(self.audio_state.beat_times)) if len(self.audio_state.beat_times) > 1 else 0.5
        total_duration = n_beats * avg_beat_duration
        remaining_beats = n_beats - current_beat
        remaining_duration = remaining_beats * avg_beat_duration
        
        # Get observation from state representation
        obs = self.state_rep.construct_observation(
            audio_state=audio_state_current,
            edit_history=edit_hist,
            remaining_duration=remaining_duration,
            total_duration=total_duration,
        )
        
        return obs.astype(np.float32)

    def get_action_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get masks for valid actions.
        
        Returns:
            Tuple of (type_mask, size_mask, amount_mask)
        """
        if self.action_space_factored is None or self.edit_history is None:
            return (
                np.ones(N_ACTION_TYPES, dtype=bool),
                np.ones(N_ACTION_SIZES, dtype=bool),
                np.ones(N_ACTION_AMOUNTS, dtype=bool),
            )
        
        return self.action_space_factored.get_combined_mask(
            self.current_beat,
            self.edit_history,
        )

    def _compute_step_reward(self, action: FactoredAction) -> float:
        """Compute minimal step reward.
        
        Most reward comes from episode end, but small signals help learning.
        """
        reward = 0.0
        
        # Progress signal
        progress = self.current_beat / max(1, len(self.audio_state.beat_times))
        
        # Small penalty if keeping too much early in episode
        if progress < 0.3:
            keep_ratio = self.edit_history.get_keep_ratio()
            if keep_ratio > 0.9:
                reward -= 0.1
        
        # Small bonus for section-level decisions (more efficient)
        if action.n_beats >= 4:
            reward += 0.05
        
        return reward

    def _compute_episode_reward(self) -> float:
        """Compute episode-end reward based on edit quality.
        
        Stores breakdown in self.episode_reward_breakdown.
        """
        self.episode_reward_breakdown = {}  # Reset breakdown
        reward = 0.0
        n_beats = len(self.audio_state.beat_times)
        
        # === 1. Keep ratio constraint ===
        keep_ratio = self.edit_history.get_keep_ratio()
        target_keep = self.config.reward.target_keep_ratio if hasattr(self.config.reward, 'target_keep_ratio') else 0.45
        
        keep_ratio_reward = 0.0
        # Penalty for keeping too much (want 35-60%)
        if keep_ratio > 0.65:
            keep_ratio_reward = -30 * (keep_ratio - 0.65)
        elif keep_ratio < 0.30:
            keep_ratio_reward = -20 * (0.30 - keep_ratio)
        else:
            # Bonus for hitting target range
            keep_ratio_reward = 10 * (1 - abs(keep_ratio - target_keep) / 0.2)
        reward += keep_ratio_reward
        self.episode_reward_breakdown['keep_ratio'] = keep_ratio_reward
        
        # === 2. Section coherence ===
        # Reward for keeping consecutive sections
        kept_beats = sorted(self.edit_history.kept_beats)
        coherence_reward = 0.0
        if len(kept_beats) > 1:
            consecutive_runs = 1
            for i in range(1, len(kept_beats)):
                if kept_beats[i] == kept_beats[i-1] + 1:
                    consecutive_runs += 1
            coherence_score = consecutive_runs / len(kept_beats)
            coherence_reward = 15 * coherence_score
        reward += coherence_reward
        self.episode_reward_breakdown['coherence'] = coherence_reward
        
        # === 3. Phrase alignment ===
        # Bonus for cutting at phrase boundaries
        phrase_size = self.action_space_factored.phrase_size if self.action_space_factored else 8
        phrase_aligned_cuts = sum(
            1 for a in self.episode_actions 
            if a.action_type == ActionType.CUT and a.beat_index % phrase_size == 0
        )
        total_cuts = sum(1 for a in self.episode_actions if a.action_type == ActionType.CUT)
        alignment_reward = 0.0
        if total_cuts > 0:
            alignment_ratio = phrase_aligned_cuts / total_cuts
            alignment_reward = 10 * alignment_ratio
        reward += alignment_reward
        self.episode_reward_breakdown['alignment'] = alignment_reward
        
        # === 4. Action diversity ===
        action_types_used = set(a.action_type for a in self.episode_actions)
        diversity_reward = 0.0
        if len(action_types_used) >= 3:
            diversity_reward += 5
        if len(action_types_used) >= 5:
            diversity_reward += 5
        reward += diversity_reward
        self.episode_reward_breakdown['diversity'] = diversity_reward
        
        # === 5. Creative action bonus ===
        creative_types = {
            ActionType.FADE_IN, ActionType.FADE_OUT,
            ActionType.DOUBLE_TIME, ActionType.HALF_TIME,
            ActionType.REVERSE, ActionType.GAIN,
            ActionType.EQ_LOW, ActionType.EQ_HIGH,
            ActionType.DISTORTION, ActionType.REVERB,
            ActionType.REPEAT_PREV, ActionType.SWAP_NEXT,
        }
        n_creative = sum(1 for a in self.episode_actions if a.action_type in creative_types)
        creative_reward = 0.0
        if 1 <= n_creative <= 8:
            creative_reward = 5 * min(n_creative, 5)  # Up to 25 bonus
        elif n_creative > 8:
            creative_reward = -5 * (n_creative - 8)  # Penalty for overuse
        reward += creative_reward
        self.episode_reward_breakdown['creative'] = creative_reward
        
        # === 6. Size diversity ===
        sizes_used = set(a.action_size for a in self.episode_actions)
        size_diversity_reward = 0.0
        if len(sizes_used) >= 3:
            size_diversity_reward = 5
        reward += size_diversity_reward
        self.episode_reward_breakdown['size_diversity'] = size_diversity_reward
        
        # === 7. Efficiency bonus ===
        # Fewer actions = more efficient editing
        n_actions = len(self.episode_actions)
        expected_actions = n_beats / 4  # Ideal: ~bar-level decisions
        efficiency_reward = 0.0
        if n_actions < expected_actions * 0.8:
            efficiency_reward = 10  # Very efficient
        elif n_actions < expected_actions * 1.2:
            efficiency_reward = 5   # Reasonably efficient
        reward += efficiency_reward
        self.episode_reward_breakdown['efficiency'] = efficiency_reward
        
        # === 8. Learned reward model (RLHF) ===
        learned_reward = 0.0
        if self.learned_reward_model is not None:
            try:
                # Get edit decisions as input to reward model
                edit_decisions = []
                for action in self.episode_actions:
                    edit_decisions.append({
                        'type': action.action_type.value,
                        'size': action.action_size.value,
                        'amount': action.action_amount.value,
                        'beat_index': action.beat_index,
                    })
                
                # Call learned reward model
                # The model should return a scalar reward
                learned_reward = self.learned_reward_model.predict(
                    audio_state=self.audio_state,
                    edit_decisions=edit_decisions,
                    kept_beats=list(self.edit_history.kept_beats),
                    cut_beats=list(self.edit_history.cut_beats),
                )
                learned_reward = float(learned_reward)
                # Scale learned reward (typically in [-1, 1]) to match environment scale
                learned_reward *= 20.0
            except Exception as e:
                logger.debug(f"Learned reward model failed: {e}")
                learned_reward = 0.0
        reward += learned_reward
        self.episode_reward_breakdown['learned'] = learned_reward
        
        # Store total
        self.episode_reward_breakdown['total'] = reward
        
        return reward

    def set_learned_reward_model(self, model: Any) -> None:
        """Set learned reward model for RLHF."""
        self.learned_reward_model = model
    
    def set_audio_state(self, audio_state: AudioState) -> None:
        """Set audio state for next reset.
        
        Args:
            audio_state: New audio state to use
        """
        self.audio_state = audio_state
    
    def get_episode_reward_breakdown(self) -> Dict[str, float]:
        """Get detailed breakdown of episode reward components.
        
        Returns:
            Dictionary mapping component names to reward values
        """
        return self.episode_reward_breakdown.copy()

    def close(self) -> None:
        """Clean up resources."""
        pass


class VectorizedEnvFactored:
    """Wrapper for running multiple factored environments in parallel."""
    
    def __init__(self, config: Config, n_envs: int = 4):
        self.config = config
        self.n_envs = n_envs
        self.envs: List[Optional[AudioEditingEnvFactored]] = [None] * n_envs
        self.audio_states: List[Optional[AudioState]] = [None] * n_envs
    
    def set_audio_states(self, audio_states: List[AudioState]):
        """Set audio states for all environments."""
        for i, state in enumerate(audio_states[:self.n_envs]):
            self.audio_states[i] = state
            self.envs[i] = AudioEditingEnvFactored(self.config, audio_state=state)
    
    def reset_all(self) -> List[np.ndarray]:
        """Reset all environments."""
        obs_list = []
        for i, env in enumerate(self.envs):
            if env is not None and self.audio_states[i] is not None:
                obs, _ = env.reset(options={"audio_state": self.audio_states[i]})
                obs_list.append(obs)
            else:
                obs_list.append(np.zeros(100, dtype=np.float32))
        return obs_list
    
    def step_all(
        self, actions: List[Tuple[int, int, int]]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[Dict]]:
        """Step all environments."""
        next_obs = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if env is not None:
                action_array = np.array(action, dtype=np.int64)
                obs, reward, terminated, truncated, info = env.step(action_array)
                next_obs.append(obs)
                rewards.append(reward)
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos.append(info)
            else:
                next_obs.append(np.zeros(100, dtype=np.float32))
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                infos.append({})
        
        return next_obs, rewards, terminateds, truncateds, infos
    
    def get_action_masks(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Get action masks for all environments."""
        type_masks = []
        size_masks = []
        amount_masks = []
        
        for env in self.envs:
            if env is not None:
                tm, sm, am = env.get_action_masks()
            else:
                tm = np.ones(N_ACTION_TYPES, dtype=bool)
                sm = np.ones(N_ACTION_SIZES, dtype=bool)
                am = np.ones(N_ACTION_AMOUNTS, dtype=bool)
            type_masks.append(tm)
            size_masks.append(sm)
            amount_masks.append(am)
        
        return type_masks, size_masks, amount_masks
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            if env is not None:
                env.close()
