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
import os
import time
from pathlib import Path

from rl_editor.config import Config
from rl_editor.state import AudioState, StateRepresentation
from rl_editor.actions import (
    ActionType, ActionSize, ActionAmount,
    FactoredAction, FactoredActionSpace, EditHistoryFactored,
    N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS,
    apply_factored_action,
    AMOUNT_TO_PITCH_UP, AMOUNT_TO_PITCH_DOWN, AMOUNT_TO_DB,
)
from rl_editor.reward import RewardCalculator

logger = logging.getLogger(__name__)


# Feature indices (based on default FeatureConfig order)
# These assume default feature extraction settings
FEATURE_RMS_IDX = 1  # Energy/loudness
FEATURE_CHROMA_START = 40  # Start of 12 chroma bins (pitch class)
FEATURE_CHROMA_END = 52    # End of chroma bins


def _get_chroma_from_beat(beat_features: np.ndarray, beat_idx: int) -> Optional[np.ndarray]:
    """Extract chroma (pitch class) features from a beat.
    
    Returns normalized 12-dimensional chroma vector or None if unavailable.
    """
    if beat_features is None or beat_idx < 0 or beat_idx >= len(beat_features):
        return None
    
    feat_dim = beat_features.shape[1] if beat_features.ndim > 1 else 0
    if feat_dim < FEATURE_CHROMA_END:
        return None  # Not enough features
    
    chroma = beat_features[beat_idx, FEATURE_CHROMA_START:FEATURE_CHROMA_END]
    
    # Normalize to unit vector for cosine similarity
    norm = np.linalg.norm(chroma)
    if norm > 1e-6:
        return chroma / norm
    return chroma


def _get_rms_from_beat(beat_features: np.ndarray, beat_idx: int) -> Optional[float]:
    """Extract RMS (energy) from a beat."""
    if beat_features is None or beat_idx < 0 or beat_idx >= len(beat_features):
        return None
    
    feat_dim = beat_features.shape[1] if beat_features.ndim > 1 else 0
    if feat_dim <= FEATURE_RMS_IDX:
        return None
    
    return float(beat_features[beat_idx, FEATURE_RMS_IDX])


def _shift_chroma(chroma: np.ndarray, semitones: int) -> np.ndarray:
    """Shift chroma vector by semitones (circular rotation)."""
    return np.roll(chroma, semitones)


def _chroma_similarity(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """Compute cosine similarity between two chroma vectors."""
    dot = np.dot(chroma1, chroma2)
    return float(np.clip(dot, -1.0, 1.0))


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
        # Reward calculator instance (keep cache across episodes to avoid recompute)
        try:
            self._reward_calculator = RewardCalculator(self.config)
        except Exception:
            self._reward_calculator = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        # Use Gymnasium's proper seeding for isolated random state per environment
        super().reset(seed=seed)
        
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
        # Temporal smoothing diagnostics: count and total penalty applied this episode
        self._temporal_penalty_count = 0
        self._temporal_penalty_total = 0.0
        
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

        # --- Temporal smoothing penalty (apply before episode-end computation) ---
        temporal_penalty = 0.0
        try:
            prev_actions = self.episode_actions[:-1]
            window_beats = 2
            recent_count = 0
            for pa in prev_actions:
                if pa.beat_index >= factored_action.beat_index - window_beats:
                    recent_count += 1
            if recent_count > 0 and factored_action.n_beats < 4:
                temporal_penalty = -0.04 * min(3, recent_count)
                try:
                    self._temporal_penalty_count += 1
                    self._temporal_penalty_total += float(abs(temporal_penalty))
                except Exception:
                    pass
        except Exception:
            temporal_penalty = 0.0

        # Apply penalty to step reward and store
        step_reward = float(step_reward) + float(temporal_penalty)
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
            "temporal_penalty": float(temporal_penalty),
            "keep_ratio": self.edit_history.get_keep_ratio(),
            "n_section_decisions": len([a for a in self.episode_actions 
                                        if a.n_beats > 1]),
        }

        

        # Attach episode reward breakdown to info when episode ends so trainer can log it
        if terminated or truncated:
            info["reward_breakdown"] = self.episode_reward_breakdown.copy()
            # Log top negative components for quick diagnostics
            try:
                # Sort breakdown items by value (ascending) and pick most negative
                items = sorted(self.episode_reward_breakdown.items(), key=lambda kv: kv[1])
                negative = [(k, v) for k, v in items if isinstance(v, (int, float)) and v < 0]
                top_neg = negative[:3]
                #if top_neg:
                #    logger.warning(
                #        "Episode ended (beat=%d) negative breakdown top: %s",
                #        self.current_beat,
                #        ", ".join([f"{k}:{v:.3f}" for k, v in top_neg])
                #    )
            except Exception:
                logger.exception("Failed to log episode reward breakdown")

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
        from rl_editor.state import EditHistory
        edit_hist = EditHistory()
        edit_hist.kept_beats = set(self.edit_history.kept_beats)
        edit_hist.cut_beats = set(self.edit_history.cut_beats)
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
        """Compute step reward with strong keep ratio shaping.

        Most reward comes from episode end, but step rewards guide learning.
        """
        reward = 0.0
        keep_ratio = self.edit_history.get_keep_ratio()
        target_keep = 0.45
        progress = self.current_beat / max(1, len(self.audio_state.beat_times))

        # === Strong immediate feedback for CUT vs KEEP based on current ratio ===
        if action.action_type == ActionType.CUT:
            if keep_ratio > 0.70:
                reward += 0.3  # Strong bonus for cutting when way over target
            elif keep_ratio > 0.55:
                reward += 0.15  # Moderate bonus for cutting when over target
            elif keep_ratio < 0.35:
                reward -= 0.1  # Penalty for cutting when already under target
        elif action.action_type == ActionType.KEEP:
            if keep_ratio > 0.70:
                reward -= 0.2  # Penalty for keeping when way over target
            elif keep_ratio > 0.55:
                reward -= 0.1  # Penalty for keeping when over target
            elif keep_ratio < 0.35:
                reward += 0.1  # Bonus for keeping when under target

        # === Bonus for being in good range ===
        if 0.35 < keep_ratio < 0.55:
            reward += 0.1  # Bonus for being in target range

        # === Progress-based guidance ===
        # Early in episode: warn if keeping too much
        if progress < 0.3 and keep_ratio > 0.90:
            reward -= 0.15  # Early warning
        # Late in episode: stronger signals
        if progress > 0.7:
            if keep_ratio > 0.65:
                reward -= 0.2  # Late penalty for high keep ratio
            elif keep_ratio < 0.30:
                reward -= 0.1  # Penalty for cutting too much

        # === Section-level decisions bonus ===
        if action.n_beats >= 4:
            reward += 0.05

        # Small bonus for action diversity (not repeating same type)
        recent_types = set(a.action_type for a in self.episode_actions[-8:])
        if action.action_type not in recent_types:
            reward += 0.02    

        # Bonus for cutting or keeping at a phrase boundary
        phrase_size = self.action_space_factored.phrase_size if self.action_space_factored else 8
        if (action.action_type in (ActionType.CUT, ActionType.KEEP)) and (action.beat_index % phrase_size == 0):
            reward += 0.09

        # Small bonus for using creative actions
        creative_types = {
            ActionType.FADE_IN, ActionType.FADE_OUT,
            ActionType.SPEED_UP, ActionType.SPEED_DOWN,
            ActionType.REVERSE, ActionType.GAIN,
            ActionType.PITCH_UP, ActionType.PITCH_DOWN,
            ActionType.EQ_LOW, ActionType.EQ_HIGH,
            ActionType.DISTORTION, ActionType.REVERB,
            ActionType.REPEAT_PREV, ActionType.SWAP_NEXT,
        }
        rare_types = {ActionType.REVERSE, ActionType.SWAP_NEXT}

        if action.action_type in rare_types:
            reward -= 0.005  # Smaller bonus for recently used rare actions
        elif action.action_type in creative_types:
            reward += 0.001  # Smaller bonus if recently used
        

        reward += self._compute_step_smart_effect_reward(action) #max reward 0.15

        # --- Step-level reward: bonus for keeping a good section (consecutive kept beats) ---
        # Only applies to KEEP actions
        if action.action_type == ActionType.KEEP and self.edit_history is not None:
            # Check if the last N beats are all kept (N = phrase_size or bar)
            N = self.action_space_factored.phrase_size if self.action_space_factored else 8
            last_kept = [b for b in range(action.beat_index, action.beat_index + action.n_beats)]
            # Check if all these beats are in kept_beats
            if all(b in self.edit_history.kept_beats for b in last_kept):
                # Optionally, only reward if this is a new contiguous section (not overlapping previous)
                # Reward for keeping a full phrase or bar
                if action.n_beats >= 4:
                    reward += 0.01  # Bar
                if action.n_beats >= N:
                    reward += 0.02  # Phrase

        # --- Step-level continuity reward: bonus if pitch, gain, energy, and EQ match previous beat ---
        if self.audio_state is not None and self.audio_state.beat_features is not None:
            beat_features = self.audio_state.beat_features
            n_beats = len(self.audio_state.beat_times)
            idx = action.beat_index
            prev_idx = max(0, idx - 1)
            # Only reward for KEEP or similar actions (not CUT)
            if action.action_type in (ActionType.KEEP, ActionType.GAIN, ActionType.FADE_IN, ActionType.FADE_OUT, ActionType.SPEED_UP, ActionType.SPEED_DOWN, ActionType.PITCH_UP, ActionType.PITCH_DOWN, ActionType.EQ_LOW, ActionType.EQ_HIGH):
                # Pitch continuity (chroma cosine similarity)
                if beat_features.shape[1] >= 52:
                    chroma_now = beat_features[idx, 40:52]
                    chroma_prev = beat_features[prev_idx, 40:52]
                    norm_now = np.linalg.norm(chroma_now)
                    norm_prev = np.linalg.norm(chroma_prev)
                    if norm_now > 1e-6 and norm_prev > 1e-6:
                        chroma_sim = float(np.dot(chroma_now, chroma_prev) / (norm_now * norm_prev))
                        if chroma_sim > 0.95:
                            reward += 0.03  # Strong pitch continuity
                        elif chroma_sim > 0.85:
                            reward += 0.01  # Moderate pitch continuity
                # Gain/energy continuity (RMS)
                rms_now = float(beat_features[idx, 1])
                rms_prev = float(beat_features[prev_idx, 1])
                if rms_prev > 1e-6:
                    rms_ratio = rms_now / rms_prev
                    if 0.95 < rms_ratio < 1.05:
                        reward += 0.02  # Good gain/energy continuity
                    elif 0.90 < rms_ratio < 1.10:
                        reward += 0.01  # Acceptable continuity

                # EQ continuity/improvement reward for EQ actions
                ROLLOFF_IDX = 4
                BANDWIDTH_IDX = 5
                FLATNESS_IDX = 6
                if action.action_type in (ActionType.EQ_LOW, ActionType.EQ_HIGH):
                    rolloff_now = float(beat_features[idx, ROLLOFF_IDX])
                    rolloff_prev = float(beat_features[prev_idx, ROLLOFF_IDX])
                    bandwidth_now = float(beat_features[idx, BANDWIDTH_IDX])
                    bandwidth_prev = float(beat_features[prev_idx, BANDWIDTH_IDX])
                    flatness_now = float(beat_features[idx, FLATNESS_IDX])
                    flatness_prev = float(beat_features[prev_idx, FLATNESS_IDX])

                    # Reward for smooth transitions (continuity)
                    rolloff_diff = abs(rolloff_now - rolloff_prev)
                    bandwidth_diff = abs(bandwidth_now - bandwidth_prev)
                    flatness_diff = abs(flatness_now - flatness_prev)

                    # Reward for small changes (continuity)
                    if rolloff_diff < 0.05:
                        reward += 0.01
                    if bandwidth_diff < 0.05:
                        reward += 0.01
                    if flatness_diff < 0.05:
                        reward += 0.01

                    # Reward for appropriate direction (optional, e.g. boosting highs should increase rolloff)
                    if action.action_type == ActionType.EQ_HIGH:
                        if rolloff_now > rolloff_prev:
                            reward += 0.01
                        if bandwidth_now > bandwidth_prev:
                            reward += 0.005
                        if flatness_now > flatness_prev:
                            reward += 0.005
                    elif action.action_type == ActionType.EQ_LOW:
                        if rolloff_now < rolloff_prev:
                            reward += 0.01
                        if bandwidth_now < bandwidth_prev:
                            reward += 0.005
                        if flatness_now < flatness_prev:
                            reward += 0.005

        return reward * 10.0  # Scale up step reward
    
    def _compute_step_smart_effect_reward(self, action: FactoredAction) -> float:
        """Step-level reward for appropriate use of pitch/gain/speed effects."""
        if self.audio_state is None or self.audio_state.beat_features is None:
            return 0.0

        beat_features = self.audio_state.beat_features
        n_beats = len(self.audio_state.beat_times)
        beat_idx = action.beat_index
        next_beat_idx = min(beat_idx + action.n_beats, n_beats - 1)
        prev_beat_idx = max(beat_idx - 1, 0)
        reward = 0.0

        # === PITCH_UP / PITCH_DOWN ===
        if action.action_type in (ActionType.PITCH_UP, ActionType.PITCH_DOWN):
            current_chroma = _get_chroma_from_beat(beat_features, beat_idx)
            next_chroma = _get_chroma_from_beat(beat_features, next_beat_idx)
            if current_chroma is not None and next_chroma is not None:
                if action.action_type == ActionType.PITCH_UP:
                    semitones = AMOUNT_TO_PITCH_UP[action.action_amount]
                else:
                    semitones = AMOUNT_TO_PITCH_DOWN[action.action_amount]
                original_sim = _chroma_similarity(current_chroma, next_chroma)
                shifted_chroma = _shift_chroma(current_chroma, semitones)
                shifted_sim = _chroma_similarity(shifted_chroma, next_chroma)
                improvement = shifted_sim - original_sim
                if improvement > 0.1:
                    reward += 0.15  # Good pitch shift
                elif improvement > 0:
                    reward += 0.05  # Slight improvement
                elif improvement < -0.2:
                    reward -= 0.10  # Made it worse

        # === GAIN ===
        elif action.action_type == ActionType.GAIN:
            current_rms = _get_rms_from_beat(beat_features, beat_idx)
            prev_rms = _get_rms_from_beat(beat_features, prev_beat_idx)
            next_rms = _get_rms_from_beat(beat_features, next_beat_idx)
            if current_rms is not None and prev_rms is not None and next_rms is not None:
                db_change = AMOUNT_TO_DB[action.action_amount]
                gain_factor = 10 ** (db_change / 20)
                adjusted_rms = current_rms * gain_factor
                target_rms = (prev_rms + next_rms) / 2
                original_error = abs(current_rms - target_rms)
                adjusted_error = abs(adjusted_rms - target_rms)
                if adjusted_error < original_error * 0.7:
                    reward += 0.10  # Good gain adjustment
                elif adjusted_error < original_error:
                    reward += 0.03  # Slight improvement
                elif adjusted_error > original_error * 1.3:
                    reward -= 0.07  # Made transition worse

        # === SPEED_UP / SPEED_DOWN ===
        elif action.action_type in (ActionType.SPEED_UP, ActionType.SPEED_DOWN):
            # Small bonus for using varied amounts (not always default)
            if action.action_amount != ActionAmount.NEUTRAL:
                reward += 0.02

        return reward * 0.1

    def _compute_episode_reward(self) -> float:
        """Compute episode-end reward based on edit quality.
        
        Stores breakdown in self.episode_reward_breakdown.
        """
        #logger.info("_compute_episode_reward: starting computation for pair_id=%s", getattr(self.audio_state, 'pair_id', 'noid'))
        self.episode_reward_breakdown = {}  # Reset breakdown
        reward = 0.0
        n_beats = len(self.audio_state.beat_times)
        
        # === 1. Keep ratio constraint with reward shaping ===
        keep_ratio = self.edit_history.get_keep_ratio()
        target_keep = self.config.reward.target_keep_ratio if hasattr(self.config.reward, 'target_keep_ratio') else 0.45

        # Reward shaping: progressive reward/penalty based on distance from target
        # Target: 45%, acceptable range: 35-55%
        distance_from_target = abs(keep_ratio - target_keep)

        keep_ratio_reward = 0.0

        # Strong positive reward for being close to target (max +20 at exact target)
        if distance_from_target < 0.10:
            # Within 10% of target: positive reward
            keep_ratio_reward = 20 * (1.0 - distance_from_target / 0.10)
        elif distance_from_target < 0.20:
            # Within 20%: small positive reward
            keep_ratio_reward = 5 * (1.0 - (distance_from_target - 0.10) / 0.10)
        else:
            # Beyond 20%: quadratic penalty (grows faster as distance increases)
            excess = distance_from_target - 0.20
            keep_ratio_reward = -100 * (excess ** 1.5)  # Quadratic-ish penalty

        # Extra asymmetric penalty for keeping too much (model's tendency)
        if keep_ratio > 0.60:
            # Additional penalty that grows with keep ratio
            over_keep = keep_ratio - 0.60
            keep_ratio_reward -= 50 * over_keep  # -50 at 100% keep, -20 at 80%

        reward += keep_ratio_reward
        self.episode_reward_breakdown['duration'] = keep_ratio_reward
        self.episode_reward_breakdown['n_keep_ratio'] = keep_ratio
        
        # === 2. Section coherence ===
        # Reward for keeping consecutive sections (but ONLY if actually cutting meaningfully)
        kept_beats = sorted(self.edit_history.kept_beats)
        coherence_reward = 0.0
        if len(kept_beats) > 1:
            consecutive_runs = 1
            for i in range(1, len(kept_beats)):
                if kept_beats[i] == kept_beats[i-1] + 1:
                    consecutive_runs += 1
            coherence_score = consecutive_runs / len(kept_beats)
            coherence_reward = 15 * coherence_score

            # Coherence reward is DISABLED when keeping too much (no reward for trivial coherence)
            # Sharp cutoff: 0 reward if keep_ratio > 0.60, scales linearly from 0.45 to 0.60
            if keep_ratio > 0.60:
                coherence_reward = 0.0  # Complete disable
            elif keep_ratio > target_keep:
                # Linear scale from full (at target) to 0 (at 0.60)
                scale = 1.0 - (keep_ratio - target_keep) / (0.60 - target_keep)
                coherence_reward *= max(0.0, scale)
        reward += coherence_reward
        self.episode_reward_breakdown['coherence'] = coherence_reward

        # === 2b. Ground truth matching (supervised signal from human edits) ===
        # Compare model's decisions against target_labels if available
        gt_match_reward = 0.0
        if hasattr(self.audio_state, 'target_labels') and self.audio_state.target_labels is not None:
            target_labels = self.audio_state.target_labels
            n_target = len(target_labels)

            # Build prediction array from edit history
            pred_labels = np.zeros(n_target, dtype=np.float32)
            for b in self.edit_history.kept_beats:
                if b < n_target:
                    pred_labels[b] = 1.0
            # cut_beats are implicitly 0

            # Compute match rate
            matches = (pred_labels == target_labels).sum()
            match_rate = matches / n_target if n_target > 0 else 0.0

            # Strong positive reward for matching human edits
            # Max +30 for perfect match, scales down linearly
            # Baseline ~50% match rate (random) = 0 reward
            baseline_match = 0.5
            if match_rate > baseline_match:
                gt_match_reward = 60 * (match_rate - baseline_match)  # Up to +30 at 100%
            else:
                gt_match_reward = 30 * (match_rate - baseline_match)  # Smaller penalty for below baseline

            self.episode_reward_breakdown['gt_match_rate'] = float(match_rate)
        reward += gt_match_reward
        self.episode_reward_breakdown['gt_match'] = gt_match_reward

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
            diversity_reward += 2
        elif len(action_types_used) >= 5:
            diversity_reward += 5
        reward += diversity_reward
        self.episode_reward_breakdown['diversity'] = diversity_reward
        
        # === 5. Size diversity ===
        sizes_used = set(a.action_size for a in self.episode_actions)
        size_diversity_reward = 0.0
        if len(sizes_used) >= 5:
            size_diversity_reward = 4
        elif len(sizes_used) >= 3:
            size_diversity_reward = 2
        reward += size_diversity_reward
        self.episode_reward_breakdown['size_diversity'] = size_diversity_reward
        

        # === 6. Efficiency bonus ===
        # Fewer actions = more efficient editing
        n_actions = len(self.episode_actions)
        expected_actions = n_beats / 4  # Ideal: ~bar-level decisions
        efficiency_reward = 0.0
        if n_actions < expected_actions * 0.8:
            efficiency_reward = 10  # Very efficient (will be scaled down)
        elif n_actions < expected_actions * 1.2:
            efficiency_reward = 5   # Reasonably efficient (will be scaled down)
        # Scale down to make episode-level counts less dominant
        efficiency_reward = float(np.clip(efficiency_reward * 0.2, -3.0, 3.0))
        reward += efficiency_reward
        self.episode_reward_breakdown['efficiency'] = efficiency_reward

        # === 7. Creative action bonus ===
        creative_types = {
            ActionType.FADE_IN, ActionType.FADE_OUT,
            ActionType.SPEED_UP, ActionType.SPEED_DOWN,
            ActionType.REVERSE, ActionType.GAIN,
            ActionType.PITCH_UP, ActionType.PITCH_DOWN,
            ActionType.EQ_LOW, ActionType.EQ_HIGH,
            ActionType.DISTORTION, ActionType.REVERB,
            ActionType.REPEAT_PREV, ActionType.SWAP_NEXT,
        }
        n_creative = sum(1 for a in self.episode_actions if a.action_type in creative_types)
        creative_ratio = n_creative / max(1, len(self.episode_actions))
        # Compute creative reward as a normalized per-action signal, then clip
        # Encourage moderate creativity (~0.45) and penalize extreme over-use
        raw_creative_score = (0.45 - creative_ratio) * 8.0  # centered near 0.45
        creative_reward = float(np.clip(raw_creative_score, -2.0, 2.0))
        reward += creative_reward
        
        self.episode_reward_breakdown['creative'] = creative_reward
        self.episode_reward_breakdown['n_creative'] = n_creative
        self.episode_reward_breakdown['n_actions'] = n_actions

        # === 7b. Action density per 8 beats ===
        # Penalize or reward based on number of actions per 8 beats (action density)
        actions_per_8 = [0] * ((n_beats + 7) // 8)
        for a in self.episode_actions:
            bin_idx = a.beat_index // 8
            if bin_idx < len(actions_per_8):
                actions_per_8[bin_idx] += 1
        avg_actions_per_8 = np.mean(actions_per_8) if actions_per_8 else 0.0
        # Ideal: 1-3 actions per 8 beats (encourages section-level editing, not micro-edits)
        action_density_reward = 0.0
        if avg_actions_per_8 < 1.0:
            action_density_reward = -5.0 * (1.0 - avg_actions_per_8)  # Too sparse
        elif avg_actions_per_8 > 3.0:
            action_density_reward = -5.0 * (avg_actions_per_8 - 3.0)  # Too dense
        else:
            action_density_reward = 5.0  # In the sweet spot
        # Clip density reward to avoid large-magnitude effects from skewed counts
        action_density_reward = float(np.clip(action_density_reward, -3.0, 3.0))
        reward += action_density_reward
        self.episode_reward_breakdown['action_density'] = action_density_reward
        
        # === 8. Smart effect rewards (pitch/gain appropriateness) ===
        smart_effect_reward = self._compute_smart_effect_reward()
        # Scale and clip smart_effect_reward to avoid large spikes from many effects
        smart_effect_contrib = float(np.clip(smart_effect_reward / 10.0, -3.0, 3.0))
        reward += smart_effect_contrib
        # Store both raw and applied values for diagnostics
        self.episode_reward_breakdown['smart_effects'] = smart_effect_contrib
        
        # === 9. Learned reward model (RLHF) ===
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
        
        # === 10. Reconstruction reward (compare produced per-beat mel to target if available) ===
        # Ensure the reconstruction key is always present so logs show 0 when unavailable.
        self.episode_reward_breakdown.setdefault('reconstruction', 0.0)
        try:
            recon_reward = 0.0
            pred_mel_list = []
            mel_source = getattr(self.audio_state, 'mel_spectrogram', None)
            # Use RewardCalculator helper to build or fetch cached per-beat target mel
            # Use persistent reward calculator (created in __init__) to leverage cache
            rc_helper = getattr(self, '_reward_calculator', None)
            if rc_helper is None:
                rc_helper = RewardCalculator(self.config)
                self._reward_calculator = rc_helper
            orig_per_beat = rc_helper.get_target_per_beat(self.audio_state)

            # Record availability flag and dump minimal debug NPZ if missing
            if orig_per_beat is None:
                self.episode_reward_breakdown['reconstruction_available'] = 0
                logger.debug(
                    "_compute_episode_reward: reconstruction unavailable for pair_id=%s (raw_audio_len=%s, mel_spectrogram=%s, target_mel=%s)",
                    getattr(self.audio_state, 'pair_id', 'noid'),
                    0 if getattr(self.audio_state, 'raw_audio', None) is None else len(self.audio_state.raw_audio),
                    None if getattr(self.audio_state, 'mel_spectrogram', None) is None else getattr(self.audio_state, 'mel_spectrogram').shape,
                    None if getattr(self.audio_state, 'target_mel', None) is None else np.array(getattr(self.audio_state, 'target_mel')).shape,
                )
            else:
                self.episode_reward_breakdown['reconstruction_available'] = 1

            if orig_per_beat is not None and len(self.episode_actions) > 0:
                # Light debug: log action counts at DEBUG level
                try:
                    action_counts = {}
                    for a in self.episode_actions:
                        action_counts[a.action_type.name] = action_counts.get(a.action_type.name, 0) + 1
                    logger.debug(
                        "_compute_episode_reward: assembling pred_mel_list: n_actions=%d action_counts=%s orig_per_beat_shape=%s",
                        len(self.episode_actions), action_counts, None if orig_per_beat is None else orig_per_beat.shape,
                    )
                except Exception:
                    logger.debug("Failed to stringify action summary for reconstruction")

                # Vectorized assembly: compute total predicted segments, preallocate array, fill sequentially
                n_mels = orig_per_beat.shape[1]
                # First pass: compute total predicted segments
                total_pred = 0
                for a in self.episode_actions:
                    n = a.n_beats
                    if a.action_type == a.action_type.CUT:
                        continue
                    name = a.action_type.name
                    if name == 'LOOP':
                        total_pred += n * 2
                    else:
                        total_pred += n

                if total_pred > 0:
                    pred_mel = np.zeros((total_pred, n_mels), dtype=orig_per_beat.dtype)
                    write_idx = 0
                    # track last written slice for REPEAT_PREV
                    last_slice_start = 0
                    last_slice_len = 0
                    for a in self.episode_actions:
                        start = a.beat_index
                        n = a.n_beats
                        seg = orig_per_beat[start:start + n]
                        if a.action_type == a.action_type.CUT:
                            continue
                        name = a.action_type.name
                        if name == 'KEEP' or name == 'SWAP_NEXT' or name == 'REORDER' or name == 'REVERSE' or name == 'SPEED_UP' or name == 'SPEED_DOWN' or name == 'DISTORTION' or name == 'REVERB' or name == 'PITCH_UP' or name == 'PITCH_DOWN' or name == 'REPEAT_PREV':
                            # default: copy segment
                            if name == 'REPEAT_PREV':
                                if last_slice_len >= n:
                                    src_start = last_slice_start + last_slice_len - n
                                    pred_mel[write_idx:write_idx + n] = pred_mel[src_start:src_start + n]
                                else:
                                    pred_mel[write_idx:write_idx + seg.shape[0]] = seg
                            else:
                                pred_mel[write_idx:write_idx + seg.shape[0]] = seg
                            last_slice_start = write_idx
                            last_slice_len = seg.shape[0]
                            write_idx += seg.shape[0]
                        elif name == 'LOOP':
                            # write seg twice
                            pred_mel[write_idx:write_idx + seg.shape[0]] = seg
                            pred_mel[write_idx + seg.shape[0]:write_idx + 2 * seg.shape[0]] = seg
                            last_slice_start = write_idx
                            last_slice_len = seg.shape[0] * 2
                            write_idx += seg.shape[0] * 2
                        elif name == 'FADE_IN':
                            L = seg.shape[0]
                            if L > 0:
                                weights = (np.arange(L) + 1) / float(max(1, L))
                                pred_mel[write_idx:write_idx + L] = seg * weights[:, None]
                                last_slice_start = write_idx
                                last_slice_len = L
                                write_idx += L
                        elif name == 'FADE_OUT':
                            L = seg.shape[0]
                            if L > 0:
                                weights = 1.0 - (np.arange(L) / float(max(1, L)))
                                pred_mel[write_idx:write_idx + L] = seg * weights[:, None]
                                last_slice_start = write_idx
                                last_slice_len = L
                                write_idx += L
                        elif name == 'GAIN':
                            db = getattr(a, 'db_change', 0.0)
                            factor = 10 ** (db / 20.0)
                            pred_mel[write_idx:write_idx + seg.shape[0]] = seg * factor
                            last_slice_start = write_idx
                            last_slice_len = seg.shape[0]
                            write_idx += seg.shape[0]
                        elif name in ('EQ_HIGH', 'EQ_LOW'):
                            k = max(1, n_mels // 3)
                            if name == 'EQ_HIGH':
                                modified = seg.copy()
                                modified[:, -k:] = modified[:, -k:] * 1.15
                            else:
                                modified = seg.copy()
                                modified[:, :k] = modified[:, :k] * 1.15
                            pred_mel[write_idx:write_idx + seg.shape[0]] = modified
                            last_slice_start = write_idx
                            last_slice_len = seg.shape[0]
                            write_idx += seg.shape[0]
                        else:
                            # fallback copy
                            pred_mel[write_idx:write_idx + seg.shape[0]] = seg
                            last_slice_start = write_idx
                            last_slice_len = seg.shape[0]
                            write_idx += seg.shape[0]

                    # Trim to actual written length if any mismatch
                    if write_idx < total_pred:
                        pred_mel = pred_mel[:write_idx]
                else:
                    pred_mel = np.zeros((0, n_mels), dtype=orig_per_beat.dtype)

                # Record diagnostic counts so subprocesses can communicate diagnostics
                self.episode_reward_breakdown['n_episode_actions'] = float(len(self.episode_actions))
                # pred_mel is the preallocated/filled array of predicted segments
                self.episode_reward_breakdown['n_pred_mel_segments'] = float(pred_mel.shape[0])
                if orig_per_beat is not None:
                    self.episode_reward_breakdown['orig_per_beat_n'] = float(orig_per_beat.shape[0])
                    self.episode_reward_breakdown['orig_per_beat_dim'] = float(orig_per_beat.shape[1])

                if pred_mel.shape[0] > 0:
                    # Use persistent reward calculator helper when available
                    rc = rc_helper if rc_helper is not None else RewardCalculator(self.config)
                    recon_reward = rc.compute_reconstruction_reward(self.audio_state, edited_mel=pred_mel, per_beat=True)
                    reward += recon_reward
                    self.episode_reward_breakdown['reconstruction'] = float(recon_reward)
                    # Also record pred_mel shape
                    self.episode_reward_breakdown['pred_mel_n'] = float(pred_mel.shape[0])
                    self.episode_reward_breakdown['pred_mel_dim'] = float(pred_mel.shape[1])
                    # Record calculator diagnostics if available
                    try:
                        if hasattr(rc, '_last_l1') and rc._last_l1 is not None:
                            self.episode_reward_breakdown['recon_l1'] = float(rc._last_l1)
                        if hasattr(rc, '_last_score') and rc._last_score is not None:
                            self.episode_reward_breakdown['recon_score'] = float(rc._last_score)
                        if hasattr(rc, '_last_reward') and rc._last_reward is not None:
                            self.episode_reward_breakdown['recon_reward_raw'] = float(rc._last_reward)
                    except Exception:
                        logger.exception("Failed to attach recon diagnostics to breakdown")
                    logger.debug("_compute_episode_reward: pred_mel_n=%d pred_mel_shape=%s recon_reward=%.6f",
                                 pred_mel.shape[0], pred_mel.shape, float(recon_reward))
                else:
                    logger.info("_compute_episode_reward: pred_mel is empty after assembling from actions")
        except Exception:
            # If any failure, skip reconstruction reward but keep the key (0)
            logger.exception("Reconstruction reward computation failed")
        # Attach temporal smoothing diagnostics recorded during the episode
        try:
            self.episode_reward_breakdown['temporal_penalty_count'] = float(getattr(self, '_temporal_penalty_count', 0))
            self.episode_reward_breakdown['temporal_penalty_total'] = float(getattr(self, '_temporal_penalty_total', 0.0))
        except Exception:
            pass
        # Store total and log breakdown for debugging
        self.episode_reward_breakdown['total'] = reward
        #logger.info("_compute_episode_reward: finished pair_id=%s total=%.6f breakdown=%s",
        #         getattr(self.audio_state, 'pair_id', 'noid'), float(reward), {k: float(v) for k, v in self.episode_reward_breakdown.items()})

        return reward

    def _compute_smart_effect_reward(self) -> float:
        """Compute reward for appropriate use of pitch/gain effects.
        
        Rewards:
        - PITCH_UP/PITCH_DOWN: Better if shifted pitch aligns harmonically with next beat
        - GAIN: Better if gain change creates smoother energy transitions
        - SPEED_UP/SPEED_DOWN: Small bonus for using amount appropriately
        
        Returns:
            Smart effect reward (can be positive or negative)
        """
        if self.audio_state is None or self.audio_state.beat_features is None:
            return 0.0
        
        beat_features = self.audio_state.beat_features
        n_beats = len(self.audio_state.beat_times)
        
        pitch_reward = 0.0
        gain_reward = 0.0
        speed_reward = 0.0
        n_pitch_actions = 0
        n_gain_actions = 0
        n_speed_actions = 0
        
        for action in self.episode_actions:
            beat_idx = action.beat_index
            next_beat_idx = min(beat_idx + action.n_beats, n_beats - 1)
            prev_beat_idx = max(beat_idx - 1, 0)
            
            # === PITCH_UP / PITCH_DOWN ===
            if action.action_type in (ActionType.PITCH_UP, ActionType.PITCH_DOWN):
                current_chroma = _get_chroma_from_beat(beat_features, beat_idx)
                next_chroma = _get_chroma_from_beat(beat_features, next_beat_idx)
                
                if current_chroma is not None and next_chroma is not None:
                    # Get semitone shift
                    if action.action_type == ActionType.PITCH_UP:
                        semitones = AMOUNT_TO_PITCH_UP[action.action_amount]
                    else:
                        semitones = AMOUNT_TO_PITCH_DOWN[action.action_amount]
                    
                    # Compute similarity before and after shift
                    original_sim = _chroma_similarity(current_chroma, next_chroma)
                    shifted_chroma = _shift_chroma(current_chroma, semitones)
                    shifted_sim = _chroma_similarity(shifted_chroma, next_chroma)
                    
                    # Reward if shift improved harmonic alignment
                    improvement = shifted_sim - original_sim
                    if improvement > 0.1:
                        pitch_reward += 3.0  # Good pitch shift
                    elif improvement > 0:
                        pitch_reward += 1.0  # Slight improvement
                    elif improvement < -0.2:
                        pitch_reward -= 2.0  # Made it worse
                    
                    n_pitch_actions += 1
            
            # === GAIN ===
            elif action.action_type == ActionType.GAIN:
                current_rms = _get_rms_from_beat(beat_features, beat_idx)
                prev_rms = _get_rms_from_beat(beat_features, prev_beat_idx)
                next_rms = _get_rms_from_beat(beat_features, next_beat_idx)
                
                if current_rms is not None and prev_rms is not None and next_rms is not None:
                    db_change = AMOUNT_TO_DB[action.action_amount]
                    
                    # Convert dB to linear scale for RMS comparison
                    gain_factor = 10 ** (db_change / 20)
                    adjusted_rms = current_rms * gain_factor
                    
                    # Compute how well the adjusted RMS fits between neighbors
                    # Target: smooth transition (closer to average of prev and next)
                    target_rms = (prev_rms + next_rms) / 2
                    
                    # How close is adjusted_rms to target vs original?
                    original_error = abs(current_rms - target_rms)
                    adjusted_error = abs(adjusted_rms - target_rms)
                    
                    if adjusted_error < original_error * 0.7:
                        gain_reward += 2.0  # Good gain adjustment
                    elif adjusted_error < original_error:
                        gain_reward += 0.5  # Slight improvement
                    elif adjusted_error > original_error * 1.3:
                        gain_reward -= 1.5  # Made transition worse
                    
                    n_gain_actions += 1
            
            # === SPEED_UP / SPEED_DOWN ===
            elif action.action_type in (ActionType.SPEED_UP, ActionType.SPEED_DOWN):
                # Small bonus for using varied amounts (not always default)
                if action.action_amount != ActionAmount.NEUTRAL:
                    speed_reward += 0.5
                n_speed_actions += 1
        
        # Normalize by number of actions (avoid over-rewarding many effects)
        total_reward = 0.0
        if n_pitch_actions > 0:
            total_reward += min(pitch_reward, 18.0)  # Cap at 18
        if n_gain_actions > 0:
            total_reward += min(gain_reward, 13.0)    # Cap at 13
        if n_speed_actions > 0:
            total_reward += min(speed_reward, 9.0)   # Cap at 9

        return total_reward

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
