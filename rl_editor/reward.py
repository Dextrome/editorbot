"""Reward signals for RL-based audio editor.

Implements three complementary signals:
1. Sparse rewards - Human feedback
2. Dense rewards - Automatic metrics
3. Learned rewards - Human preference model
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import librosa

from .config import Config, RewardConfig
from .utils import estimate_tempo, get_energy_contour
from .utils import compute_mel_spectrogram as utils_compute_mel


@dataclass
class RewardComponents:
    """Components of computed reward."""

    total_reward: float
    sparse_reward: Optional[float] = None
    tempo_consistency: float = 0.0
    energy_flow: float = 0.0
    phrase_completeness: float = 0.0
    transition_quality: float = 0.0
    dense_reward: float = 0.0
    learned_reward: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total": self.total_reward,
            "sparse": self.sparse_reward,
            "tempo_consistency": self.tempo_consistency,
            "energy_flow": self.energy_flow,
            "phrase_completeness": self.phrase_completeness,
            "transition_quality": self.transition_quality,
            "dense": self.dense_reward,
            "learned": self.learned_reward,
        }


class RewardCalculator:
    """Computes rewards for audio edits."""

    def __init__(self, config: Config) -> None:
        """Initialize reward calculator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.reward_config: RewardConfig = config.reward
        self.audio_config = config.audio

    def compute_dense_reward(
        self,
        edited_audio: np.ndarray,
        original_audio: np.ndarray,
        beat_times: np.ndarray,
        beat_frames: np.ndarray,
        sr: int = 22050,
    ) -> RewardComponents:
        """Compute dense reward from automatic metrics.

        Combines:
        - Tempo consistency: BPM stays similar throughout
        - Energy flow: Smooth dynamics without jarring drops
        - Phrase completeness: Respect musical phrases
        - Transition quality: Beats align at boundaries

        Args:
            edited_audio: Edited audio array
            original_audio: Original audio array
            beat_times: Beat times in original
            beat_frames: Beat frames in original
            sr: Sample rate

        Returns:
            RewardComponents with computed dense metrics
        """
        components = RewardComponents(total_reward=0.0)

        try:
            # 1. Tempo consistency
            original_tempo = estimate_tempo(original_audio, sr)
            edited_tempo = estimate_tempo(edited_audio, sr)
            tempo_diff = abs(original_tempo - edited_tempo)
            # Normalize: penalty for difference > 5 BPM
            tempo_consistency = max(0.0, 1.0 - (tempo_diff / 5.0))
            components.tempo_consistency = float(tempo_consistency)

            # 2. Energy flow smoothness
            original_energy = get_energy_contour(original_audio, sr)
            edited_energy = get_energy_contour(edited_audio, sr)

            # Compute energy variance (lower is smoother)
            original_energy_var = np.var(original_energy)
            edited_energy_var = np.var(edited_energy)

            if original_energy_var > 0:
                energy_flow = max(
                    0.0, 1.0 - (edited_energy_var / (original_energy_var + 1e-6))
                )
            else:
                energy_flow = 1.0
            components.energy_flow = float(energy_flow)

            # 3. Phrase completeness
            # Simple heuristic: penalty if beats are heavily edited at phrase boundaries
            # Assume 8-beat phrases (standard)
            phrase_length_beats = 8
            n_beats = len(beat_times)
            phrase_boundary_penalty = 0.0

            if n_beats > phrase_length_beats:
                for i in range(phrase_length_beats, n_beats, phrase_length_beats):
                    # Check if beat at phrase boundary is kept
                    if i < n_beats:
                        phrase_boundary_penalty += 0.05
                phrase_completeness = max(0.0, 1.0 - phrase_boundary_penalty)
            else:
                phrase_completeness = 1.0
            components.phrase_completeness = float(phrase_completeness)

            # 4. Transition quality (from beat alignment)
            # Heuristic: higher if audio length matches beat grid
            expected_duration = beat_times[-1] if len(beat_times) > 0 else len(original_audio) / sr
            actual_duration = len(edited_audio) / sr
            duration_diff = abs(expected_duration - actual_duration)
            transition_quality = max(0.0, 1.0 - (duration_diff / (expected_duration + 1e-6)))
            components.transition_quality = float(transition_quality)

            # Combine dense rewards with weights
            components.dense_reward = (
                self.reward_config.tempo_consistency_weight * components.tempo_consistency
                + self.reward_config.energy_flow_weight * components.energy_flow
                + self.reward_config.phrase_completeness_weight * components.phrase_completeness
                + self.reward_config.transition_quality_weight * components.transition_quality
            )
            # Normalize by total weight
            total_weight = (
                self.reward_config.tempo_consistency_weight
                + self.reward_config.energy_flow_weight
                + self.reward_config.phrase_completeness_weight
                + self.reward_config.transition_quality_weight
            )
            components.dense_reward /= total_weight

        except Exception as e:
            # If computation fails, return zero components
            components.dense_reward = 0.0

        components.total_reward = components.dense_reward
        return components

    def compute_sparse_reward(self, human_rating: float) -> RewardComponents:
        """Compute sparse reward from human feedback.

        Args:
            human_rating: Human rating 1-10 (or -1 for no feedback)

        Returns:
            RewardComponents with sparse reward
        """
        components = RewardComponents(total_reward=0.0)

        if human_rating >= 0:
            # Normalize to [-1, 1]
            sparse_reward = (human_rating - 5.0) / 5.0
            components.sparse_reward = float(sparse_reward)
            components.total_reward = components.sparse_reward
        else:
            components.sparse_reward = None

        return components

    def compute_combined_reward(
        self,
        edited_audio: np.ndarray,
        original_audio: np.ndarray,
        beat_times: np.ndarray,
        beat_frames: np.ndarray,
        human_rating: Optional[float] = None,
        learned_reward: Optional[float] = None,
        sr: int = 22050,
    ) -> RewardComponents:
        """Compute combined reward from all available signals.

        Args:
            edited_audio: Edited audio array
            original_audio: Original audio array
            beat_times: Beat times in original
            beat_frames: Beat frames in original
            human_rating: Optional human rating 1-10
            learned_reward: Optional learned reward from preference model
            sr: Sample rate

        Returns:
            RewardComponents with all metrics
        """
        # Start with dense rewards
        components = self.compute_dense_reward(
            edited_audio, original_audio, beat_times, beat_frames, sr
        )

        # Add sparse reward if available
        if human_rating is not None and human_rating >= 0:
            sparse_components = self.compute_sparse_reward(human_rating)
            components.sparse_reward = sparse_components.sparse_reward

        # Add learned reward if available
        if learned_reward is not None:
            components.learned_reward = learned_reward

        # Combine all signals
        total_reward = 0.0
        weight_sum = 0.0

        if self.reward_config.use_dense_rewards:
            total_reward += components.dense_reward
            weight_sum += 1.0

        if self.reward_config.use_sparse_rewards and components.sparse_reward is not None:
            total_reward += components.sparse_reward
            weight_sum += 1.0

        if self.reward_config.use_learned_rewards and components.learned_reward is not None:
            total_reward += components.learned_reward
            weight_sum += 1.0

        if weight_sum > 0:
            components.total_reward = total_reward / weight_sum
        else:
            components.total_reward = 0.0

        return components

    def compute_reconstruction_reward(
        self,
        audio_state,
        edited_mel: Optional[np.ndarray] = None,
        per_beat: bool = True,
    ) -> float:
        """Compute a small reconstruction reward based on mel distance.

        If `edited_mel` is provided, compute L1 between `edited_mel` and
        `audio_state.target_mel` (if available). For per-beat vectors, match
        by beat index; otherwise compute global mel and compare.
        Returns a scalar in roughly [-1, 1] scaled by config.reconstruction_weight.
        """
        try:
            cfg = self.reward_config
            # Prefer passed edited_mel, otherwise try audio_state.target_mel
            target = edited_mel if edited_mel is not None else getattr(audio_state, "target_mel", None)
            if target is None:
                return 0.0

            # If target is per-beat and audio_state has per-beat mel, compute mean L1
            if per_beat and hasattr(audio_state, "target_mel") and audio_state.target_mel is not None:
                # audio_state.target_mel expected shape (n_beats, mel_dim)
                pred = np.array(audio_state.target_mel)
                tgt = np.array(target)
                # Align shapes
                n = min(pred.shape[0], tgt.shape[0])
                if n == 0:
                    return 0.0
                l1 = np.mean(np.abs(pred[:n] - tgt[:n]))
            else:
                # Fallback: compute mel for audio_state.raw_audio and compare
                raw = getattr(audio_state, "raw_audio", None)
                sr = getattr(audio_state, "sample_rate", 22050)
                if raw is None:
                    return 0.0
                mel = utils_compute_mel(raw, sr=sr, n_mels=self.audio_config.n_mels,
                                       n_fft=self.audio_config.n_fft, hop_length=self.audio_config.hop_length)
                mel = np.array(mel)
                tgt = np.array(target)
                # Reduce to comparable summaries
                mel_mean = mel.mean()
                tgt_mean = tgt.mean()
                l1 = abs(mel_mean - tgt_mean)

            # Convert to a reward: smaller L1 -> higher reward
            # Use a soft scaling to keep reward small
            reward = max(-1.0, min(1.0, -l1 / (l1 + 1e-6)))
            return float(reward * getattr(cfg, 'reconstruction_weight', 0.1))
        except Exception:
            return 0.0

    def compute_edit_efficiency_penalty(
        self, n_actions_taken: int, total_duration: float
    ) -> float:
        """Compute penalty for inefficient editing (too many actions).

        Args:
            n_actions_taken: Number of edit actions taken
            total_duration: Total track duration in seconds

        Returns:
            Efficiency penalty (0 to -1)
        """
        # Heuristic: expect ~1 action per 10 seconds of audio
        expected_actions = total_duration / 10.0
        action_ratio = n_actions_taken / (expected_actions + 1e-6)

        # Penalty for excessive actions
        if action_ratio > 2.0:
            return -0.5 * (action_ratio - 2.0)
        return 0.0

    def compute_edit_completeness_bonus(
        self, n_beats_edited: int, total_beats: int
    ) -> float:
        """Compute bonus for editing sufficient portion of track.

        Args:
            n_beats_edited: Number of beats edited
            total_beats: Total number of beats

        Returns:
            Completeness bonus (0 to +0.5)
        """
        target_ratio = self.reward_config.target_keep_ratio
        actual_ratio = n_beats_edited / (total_beats + 1e-6)

        if actual_ratio >= target_ratio:
            return 0.5
        elif actual_ratio > 0.2 * target_ratio:
            return 0.25 * (actual_ratio / target_ratio)
        else:
            return 0.0


def compute_trajectory_return(
    rewards: list, gamma: float = 0.99, normalize: bool = True
) -> Tuple[list, float]:
    """Compute discounted cumulative returns from reward trajectory.

    Args:
        rewards: List of rewards
        gamma: Discount factor
        normalize: Normalize returns to zero mean, unit variance

    Returns:
        Tuple of (returns, mean_return)
    """
    returns = []
    cumulative_return = 0.0

    for reward in reversed(rewards):
        cumulative_return = reward + gamma * cumulative_return
        returns.insert(0, cumulative_return)

    returns = np.array(returns, dtype=np.float32)

    if normalize and len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return > 0:
            returns = (returns - mean_return) / std_return
        else:
            returns = returns - mean_return
    else:
        mean_return = returns[0] if len(returns) > 0 else 0.0

    return list(returns), float(mean_return)
