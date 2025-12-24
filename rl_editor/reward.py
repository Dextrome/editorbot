"""Reward signals for RL-based audio editor.

Implements three complementary signals:
1. Sparse rewards - Human feedback
2. Dense rewards - Automatic metrics
3. Learned rewards - Human preference model
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import librosa
import logging

from .config import Config, RewardConfig
from .utils import estimate_tempo, get_energy_contour
from .utils import compute_mel_spectrogram as utils_compute_mel

logger = logging.getLogger(__name__)


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
        # LRU cache for per-audio computed per-beat mel targets
        # Keyed by pair_id when available, else by (raw_len, n_beats)
        self._cache: OrderedDict = OrderedDict()
        self._cache_max_size = 500  # Limit cache to prevent OOM

    def _cache_put(self, key, value):
        """Add item to cache with LRU eviction."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        # Evict oldest entries if over limit
        while len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)

    def get_target_per_beat(self, audio_state) -> Optional[np.ndarray]:
        """Return per-beat mel targets for an audio_state, using cache.

        Tries in order: audio_state.target_mel, audio_state.mel_spectrogram,
        and finally computes mel from raw audio. Returns array shape (n_beats, n_mels)
        or None if unavailable.
        """
        n_beats = len(getattr(audio_state, 'beat_times', []))
        if n_beats <= 0:
            return None

        pair_id = getattr(audio_state, 'pair_id', None)
        raw_len = 0 if getattr(audio_state, 'raw_audio', None) is None else len(audio_state.raw_audio)
        cache_key = ("pair", pair_id, n_beats) if pair_id else ("raw", raw_len, n_beats)
        if cache_key in self._cache:
            # Move to end for LRU
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        def _aggregate_spec_to_beats(spec: np.ndarray, n_beats: int) -> Optional[np.ndarray]:
            if spec is None or spec.ndim != 2:
                return None
            n_mels, n_frames = spec.shape
            if n_frames < n_beats:
                return None
            frames_per_beat = n_frames // n_beats
            if frames_per_beat <= 0:
                return None
            trim = frames_per_beat * n_beats
            spec_trim = spec[:, :trim]
            spec_rs = spec_trim.reshape(n_mels, n_beats, frames_per_beat)
            perbeat = spec_rs.mean(axis=2).T
            return perbeat

        # 1) target_mel
        try:
            tm = getattr(audio_state, 'target_mel', None)
            if tm is not None:
                tm = np.array(tm)
                if tm.ndim == 2 and tm.shape[0] == n_beats:
                    self._cache_put(cache_key, tm)
                    return tm
                if tm.ndim == 2 and tm.shape[1] == n_beats:
                    out = tm.T
                    self._cache_put(cache_key, out)
                    return out
                if tm.ndim == 2 and tm.shape[1] >= n_beats:
                    out = _aggregate_spec_to_beats(tm, n_beats)
                    if out is not None:
                        self._cache_put(cache_key, out)
                        return out
        except Exception:
            pass

        # 2) mel_spectrogram
        try:
            mel_spec = getattr(audio_state, 'mel_spectrogram', None)
            if mel_spec is not None:
                mel_spec = np.array(mel_spec)
                out = _aggregate_spec_to_beats(mel_spec, n_beats)
                if out is not None:
                    self._cache_put(cache_key, out)
                    return out
        except Exception:
            pass

        # 3) raw audio compute mel
        try:
            raw = getattr(audio_state, 'raw_audio', None)
            sr = getattr(audio_state, 'sample_rate', 22050)
            if raw is not None:
                mel_spec = utils_compute_mel(raw, sr=sr, n_mels=self.audio_config.n_mels,
                                            n_fft=self.audio_config.n_fft, hop_length=self.audio_config.hop_length)
                mel_spec = np.array(mel_spec)
                out = _aggregate_spec_to_beats(mel_spec, n_beats)
                if out is not None:
                    self._cache_put(cache_key, out)
                    return out
        except Exception:
            pass

        return None

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
            # Diagnostic: log what fields are present on audio_state
            # Quiet debug: record shapes at debug level only
            try:
                target_shape = None
                if getattr(audio_state, 'target_mel', None) is not None:
                    target_shape = np.array(audio_state.target_mel).shape
                mel_spec_shape = None
                if getattr(audio_state, 'mel_spectrogram', None) is not None:
                    mel_spec_shape = np.array(audio_state.mel_spectrogram).shape
                raw_len = 0 if getattr(audio_state, 'raw_audio', None) is None else len(audio_state.raw_audio)
                edited_shape = None
                if edited_mel is not None:
                    edited_shape = np.array(edited_mel).shape
                logger.debug(
                    "compute_reconstruction_reward: audio_state fields: target_mel=%s mel_spectrogram=%s raw_audio_len=%s edited_mel=%s",
                    str(target_shape), str(mel_spec_shape), int(raw_len), str(edited_shape),
                )
            except Exception:
                logger.debug("compute_reconstruction_reward: failed to stringify audio_state fields")
            # Prefer passed edited_mel; ensure we can build a per-beat target when requested
            if edited_mel is None and getattr(audio_state, 'target_mel', None) is None and getattr(audio_state, 'mel_spectrogram', None) is None and getattr(audio_state, 'raw_audio', None) is None:
                logger.info("compute_reconstruction_reward: no edited or target mel/raw audio available, returning 0")
                return 0.0

            # Per-beat handling: build `target_per_beat` from available audio_state fields
            if per_beat:
                target_per_beat = None
                n_beats = len(getattr(audio_state, 'beat_times', []))

                # Determine cache key
                pair_id = getattr(audio_state, 'pair_id', None)
                raw_len = 0 if getattr(audio_state, 'raw_audio', None) is None else len(audio_state.raw_audio)
                cache_key = None
                if pair_id:
                    cache_key = ("pair", pair_id, n_beats)
                else:
                    cache_key = ("raw", raw_len, n_beats)

                # Try cache (move to end for LRU)
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    target_per_beat = self._cache[cache_key]

                # Helper to aggregate full spectrogram to per-beat in a vectorized way
                def _aggregate_spec_to_beats(spec: np.ndarray, n_beats: int) -> Optional[np.ndarray]:
                    # spec expected shape (n_mels, n_frames)
                    if spec.ndim != 2 or n_beats <= 0 or spec.shape[1] < 1:
                        return None
                    n_mels, n_frames = spec.shape
                    # frames per beat (floor); if fewer frames than beats, return None
                    if n_frames < n_beats:
                        return None
                    frames_per_beat = n_frames // n_beats
                    if frames_per_beat <= 0:
                        return None
                    # Trim to an exact multiple
                    trim = frames_per_beat * n_beats
                    spec_trim = spec[:, :trim]
                    # reshape -> (n_mels, n_beats, frames_per_beat)
                    spec_rs = spec_trim.reshape(n_mels, n_beats, frames_per_beat)
                    # mean over frames_per_beat -> (n_mels, n_beats), transpose -> (n_beats, n_mels)
                    perbeat = spec_rs.mean(axis=2).T
                    return perbeat

                # 1) If audio_state.target_mel looks like per-beat (n_beats rows), accept it
                if target_per_beat is None and getattr(audio_state, 'target_mel', None) is not None:
                    try:
                        tm = np.array(audio_state.target_mel)
                        if tm.ndim == 2 and tm.shape[0] == n_beats:
                            target_per_beat = tm
                        elif tm.ndim == 2 and tm.shape[1] == n_beats:
                            target_per_beat = tm.T
                        elif tm.ndim == 2 and tm.shape[1] >= n_beats:
                            target_per_beat = _aggregate_spec_to_beats(tm, n_beats)
                    except Exception:
                        target_per_beat = None

                # 2) Aggregate mel_spectrogram per-beat
                if target_per_beat is None and getattr(audio_state, 'mel_spectrogram', None) is not None:
                    try:
                        mel_spec = np.array(audio_state.mel_spectrogram)
                        target_per_beat = _aggregate_spec_to_beats(mel_spec, n_beats)
                    except Exception:
                        target_per_beat = None

                # 3) Fallback: compute mel from raw audio and aggregate per-beat (cache result)
                if target_per_beat is None:
                    raw = getattr(audio_state, 'raw_audio', None)
                    sr = getattr(audio_state, 'sample_rate', 22050)
                    if raw is not None and n_beats > 0:
                        try:
                            mel_spec = utils_compute_mel(raw, sr=sr, n_mels=self.audio_config.n_mels,
                                                        n_fft=self.audio_config.n_fft, hop_length=self.audio_config.hop_length)
                            mel_spec = np.array(mel_spec)
                            target_per_beat = _aggregate_spec_to_beats(mel_spec, n_beats)
                        except Exception:
                            target_per_beat = None

                # Cache if we have targets (with LRU eviction)
                if target_per_beat is not None and cache_key is not None:
                    try:
                        self._cache_put(cache_key, target_per_beat)
                    except Exception:
                        pass

                if target_per_beat is None:
                    logger.info("compute_reconstruction_reward: no target per-beat mel available (n_beats=%s), returning 0", n_beats)
                    return 0.0

                # Require edited_mel to compare
                if edited_mel is None:
                    logger.info("compute_reconstruction_reward: edited_mel missing for per-beat comparison, returning 0")
                    return 0.0

                edited = np.array(edited_mel)

                # Align lengths and mel-dim
                n = min(target_per_beat.shape[0], edited.shape[0])
                if n == 0:
                    return 0.0

                # If mel-dim mismatch, trim/pad target to match edited
                if target_per_beat.shape[1] != edited.shape[1]:
                    if target_per_beat.shape[1] > edited.shape[1]:
                        target_per_beat = target_per_beat[:, : edited.shape[1]]
                    else:
                        pad_width = edited.shape[1] - target_per_beat.shape[1]
                        if pad_width > 0:
                            target_per_beat = np.pad(target_per_beat, ((0,0),(0,pad_width)), mode='constant')

                l1 = np.mean(np.abs(target_per_beat[:n] - edited[:n]))

                score = 1.0 / (1.0 + float(l1))
                # Map score in (0,1] to [-1, 1] then scale to configured max reward
                max_r = float(getattr(cfg, 'reconstruction_max_reward', 10.0))
                weight = float(getattr(cfg, 'reconstruction_weight', 1.0))
                norm = float(score * 2.0 - 1.0)
                reward = float(norm * max_r * weight)
                # Store diagnostics on the calculator instance for callers to inspect
                try:
                    self._last_l1 = float(l1)
                    self._last_score = float(score)
                    self._last_reward = float(reward)
                except Exception:
                    self._last_l1 = None
                    self._last_score = None
                    self._last_reward = float(reward)
                #logger.info(
                #    "compute_reconstruction_reward: used_target='per_beat' n=%d l1=%.6f score=%.6f norm=%.6f reward=%.6f (weight=%.6f max=%.6f)",
                #    n, float(l1), score, norm, reward, weight, max_r,
                #)
                return reward
        except Exception:
            logger.exception("compute_reconstruction_reward failed")
            try:
                self._last_l1 = None
                self._last_score = None
                self._last_reward = 0.0
            except Exception:
                pass
            return 0.0


def compute_trajectory_return(
    rewards: list,
    gamma: float = 0.99,
    normalize: bool = True,
    clip_min: float = None,
    clip_max: float = None,
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

    # Clip returns to a reasonable range to avoid extreme value targets
    try:
        # Use provided clip values if given, otherwise fall back to large bounds
        low = -1000.0 if clip_min is None else float(clip_min)
        high = 1000.0 if clip_max is None else float(clip_max)
        returns = np.clip(returns, low, high)
    except Exception:
        returns = returns

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
