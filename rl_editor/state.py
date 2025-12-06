"""State representation for RL-based audio editor.

Handles observation space construction from audio features and edit history.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

from .config import Config, StateConfig
from .utils import compute_mel_spectrogram, detect_beats, estimate_tempo


@dataclass
class AudioState:
    """Raw audio state information."""

    beat_index: int
    beat_times: np.ndarray  # Shape: (n_beats,)
    beat_features: np.ndarray  # Shape: (n_beats, feature_dim)
    mel_spectrogram: Optional[np.ndarray] = None  # Shape: (n_mels, n_frames)
    stem_features: Optional[Dict[str, np.ndarray]] = None  # {stem: (n_beats, feature_dim)}
    global_features: Optional[np.ndarray] = None  # Shape: (global_feature_dim,)
    tempo: Optional[float] = None
    energy_contour: Optional[np.ndarray] = None  # Shape: (n_frames,)
    target_labels: Optional[np.ndarray] = None  # Shape: (n_beats,) - Ground truth KEEP=1/CUT=0 from human edits
    raw_audio: Optional[np.ndarray] = None  # Shape: (n_samples,) - Raw audio for trajectory reward
    sample_rate: int = 22050  # Audio sample rate


@dataclass
class EditHistory:
    """Edit history tracking."""

    kept_beats: list = None  # Indices of kept beats
    cut_beats: list = None  # Indices of cut beats
    looped_beats: dict = None  # {beat_index: loop_times} for looped beats
    crossfaded_pairs: list = None  # Pairs of crossfaded beats
    reordered_pairs: list = None  # Pairs of reordered sections
    total_duration_edited: float = 0.0  # Total duration of edits applied

    def __post_init__(self):
        """Initialize list fields."""
        if self.kept_beats is None:
            self.kept_beats = []
        if self.cut_beats is None:
            self.cut_beats = []
        if self.looped_beats is None:
            self.looped_beats = {}
        if self.crossfaded_pairs is None:
            self.crossfaded_pairs = []
        if self.reordered_pairs is None:
            self.reordered_pairs = []

    def get_edited_beats(self) -> list:
        """Get all edited beat indices."""
        return sorted(
            list(
                set(
                    self.kept_beats
                    + self.cut_beats
                    + list(self.looped_beats.keys())
                    + [b for pair in self.crossfaded_pairs for b in pair]
                    + [b for pair in self.reordered_pairs for b in pair]
                )
            )
        )

    def add_keep(self, beat_index: int) -> None:
        """Add kept beat."""
        if beat_index not in self.kept_beats:
            self.kept_beats.append(beat_index)

    def add_cut(self, beat_index: int) -> None:
        """Add cut beat."""
        if beat_index not in self.cut_beats:
            self.cut_beats.append(beat_index)

    def add_loop(self, beat_index: int, loop_times: int = 2) -> None:
        """Add looped beat with loop count.
        
        Args:
            beat_index: Beat to loop
            loop_times: Number of times to play beat (2 = 2x, 3 = 3x, etc.)
        """
        self.looped_beats[beat_index] = loop_times
        # Also add to kept_beats since looped beats are kept
        if beat_index not in self.kept_beats:
            self.kept_beats.append(beat_index)

    def add_crossfade(self, beat_i: int, beat_j: int) -> None:
        """Add crossfaded pair."""
        pair = tuple(sorted([beat_i, beat_j]))
        if pair not in self.crossfaded_pairs:
            self.crossfaded_pairs.append(pair)

    def add_reorder(self, section_a: int, section_b: int) -> None:
        """Add reordered pair."""
        pair = tuple(sorted([section_a, section_b]))
        if pair not in self.reordered_pairs:
            self.reordered_pairs.append(pair)


class StateRepresentation:
    """Constructs observation state from audio features and edit history."""

    def __init__(self, config: Config, beat_feature_dim: int = None) -> None:
        """Initialize state representation.

        Args:
            config: Configuration object
            beat_feature_dim: Dimension of beat features (auto-detected if None)
        """
        self.config = config
        self.state_config: StateConfig = config.state
        self.audio_config = config.audio
        self._feature_dim: Optional[int] = None
        
        # Determine beat feature dimension from config
        if beat_feature_dim is not None:
            self.beat_feature_dim = beat_feature_dim
        else:
            # Auto-detect from feature mode
            feature_mode = getattr(config.features, 'feature_mode', 'basic') if hasattr(config, 'features') else 'basic'
            if feature_mode == "basic":
                self.beat_feature_dim = 4
            elif feature_mode == "enhanced":
                self.beat_feature_dim = 109
            elif feature_mode == "full":
                self.beat_feature_dim = 121
            else:
                self.beat_feature_dim = 4  # fallback

    @property
    def feature_dim(self) -> int:
        """Get total feature dimension."""
        if self._feature_dim is None:
            self._compute_feature_dim()
        return self._feature_dim
    
    def set_beat_feature_dim(self, dim: int) -> None:
        """Set beat feature dimension and recompute total dimension.
        
        Call this after loading data to ensure dimension matches actual features.
        """
        self.beat_feature_dim = dim
        self._feature_dim = None  # Force recomputation

    def _compute_feature_dim(self) -> None:
        """Compute total feature dimension."""
        dim = 0

        # Beat context features
        beat_context_size = self.state_config.beat_context_size
        beats_in_context = 2 * beat_context_size + 1  # current + before + after

        if self.state_config.use_beat_descriptors:
            # Beat descriptors: use actual beat feature dimension
            dim += beats_in_context * self.beat_feature_dim
        else:
            # Even if not used, provide zeros with correct dimension
            dim += beats_in_context * self.beat_feature_dim

        if self.state_config.use_mel_spectrogram:
            # Mel-spectrogram: reduced to beat-level summary
            # Skip because it's optional and depends on provided mel_spectrogram
            pass

        if self.state_config.use_stem_features:
            # Stem features: 4 stems (drums, bass, vocals, other) × 3 descriptors
            # Skip because it's optional and depends on provided stem_features
            pass

        if self.state_config.use_global_features:
            # Global features: tempo, key, overall energy, etc.
            dim += 5

        # Edit history features
        dim += 6  # 6 binary flags for edit types (keep, cut, loop, reorder, auto_crossfade, has_edits)

        # Duration constraints
        dim += 2  # remaining_duration, target_keep_ratio

        # Current beat index (normalized)
        dim += 1

        self._feature_dim = dim

    def compute_beat_descriptors(
        self, y: np.ndarray, beat_frames: np.ndarray, sr: int = 22050
    ) -> np.ndarray:
        """Compute beat-level descriptors.

        Args:
            y: Audio array
            beat_frames: Beat frame indices
            sr: Sample rate

        Returns:
            Beat descriptors (n_beats, 4) with onset strength, spectral centroid, zcr, rms
        """
        n_beats = len(beat_frames)
        descriptors = np.zeros((n_beats, 4))

        # Onset strength at each beat
        onset_env = np.abs(np.diff(np.abs(np.fft.rfft(y))))
        for i, frame in enumerate(beat_frames):
            if frame < len(onset_env):
                descriptors[i, 0] = onset_env[min(frame, len(onset_env) - 1)]

        # Spectral centroid
        S = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        centroid = np.sum(freqs[:, np.newaxis] * S, axis=0) / (np.sum(S, axis=0) + 1e-10)

        for i, frame in enumerate(beat_frames):
            if frame < len(centroid):
                descriptors[i, 1] = centroid[min(frame, len(centroid) - 1)]

        # Zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(y))))
        descriptors[:, 2] = zcr

        # RMS energy
        frame_length = 2048
        hop_length = 512
        rms = np.sqrt(np.mean(y**2))
        descriptors[:, 3] = rms

        return descriptors

    def get_beat_context(
        self, beat_index: int, beat_features: np.ndarray, context_size: int = 3
    ) -> np.ndarray:
        """Extract beat context around a beat index.

        Args:
            beat_index: Current beat index
            beat_features: Beat feature array (n_beats, feature_dim)
            context_size: Number of beats before/after

        Returns:
            Context features (context_size*2+1, feature_dim) with padding
        """
        n_beats = beat_features.shape[0]
        feature_dim = beat_features.shape[1]
        context = np.zeros((2 * context_size + 1, feature_dim))

        for i in range(-context_size, context_size + 1):
            idx = beat_index + i
            if 0 <= idx < n_beats:
                context[i + context_size] = beat_features[idx]

        return context.flatten()

    def construct_observation(
        self,
        audio_state: AudioState,
        edit_history: EditHistory,
        remaining_duration: float,
        total_duration: float,
    ) -> np.ndarray:
        """Construct observation vector from audio state and edit history.

        Args:
            audio_state: Audio state information
            edit_history: Edit history
            remaining_duration: Remaining duration budget
            total_duration: Total track duration

        Returns:
            Observation vector (feature_dim,)
        """
        features = []
        context_size = self.state_config.beat_context_size

        # 1. Beat context descriptors
        if self.state_config.use_beat_descriptors:
            beat_descriptors = audio_state.beat_features
            context = self.get_beat_context(
                audio_state.beat_index, beat_descriptors, context_size
            )
            features.append(context)

        # 2. Mel-spectrogram context
        if self.state_config.use_mel_spectrogram and audio_state.mel_spectrogram is not None:
            # Reduce to beat-level by averaging over frames
            mel_spec = audio_state.mel_spectrogram
            n_frames_per_beat = mel_spec.shape[1] // len(audio_state.beat_times)
            mel_beat_features = np.zeros(
                (len(audio_state.beat_times), mel_spec.shape[0])
            )
            for i, t in enumerate(audio_state.beat_times):
                frame_start = int(i * n_frames_per_beat)
                frame_end = int((i + 1) * n_frames_per_beat)
                if frame_start < mel_spec.shape[1]:
                    mel_beat_features[i] = np.mean(
                        mel_spec[:, frame_start:frame_end], axis=1
                    )
            context = self.get_beat_context(
                audio_state.beat_index, mel_beat_features, context_size
            )
            features.append(context)

        # 3. Stem features
        if self.state_config.use_stem_features and audio_state.stem_features is not None:
            for stem_name, stem_features in audio_state.stem_features.items():
                context = self.get_beat_context(
                    audio_state.beat_index, stem_features, context_size
                )
                features.append(context)

        # 4. Global features
        if self.state_config.use_global_features and audio_state.global_features is not None:
            features.append(audio_state.global_features)
        else:
            # Placeholder global features
            global_feats = np.zeros(5)
            if audio_state.tempo is not None:
                global_feats[0] = audio_state.tempo / 200.0  # Normalize
            features.append(global_feats)

        # 5. Edit history flags
        edited_beats = set(edit_history.get_edited_beats())
        edit_features = np.array(
            [
                float(len(edit_history.kept_beats) > 0),
                float(len(edit_history.cut_beats) > 0),
                float(len(edit_history.looped_beats) > 0),
                float(len(edit_history.crossfaded_pairs) > 0),
                float(len(edit_history.reordered_pairs) > 0),
                float(audio_state.beat_index in edited_beats),
            ]
        )
        features.append(edit_features)

        # 6. Duration constraints
        target_keep_ratio = self.config.reward.target_keep_ratio
        duration_features = np.array(
            [
                remaining_duration / (total_duration + 1e-6),
                target_keep_ratio,
            ]
        )
        features.append(duration_features)

        # 7. Current beat index (normalized)
        n_beats = len(audio_state.beat_times)
        beat_index_norm = audio_state.beat_index / (n_beats + 1e-6)
        features.append(np.array([beat_index_norm]))

        # Concatenate all features
        observation = np.concatenate(features)
        assert (
            observation.shape[0] == self.feature_dim
        ), f"Observation dim mismatch: got {observation.shape[0]}, expected {self.feature_dim}"

        return observation.astype(np.float32)

    def observation_to_dict(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat observation to dictionary of components.

        Args:
            observation: Flat observation vector

        Returns:
            Dictionary with named feature groups
        """
        result = {}
        offset = 0

        if self.state_config.use_beat_descriptors:
            context_size = self.state_config.beat_context_size
            n_features = (2 * context_size + 1) * 3
            result["beat_descriptors"] = observation[offset : offset + n_features]
            offset += n_features

        if self.state_config.use_mel_spectrogram:
            context_size = self.state_config.beat_context_size
            n_features = (2 * context_size + 1) * self.audio_config.n_mels
            result["mel_spectrogram"] = observation[offset : offset + n_features]
            offset += n_features

        if self.state_config.use_stem_features:
            context_size = self.state_config.beat_context_size
            n_features = (2 * context_size + 1) * 4 * 3
            result["stem_features"] = observation[offset : offset + n_features]
            offset += n_features

        if self.state_config.use_global_features:
            result["global_features"] = observation[offset : offset + 5]
            offset += 5

        result["edit_history"] = observation[offset : offset + 6]
        offset += 6

        result["duration_constraints"] = observation[offset : offset + 2]
        offset += 2

        result["beat_index"] = observation[offset : offset + 1]
        offset += 1

        return result
