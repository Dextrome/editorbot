"""Audio analysis functionality for feature extraction."""

from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn



class AudioAnalyzer:
    """Analyzes audio to extract features for AI processing."""

    def detect_key_short(self, audio_data: np.ndarray) -> str:
        """
        Detect the musical key of a short audio slice (e.g., 1-2 seconds).
        Args:
            audio_data: Audio data to analyze (short segment).
        Returns:
            Detected key as a string (e.g., "C major", "A minor").
        """
        if len(audio_data) < self.sample_rate // 2:
            return "unknown"
        # Use a smaller n_fft for short signals to avoid warnings and improve accuracy
        n_fft = 256 if len(audio_data) < 1024 else 1024
        try:
            chroma = librosa.feature.chroma_cqt(y=audio_data, sr=self.sample_rate, n_chroma=12, n_fft=n_fft)
        except Exception:
            # Fallback to default if CQT fails
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate, n_chroma=12, n_fft=n_fft)
        # Defensive checks: ensure chroma has expected shape and valid values
        chroma_avg = np.mean(chroma, axis=1) if hasattr(chroma, 'shape') and chroma.size > 0 else np.array([])
        if chroma_avg.size != 12 or np.all(np.isnan(chroma_avg)):
            return "unknown"

        # Normalize and guard against degenerate values
        chroma_avg = np.nan_to_num(chroma_avg, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(chroma_avg) <= 0:
            return "unknown"

        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_index = int(np.argmax(chroma_avg))
        if key_index < 0 or key_index >= len(keys):
            return "unknown"

        major_profile = chroma_avg[key_index]
        minor_index = (key_index + 9) % 12  # Relative minor
        minor_profile = chroma_avg[minor_index]
        if major_profile >= minor_profile:
            return f"{keys[key_index]} major"
        return f"{keys[minor_index]} minor"

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio analyzer.

        Args:
            sample_rate: Sample rate of the audio to analyze.
        """
        self.sample_rate = sample_rate
        # Persistent encoder (projection + NATTEN) for framing -> contextual features
        # Lazily created by `extract_attention_features` or set via `set_encoder`.
        self.encoder: Optional[torch.nn.Module] = None
        self.encoder_frame_size: Optional[int] = None
        self.encoder_proj_dim: int = 8

    def detect_tempo(self, audio_data: np.ndarray, use_natten: bool = False) -> float:
        """
        Detect the tempo (BPM) of the audio.

        This function optionally uses NATTEN-derived contextual features when
        `use_natten=True`. NATTEN provides contextualized frame embeddings which
        can improve onset/tempo estimates when the raw waveform is noisy.

        Args:
            audio_data: Audio data to analyze.
            use_natten: If True, use `extract_attention_features` to compute features
                        and compute tempo from those features. Defaults to False.

        Returns:
            Estimated tempo in BPM.
        """
        def _tempo_from_onset_env(onset_env):
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
            return float(tempo)

        if not use_natten:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            return float(tempo)

        # Use NATTEN features to compute an onset envelope
        feats = self.extract_attention_features(audio_data)
        # Compute a simple onset envelope from feature deltas
        feat_env = np.mean(np.abs(np.diff(feats, axis=0)), axis=1)
        feat_env = feat_env / (np.max(feat_env) + 1e-9)
        return _tempo_from_onset_env(feat_env)

    def analyze_with_allin1(self, audio_path: Optional[str] = None, audio_data: Optional[np.ndarray] = None, sample_rate: Optional[int] = None) -> dict:
        """
        Analyze audio using the external `allin1` music-structure analyzer if available.

        This is an optional integration. If `allin1` is not installed, this method
        will raise a RuntimeError with installation instructions.

        Args:
            audio_path: Path to an audio file. Either `audio_path` or `audio_data` must be provided.
            audio_data: Raw audio samples as a numpy array.
            sample_rate: Sample rate for the audio data (if providing `audio_data`).

        Returns:
            A dict containing keys such as `tempo`, `beats`, `downbeats`, `segments`, `labels`, and optionally `separation`.
        """
        # Lazy import to avoid a hard dependency
        try:
            import allin1
        except Exception as e:
            raise RuntimeError(
                "allin1 package is not available. Install it with `pip install allin1` "
                "or follow the project's installation instructions. Original error: " + str(e)
            )

        # Prepare inputs
        if audio_path is None and audio_data is None:
            raise ValueError("Either audio_path or audio_data must be provided")

        # Try multiple common API entry points used by similar analyzers
        result = None
        # 1) allin1.analyze_file(path)
        if hasattr(allin1, "analyze_file") and audio_path is not None:
            result = allin1.analyze_file(audio_path)
        # 2) allin1.analyze(path) or allin1.analyze_audio
        elif hasattr(allin1, "analyze") and audio_path is not None:
            result = allin1.analyze(audio_path)
        elif hasattr(allin1, "analyze_audio") and audio_data is not None:
            result = allin1.analyze_audio(audio_data, sr=sample_rate or self.sample_rate)
        # 3) model-based API: allin1.load_model() -> model.predict / model.analyze
        elif hasattr(allin1, "load_model"):
            model = allin1.load_model()
            if audio_path is not None and hasattr(model, "analyze_file"):
                result = model.analyze_file(audio_path)
            elif audio_data is not None and hasattr(model, "analyze_audio"):
                result = model.analyze_audio(audio_data, sr=sample_rate or self.sample_rate)
            elif audio_path is not None and hasattr(model, "analyze"):
                result = model.analyze(audio_path)

        if result is None:
            raise RuntimeError("Found `allin1` package but could not find a supported API entry point. "
                               "Please check your allin1 installation and APIs.")

        # Normalize result into expected dict keys (best-effort mapping)
        out = {}
        # Many analyzers return JSON-like dicts; try to extract common fields
        if isinstance(result, dict):
            out.update(result)
        else:
            # If result is an object with attributes, map common names
            for k in ("tempo", "beats", "downbeats", "segments", "labels", "separation"):
                if hasattr(result, k):
                    out[k] = getattr(result, k)

        return out

    def detect_beats(self, audio_data: np.ndarray, use_natten: bool = False) -> np.ndarray:
        """
        Detect beat positions in the audio.

        Args:
            audio_data: Audio data to analyze.
            use_natten: If True, compute beats from NATTEN-derived features.

        Returns:
            Array of beat positions in seconds.
        """
        if not use_natten:
            _, beat_frames = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            return librosa.frames_to_time(beat_frames, sr=self.sample_rate)

        # Compute NATTEN features and derive an onset envelope
        feats = self.extract_attention_features(audio_data)
        onset_env = np.mean(np.abs(np.diff(feats, axis=0)), axis=1)
        # Use librosa beat on the feature-derived onset env (frame-based)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        # Convert beat frames (from onset_env frames) into times using hop of 256 samples
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate, hop_length=256)
        return beat_times

    def extract_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract audio features for ML processing.

        Args:
            audio_data: Audio data to analyze.

        Returns:
            Dictionary of extracted features.
        """
        # Use adaptive hop_length for long files
        target_frames = 5000
        hop_length = max(512, int(len(audio_data) / target_frames))
        
        features = {}

        # Mel-frequency cepstral coefficients
        features["mfcc"] = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length)

        # Spectral features
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate, hop_length=hop_length
        )
        features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            y=audio_data, sr=self.sample_rate, hop_length=hop_length
        )

        # Chroma features (pitch class)
        features["chroma"] = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate, hop_length=hop_length)

        # Zero crossing rate
        features["zcr"] = librosa.feature.zero_crossing_rate(audio_data, hop_length=hop_length)

        # RMS energy
        features["rms"] = librosa.feature.rms(y=audio_data, hop_length=hop_length)

        return features

    def extract_attention_features(
        self,
        audio_data: Optional[np.ndarray] = None,
        frame_size: int = 512,
        hop_size: int = 256,
        proj_dim: int = 8,
        kernel_size: int = 7,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Extract contextualized frame features using NATTEN 1D attention.

        Pipeline:
        - Frame the audio into overlapping frames (shape: n_frames x frame_size)
        - Project each frame to a low-dimensional vector
        - Run NATTEN 1D self-attention across frames to get contextualized features

        Args:
            audio_data: Optional numpy array; uses stored audio if None.
            frame_size: Frame length in samples.
            hop_size: Hop length in samples.
            proj_dim: Output projection dimension (must be supported by NATTEN head sizes, e.g., 8,16,32...).
            kernel_size: Neighborhood kernel size for NATTEN.
            device: Torch device; defaults to CUDA if available.

        Returns:
            Numpy array of shape (n_frames, proj_dim) containing contextualized features.
        """
        data = audio_data if audio_data is not None else getattr(self, "audio_data", None)
        if data is None:
            raise ValueError("No audio data provided for attention feature extraction")

        # Frame the audio: shape (frame_size, n_frames)
        frames = librosa.util.frame(data, frame_length=frame_size, hop_length=hop_size).T

        # Convert frames to torch tensor: (1, n_frames, frame_size)
        tensor = torch.from_numpy(frames.copy()).float().unsqueeze(0)

        # Choose device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tensor = tensor.to(device)

        # Use or initialize a persistent encoder module (projection + NATTEN)
        self.encoder_proj_dim = proj_dim or self.encoder_proj_dim
        if self.encoder is None or self.encoder_frame_size != frame_size or getattr(self.encoder, "proj", None) is None or getattr(self.encoder, "proj").in_features != frame_size or getattr(self.encoder, "proj").out_features != self.encoder_proj_dim:
            # Lazily create encoder
            from src.ai.natten_encoder import NattenFrameEncoder

            self.encoder = NattenFrameEncoder(frame_size=frame_size, proj_dim=self.encoder_proj_dim, kernel_size=kernel_size, num_heads=1)
            self.encoder_frame_size = frame_size

        # Move encoder to device
        self.encoder = self.encoder.to(device)

        self.encoder.eval()
        with torch.no_grad():
            out_tensor = self.encoder(tensor)  # (1, n_frames, proj_dim)

        # Return (n_frames, proj_dim) on CPU as numpy
        return out_tensor.squeeze(0).cpu().numpy()

    def save_encoder(self, path: str | Path) -> None:
        """
        Save the persisted encoder (projection + NATTEN) to disk.

        The saved file contains a dict with keys: `state_dict`, `frame_size`, `proj_dim`.

        Args:
            path: File path to save the encoder state.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder available to save. Create it by calling extract_attention_features first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.encoder.state_dict(),
            "frame_size": self.encoder_frame_size,
            "proj_dim": self.encoder_proj_dim,
        }

        # Save on CPU for portability
        torch.save(payload, str(path))

    def load_encoder(self, path: str | Path, map_location: Optional[str | torch.device] = "cpu") -> None:
        """
        Load an encoder saved by `save_encoder` and set it as this analyzer's encoder.

        Args:
            path: Path to the saved encoder file.
            map_location: Device mapping for loading (e.g., 'cpu' or torch.device('cuda:0')).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Encoder file not found: {path}")

        payload = torch.load(str(path), map_location=map_location)
        frame_size = int(payload.get("frame_size"))
        proj_dim = int(payload.get("proj_dim"))

        # Lazily import encoder class to avoid top-level import cycles
        from src.ai.natten_encoder import NattenFrameEncoder

        encoder = NattenFrameEncoder(frame_size=frame_size, proj_dim=proj_dim, kernel_size=7, num_heads=1)
        encoder.load_state_dict(payload["state_dict"])

        # Set as persisted encoder
        device = torch.device(map_location if map_location is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = encoder.to(device)
        self.encoder_frame_size = frame_size
        self.encoder_proj_dim = proj_dim

    def set_encoder(self, encoder: torch.nn.Module, frame_size: int, proj_dim: int, device: Optional[torch.device] = None) -> None:
        """
        Register an external encoder (e.g., `NattenFrameEncoder`) for use by this analyzer.

        Args:
            encoder: A PyTorch module that accepts input shape (1, n_frames, frame_size)
            frame_size: Frame length (in samples) that the encoder expects.
            proj_dim: Output projection dimension of the encoder.
            device: Optional device to move the encoder to (defaults to CUDA if available).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder.to(device)
        self.encoder_frame_size = int(frame_size)
        self.encoder_proj_dim = int(proj_dim)

    def detect_sections(
        self, audio_data: np.ndarray, num_segments: int = 10, use_natten: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect structural sections in the audio (intro, verse, chorus, etc.).

        Args:
            audio_data: Audio data to analyze.
            num_segments: Number of segments to detect.

        Returns:
            Tuple of (segment_boundaries, segment_labels).
        """
        # Use adaptive hop_length for long files
        target_frames = 5000
        hop_length = max(512, int(len(audio_data) / target_frames))

        if not use_natten:
            # Compute self-similarity matrix using MFCCs
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, hop_length=hop_length)
            # Use structural segmentation
            bounds = librosa.segment.agglomerative(mfcc, num_segments)
            bound_times = librosa.frames_to_time(bounds, sr=self.sample_rate, hop_length=hop_length)
            labels = np.arange(len(bounds))
            return bound_times, labels

        # Use NATTEN features to create a similarity matrix for segmentation
        feats = self.extract_attention_features(audio_data, frame_size=256, hop_size=128)
        # Try to use librosa's segmentation using our features; fallback to MFCCs on error
        try:
            bounds = librosa.segment.agglomerative(feats.T, num_segments)
            bound_times = librosa.frames_to_time(bounds, sr=self.sample_rate, hop_length=128)
            labels = np.arange(len(bounds))
        except Exception:
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, hop_length=hop_length)
            bounds = librosa.segment.agglomerative(mfcc, num_segments)
            bound_times = librosa.frames_to_time(bounds, sr=self.sample_rate, hop_length=hop_length)
            labels = np.arange(len(bounds))

        return bound_times, labels

    def get_loudness(self, audio_data: np.ndarray) -> float:
        """
        Calculate the perceived loudness of the audio.

        Args:
            audio_data: Audio data to analyze.

        Returns:
            Loudness value in dB.
        """
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            return 20 * np.log10(rms)
        return -np.inf

    def detect_key(self, audio_data: np.ndarray) -> str:
        """
        Detect the musical key of the audio.

        Args:
            audio_data: Audio data to analyze.

        Returns:
            Detected key as a string (e.g., "C major", "A minor").
        """
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=self.sample_rate)
        chroma_avg = np.mean(chroma, axis=1)

        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_index = int(np.argmax(chroma_avg))

        # Simple major/minor detection based on relative minor relationship
        major_profile = chroma_avg[key_index]
        minor_index = (key_index + 9) % 12  # Relative minor
        minor_profile = chroma_avg[minor_index]

        if major_profile >= minor_profile:
            return f"{keys[key_index]} major"
        return f"{keys[minor_index]} minor"

    def detect_silence_regions(
        self, audio_data: np.ndarray, threshold_db: float = -40.0, min_duration: float = 0.5
    ) -> list[tuple[float, float]]:
        """
        Detect regions of silence or very low audio.

        Args:
            audio_data: Audio data to analyze.
            threshold_db: Threshold in dB below which is considered silence.
            min_duration: Minimum duration in seconds to count as silence region.

        Returns:
            List of (start_time, end_time) tuples for silence regions.
        """
        # Convert to frame-based RMS
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find frames below threshold
        is_silent = rms_db < threshold_db
        
        # Convert to time regions
        silence_regions = []
        in_silence = False
        start_frame = 0
        
        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                start_frame = i
                in_silence = True
            elif not silent and in_silence:
                end_frame = i
                start_time = librosa.frames_to_time(start_frame, sr=self.sample_rate, hop_length=hop_length)
                end_time = librosa.frames_to_time(end_frame, sr=self.sample_rate, hop_length=hop_length)
                if end_time - start_time >= min_duration:
                    silence_regions.append((start_time, end_time))
                in_silence = False
        
        # Handle silence at end
        if in_silence:
            end_time = librosa.frames_to_time(len(is_silent), sr=self.sample_rate, hop_length=hop_length)
            start_time = librosa.frames_to_time(start_frame, sr=self.sample_rate, hop_length=hop_length)
            if end_time - start_time >= min_duration:
                silence_regions.append((start_time, end_time))
        
        return silence_regions

    def detect_anomalies(
        self, audio_data: np.ndarray, sensitivity: float = 2.0
    ) -> list[tuple[float, float, str]]:
        """
        Detect anomalous regions (clipping, sudden volume spikes, potential mistakes).

        Args:
            audio_data: Audio data to analyze.
            sensitivity: How sensitive to anomalies (lower = more sensitive).

        Returns:
            List of (start_time, end_time, anomaly_type) tuples.
        """
        anomalies = []
        frame_length = int(0.05 * self.sample_rate)  # 50ms frames
        hop_length = int(0.025 * self.sample_rate)   # 25ms hop
        
        # Detect clipping
        clipping_threshold = 0.99
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            if np.any(np.abs(frame) > clipping_threshold):
                start_time = i / self.sample_rate
                end_time = (i + frame_length) / self.sample_rate
                anomalies.append((start_time, end_time, "clipping"))
        
        # Detect sudden volume changes (potential mistakes/false starts)
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        rms_diff = np.diff(rms)
        rms_std = np.std(rms_diff)
        
        for i, diff in enumerate(rms_diff):
            if abs(diff) > sensitivity * rms_std:
                start_time = librosa.frames_to_time(i, sr=self.sample_rate, hop_length=hop_length)
                end_time = librosa.frames_to_time(i + 2, sr=self.sample_rate, hop_length=hop_length)
                anomaly_type = "sudden_increase" if diff > 0 else "sudden_drop"
                anomalies.append((start_time, end_time, anomaly_type))
        
        return anomalies

    def detect_repeated_sections(
        self, audio_data: np.ndarray, min_similarity: float = 0.85
    ) -> list[tuple[float, float, float, float]]:
        """
        Detect repeated/similar sections (potential false starts or retakes).

        Args:
            audio_data: Audio data to analyze.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of (section1_start, section1_end, section2_start, section2_end) for similar sections.
        """
        # Use larger hop length for long files to avoid memory explosion
        # Target max ~5000 frames to keep matrix under ~200MB
        duration = len(audio_data) / self.sample_rate
        target_frames = 5000
        hop_length = max(512, int(len(audio_data) / target_frames))
        
        # Compute features for comparison
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length)
        
        # Compute self-similarity matrix
        sim_matrix = librosa.segment.recurrence_matrix(
            mfcc, mode='affinity', sym=True
        )
        
        # Find repeated sections using diagonal matching
        repeated = []
        window_frames = int(2.0 * self.sample_rate / hop_length)  # ~2 second windows
        
        for i in range(0, sim_matrix.shape[0] - window_frames, window_frames // 2):
            for j in range(i + window_frames, sim_matrix.shape[1] - window_frames, window_frames // 2):
                # Check diagonal similarity
                diag_sim = np.mean([sim_matrix[i + k, j + k] for k in range(window_frames)])
                if diag_sim > min_similarity:
                    t1_start = librosa.frames_to_time(i, sr=self.sample_rate, hop_length=hop_length)
                    t1_end = librosa.frames_to_time(i + window_frames, sr=self.sample_rate, hop_length=hop_length)
                    t2_start = librosa.frames_to_time(j, sr=self.sample_rate, hop_length=hop_length)
                    t2_end = librosa.frames_to_time(j + window_frames, sr=self.sample_rate, hop_length=hop_length)
                    repeated.append((t1_start, t1_end, t2_start, t2_end))
        
        return repeated

    def find_best_take(
        self, audio_data: np.ndarray, sections: list[tuple[float, float]]
    ) -> int:
        """
        Among similar sections (potential retakes), find the best quality one.

        Args:
            audio_data: Audio data.
            sections: List of (start_time, end_time) for similar sections.

        Returns:
            Index of the best section.
        """
        scores = []
        
        for start, end in sections:
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            section = audio_data[start_sample:end_sample]
            
            # Score based on: consistent loudness, no clipping, spectral balance
            rms = librosa.feature.rms(y=section)[0]
            rms_consistency = 1.0 / (np.std(rms) + 0.001)  # More consistent = better
            
            # Penalize clipping
            clipping_penalty = np.sum(np.abs(section) > 0.99) / len(section)
            
            # Spectral flatness (avoid overly harsh sections)
            flatness = np.mean(librosa.feature.spectral_flatness(y=section))
            
            score = rms_consistency * (1 - clipping_penalty) * (1 - flatness * 0.5)
            scores.append(score)
        
        return int(np.argmax(scores))
