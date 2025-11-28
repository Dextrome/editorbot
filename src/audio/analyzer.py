"""Audio analysis functionality for feature extraction."""

from typing import Dict, Optional, Tuple

import numpy as np
import librosa


class AudioAnalyzer:
    """Analyzes audio to extract features for AI processing."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio analyzer.

        Args:
            sample_rate: Sample rate of the audio to analyze.
        """
        self.sample_rate = sample_rate

    def detect_tempo(self, audio_data: np.ndarray) -> float:
        """
        Detect the tempo (BPM) of the audio.

        Args:
            audio_data: Audio data to analyze.

        Returns:
            Estimated tempo in BPM.
        """
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        return float(tempo)

    def detect_beats(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Detect beat positions in the audio.

        Args:
            audio_data: Audio data to analyze.

        Returns:
            Array of beat positions in seconds.
        """
        _, beat_frames = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        return librosa.frames_to_time(beat_frames, sr=self.sample_rate)

    def extract_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract audio features for ML processing.

        Args:
            audio_data: Audio data to analyze.

        Returns:
            Dictionary of extracted features.
        """
        features = {}

        # Mel-frequency cepstral coefficients
        features["mfcc"] = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)

        # Spectral features
        features["spectral_centroid"] = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate
        )
        features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            y=audio_data, sr=self.sample_rate
        )

        # Chroma features (pitch class)
        features["chroma"] = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)

        # Zero crossing rate
        features["zcr"] = librosa.feature.zero_crossing_rate(audio_data)

        # RMS energy
        features["rms"] = librosa.feature.rms(y=audio_data)

        return features

    def detect_sections(
        self, audio_data: np.ndarray, num_segments: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect structural sections in the audio (intro, verse, chorus, etc.).

        Args:
            audio_data: Audio data to analyze.
            num_segments: Number of segments to detect.

        Returns:
            Tuple of (segment_boundaries, segment_labels).
        """
        # Compute self-similarity matrix
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate)
        
        # Use structural segmentation
        bounds = librosa.segment.agglomerative(mfcc, num_segments)
        bound_times = librosa.frames_to_time(bounds, sr=self.sample_rate)
        
        # Generate simple labels
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
        # Compute features for comparison
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        
        # Compute self-similarity matrix
        sim_matrix = librosa.segment.recurrence_matrix(
            mfcc, mode='affinity', sym=True
        )
        
        # Find repeated sections using diagonal matching
        repeated = []
        window_frames = int(2.0 * self.sample_rate / 512)  # ~2 second windows
        
        for i in range(0, sim_matrix.shape[0] - window_frames, window_frames // 2):
            for j in range(i + window_frames, sim_matrix.shape[1] - window_frames, window_frames // 2):
                # Check diagonal similarity
                diag_sim = np.mean([sim_matrix[i + k, j + k] for k in range(window_frames)])
                if diag_sim > min_similarity:
                    t1_start = librosa.frames_to_time(i, sr=self.sample_rate)
                    t1_end = librosa.frames_to_time(i + window_frames, sr=self.sample_rate)
                    t2_start = librosa.frames_to_time(j, sr=self.sample_rate)
                    t2_end = librosa.frames_to_time(j + window_frames, sr=self.sample_rate)
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
