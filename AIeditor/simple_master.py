"""
Simple mastering/normalization for alignment.
Applies consistent processing to both raw and edit files
so they have similar spectral characteristics for matching.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
from scipy import signal
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMaster:
    """
    Apply consistent mastering to audio files.
    Goal: Make raw and edit files comparable for alignment.
    """
    
    def __init__(
        self,
        sr: int = 44100,
        target_lufs: float = -14.0,  # Streaming standard
        highpass_freq: float = 30.0,  # Remove sub-bass rumble
        lowpass_freq: float = 18000.0,  # Remove ultra-highs
    ):
        self.sr = sr
        self.target_lufs = target_lufs
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        
    def highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply highpass filter to remove rumble."""
        nyquist = self.sr / 2
        normalized_freq = self.highpass_freq / nyquist
        b, a = signal.butter(4, normalized_freq, btype='high')
        return signal.filtfilt(b, a, audio, axis=0)
    
    def lowpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply lowpass filter."""
        nyquist = self.sr / 2
        normalized_freq = min(self.lowpass_freq / nyquist, 0.99)
        b, a = signal.butter(4, normalized_freq, btype='low')
        return signal.filtfilt(b, a, audio, axis=0)
    
    def compute_lufs(self, audio: np.ndarray) -> float:
        """
        Compute integrated LUFS (simplified version).
        Based on ITU-R BS.1770-4.
        """
        # K-weighting filters (simplified)
        # High shelf
        nyquist = self.sr / 2
        high_shelf_freq = 1500 / nyquist
        b_high, a_high = signal.butter(2, high_shelf_freq, btype='high')
        
        # Apply K-weighting
        weighted = signal.filtfilt(b_high, a_high, audio, axis=0)
        
        # Mean square
        if audio.ndim == 1:
            ms = np.mean(weighted ** 2)
        else:
            # Stereo: sum channels
            ms = np.mean(np.sum(weighted ** 2, axis=1))
        
        # To LUFS
        if ms > 0:
            lufs = -0.691 + 10 * np.log10(ms)
        else:
            lufs = -70.0
            
        return lufs
    
    def normalize_lufs(self, audio: np.ndarray, target_lufs: float) -> np.ndarray:
        """Normalize audio to target LUFS."""
        current_lufs = self.compute_lufs(audio)
        
        if current_lufs < -60:
            logger.warning(f"Audio too quiet (LUFS: {current_lufs:.1f}), skipping normalization")
            return audio
            
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)
        
        normalized = audio * gain_linear
        
        # Soft clip if needed
        if np.max(np.abs(normalized)) > 1.0:
            normalized = np.tanh(normalized)
            
        return normalized
    
    def soft_clip(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Soft clipping/limiting."""
        # Gentle saturation above threshold
        mask = np.abs(audio) > threshold
        if np.any(mask):
            sign = np.sign(audio)
            magnitude = np.abs(audio)
            # Soft knee compression above threshold
            compressed = threshold + (1 - threshold) * np.tanh((magnitude - threshold) / (1 - threshold))
            audio = np.where(mask, sign * compressed, audio)
        return audio
    
    def simple_multiband_compress(self, audio: np.ndarray) -> np.ndarray:
        """
        Very simple 3-band compression to even out dynamics.
        """
        nyquist = self.sr / 2
        
        # Split into 3 bands
        low_freq = 250 / nyquist
        high_freq = 4000 / nyquist
        
        # Low band
        b_low, a_low = signal.butter(4, low_freq, btype='low')
        low = signal.filtfilt(b_low, a_low, audio, axis=0)
        
        # High band  
        b_high, a_high = signal.butter(4, high_freq, btype='high')
        high = signal.filtfilt(b_high, a_high, audio, axis=0)
        
        # Mid band
        mid = audio - low - high
        
        # Compress each band (simple RMS-based)
        def compress_band(band, ratio=3.0, threshold=-20):
            rms = np.sqrt(np.mean(band ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            
            if rms_db > threshold:
                gain_reduction = (rms_db - threshold) * (1 - 1/ratio)
                gain = 10 ** (-gain_reduction / 20)
                return band * gain
            return band
        
        low_comp = compress_band(low, ratio=4.0, threshold=-24)
        mid_comp = compress_band(mid, ratio=3.0, threshold=-20)
        high_comp = compress_band(high, ratio=2.5, threshold=-18)
        
        return low_comp + mid_comp + high_comp
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply full mastering chain.
        """
        # 1. Highpass to remove rumble
        audio = self.highpass_filter(audio)
        
        # 2. Simple multiband compression
        audio = self.simple_multiband_compress(audio)
        
        # 3. Lowpass
        audio = self.lowpass_filter(audio)
        
        # 4. Normalize to target LUFS
        audio = self.normalize_lufs(audio, self.target_lufs)
        
        # 5. Soft clip for safety
        audio = self.soft_clip(audio, threshold=0.95)
        
        return audio
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        target_sr: Optional[int] = None
    ) -> str:
        """
        Process a single file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed file
            target_sr: Output sample rate (None = same as input)
            
        Returns:
            Path to processed file
        """
        logger.info(f"Processing: {input_path}")
        
        # Load at processing sample rate
        audio, orig_sr = librosa.load(input_path, sr=self.sr, mono=False)
        
        # Handle mono vs stereo
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Process each channel
        processed_channels = []
        for ch in range(audio.shape[0]):
            processed = self.process(audio[ch])
            processed_channels.append(processed)
        
        processed = np.stack(processed_channels, axis=0)
        
        # Transpose for soundfile (samples, channels)
        if processed.shape[0] <= 2:
            processed = processed.T
        
        # Resample if needed
        out_sr = target_sr or self.sr
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed, out_sr)
        
        duration = len(audio[0]) / self.sr
        logger.info(f"  Saved: {output_path} ({duration/60:.1f} min)")
        
        return output_path


def process_training_pairs(
    input_dir: str = "training_data/input",
    output_dir: str = "training_data/desired_output",
    mastered_input_dir: str = "training_data/input_mastered",
    mastered_output_dir: str = "training_data/output_mastered",
    target_lufs: float = -14.0
):
    """
    Process all training pairs with consistent mastering.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    master = SimpleMaster(target_lufs=target_lufs)
    
    # Find pairs
    raw_files = list(input_path.glob("*_raw.wav"))
    
    logger.info(f"Found {len(raw_files)} raw files to process")
    
    for raw_file in tqdm(raw_files, desc="Mastering files"):
        base_name = raw_file.stem.replace("_raw", "")
        edit_file = output_path / f"{base_name}_edit.wav"
        
        if not edit_file.exists():
            logger.warning(f"No edit file for {raw_file.name}, skipping")
            continue
        
        # Process raw
        mastered_raw = Path(mastered_input_dir) / f"{base_name}_raw_mastered.wav"
        master.process_file(str(raw_file), str(mastered_raw))
        
        # Process edit
        mastered_edit = Path(mastered_output_dir) / f"{base_name}_edit_mastered.wav"
        master.process_file(str(edit_file), str(mastered_edit))
    
    logger.info("Done mastering all pairs!")
    logger.info(f"Mastered raw files in: {mastered_input_dir}")
    logger.info(f"Mastered edit files in: {mastered_output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Master training audio pairs")
    parser.add_argument("--input", "-i", default="training_data/input",
                       help="Directory with raw files")
    parser.add_argument("--output", "-o", default="training_data/desired_output",
                       help="Directory with edit files")
    parser.add_argument("--mastered-input", "-mi", default="training_data/input_mastered",
                       help="Where to save mastered raw files")
    parser.add_argument("--mastered-output", "-mo", default="training_data/output_mastered",
                       help="Where to save mastered edit files")
    parser.add_argument("--lufs", "-l", type=float, default=-14.0,
                       help="Target LUFS level")
    
    args = parser.parse_args()
    
    process_training_pairs(
        input_dir=args.input,
        output_dir=args.output,
        mastered_input_dir=args.mastered_input,
        mastered_output_dir=args.mastered_output,
        target_lufs=args.lufs
    )
