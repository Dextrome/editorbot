"""Convert cached stem audio files to mel spectrograms.

The precache_stems.py saved raw audio waveforms. This script converts them
to mel spectrograms that the pointer network model expects.

Usage:
    python scripts/convert_stems_to_mel.py
"""
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# Audio params (must match training)
SR = 22050
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    """Convert raw audio to normalized mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80) / 80  # Normalize to [0, 1]
    mel_db = np.clip(mel_db, 0, 1)
    return mel_db


def main():
    stems_dir = Path("F:/editorbot/cache/stems")
    output_dir = Path("F:/editorbot/cache/stems_mel")
    output_dir.mkdir(exist_ok=True)

    stem_files = list(stems_dir.glob("*_stems.npz"))
    print(f"Found {len(stem_files)} stem files to convert")

    for stem_file in tqdm(stem_files, desc="Converting stems to mel"):
        output_file = output_dir / stem_file.name

        # Skip if already converted
        if output_file.exists():
            continue

        try:
            data = np.load(stem_file)

            # Convert each stem to mel
            stems_mel = {}
            for stem_name in ['drums', 'bass', 'vocals', 'other']:
                if stem_name in data:
                    audio = data[stem_name]
                    mel = audio_to_mel(audio)
                    stems_mel[stem_name] = mel

            # Save as npz
            np.savez_compressed(output_file, **stems_mel)

        except Exception as e:
            print(f"Error converting {stem_file.name}: {e}")

    print(f"Done! Converted stems saved to {output_dir}")


if __name__ == "__main__":
    main()
