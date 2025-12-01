"""
demucs_wrapper.py - Utility to run Demucs and return separated stems as numpy arrays.
"""
import os
import logging
import tempfile
import subprocess
import sys
import shutil
import numpy as np
import soundfile as sf
import librosa
from typing import Dict, Optional

class DemucsSeparator:
    def __init__(self, model: str = "htdemucs"):
        self.model = model

    def separate(self, audio_path: str, output_dir: Optional[str] = None, resample_to: Optional[int] = None, device: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Run Demucs on the given audio file and return a dict of stems as numpy arrays.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory where Demucs will write outputs. If None, a temporary dir will be used.
            resample_to: Optional target sample rate for returned stems; resample if necessary.
            device: Optional override for Demucs device (e.g., 'cuda' or 'cpu'). When None, the wrapper will
               detect and pick 'cuda' if `torch.cuda.is_available()` returns True, otherwise 'cpu'.

        Returns:
            Dict mapping stem name to numpy array. The returned dict includes a '_sr' entry with sample rate if available.
        """
        if output_dir is None:
            tmpdir = tempfile.TemporaryDirectory()
            output_dir = tmpdir.name
        else:
            tmpdir = None

        # Prefer invoking demucs via the current Python interpreter to ensure the same venv is used
        # i.e. `python -m demucs ...` avoids relying on PATH for the demucs script
        # First check that the demucs module is importable in this Python environment
        try:
            subprocess.run([sys.executable, "-c", "import demucs"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"Demucs Python package is not installed in the active Python environment ({sys.executable}).\n"
                "Install it with: 'pip install demucs' (using the same Python executable), or follow the README instructions."
            )

        # Auto-detect device using torch if device isn't provided
        auto_device = device
        if auto_device is None:
            try:
                import importlib
                torch_mod = importlib.import_module('torch')
                auto_device = 'cuda' if getattr(torch_mod, 'cuda', None) and torch_mod.cuda.is_available() else 'cpu'
            except Exception:
                auto_device = 'cpu'

        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "--out",
            output_dir,
            "-n",
            self.model,
            "--device",
            auto_device,
            audio_path,
        ]
        logger = logging.getLogger(__name__)
        logger.info("[DemucsSeparator] running demucs command: %s --device %s ...", ' '.join(cmd[:6]), auto_device)
        # Ensure ffmpeg is in PATH when running Demucs. If not available, try the imageio_ffmpeg binary as fallback.
        env = os.environ.copy()
        ffmpeg_bin = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
        if ffmpeg_bin is None:
            try:
                import importlib
                imageio_ffmpeg = importlib.import_module('imageio_ffmpeg')
                # Use get_ffmpeg_exe if available, fall back to get_exe for older versions
                ffmpeg_exe = None
                if hasattr(imageio_ffmpeg, 'get_ffmpeg_exe'):
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                elif hasattr(imageio_ffmpeg, 'get_exe'):
                    ffmpeg_exe = imageio_ffmpeg.get_exe()
                else:
                    # Inspect the package for probable binary path if no helper exists
                    ffmpeg_exe = None
                # Add the containing directory to PATH
                if ffmpeg_exe is not None:
                    env['PATH'] = env.get('PATH', '') + os.pathsep + os.path.dirname(ffmpeg_exe)
                    logger.info("[DemucsSeparator] Added imageio-ffmpeg binary dir to PATH: %s", os.path.dirname(ffmpeg_exe))
                else:
                    logger.warning("[DemucsSeparator] imageio-ffmpeg installed but couldn't resolve the binary executable location.")
            except Exception:
                # No ffmpeg available at all; continue and let demucs failure message be clear
                logger.warning("[DemucsSeparator] Warning: ffmpeg not found on PATH and imageio_ffmpeg not installed. Demucs may fail for some inputs.")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            # demucs module exists, but it returned an error while running
            raise RuntimeError(f"Demucs failed during separation (return code: {e.returncode}). See stderr for details.") from e

        # Demucs outputs to output_dir/model/songname/vocals.wav, drums.wav, bass.wav, other.wav
        songname = os.path.splitext(os.path.basename(audio_path))[0]
        stem_dir = os.path.join(output_dir, self.model, songname)
        stems = {}
        sr = None
        for stem in ["drums", "bass", "other", "vocals"]:
            stem_path = os.path.join(stem_dir, f"{stem}.wav")
            if os.path.exists(stem_path):
                y, sr = sf.read(stem_path)
                # If requested, resample to target rate
                if resample_to is not None and sr is not None and sr != resample_to:
                    # librosa expects float arrays. Ensure y is float32
                    y = y.astype(np.float32)
                    if y.ndim == 1:
                        y = librosa.resample(y, orig_sr=sr, target_sr=resample_to)
                    else:
                        # Resample each channel independently
                        y = np.stack([librosa.resample(y[:, ch], orig_sr=sr, target_sr=resample_to) for ch in range(y.shape[1])], axis=-1)
                    sr = resample_to
                stems[stem] = y
        if sr is not None:
            stems['_sr'] = sr
        if tmpdir:
            tmpdir.cleanup()
        return stems
