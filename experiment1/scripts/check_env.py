#!/usr/bin/env python
"""
scripts/check_env.py
Utility to check Python environment for key dependencies: Python, torch, CUDA, demucs, ffmpeg, librosa, soundfile, numpy
"""
from __future__ import annotations
import sys
import shutil
import subprocess

def print_ok(k, v):
    print(f"{k}: {v}")

def check_module(name):
    try:
        m = __import__(name)
        version = getattr(m, '__version__', 'unknown')
        return True, version
    except Exception as e:
        return False, str(e)

def main():
    print_ok("Python", sys.executable)
    # Python version
    print_ok("Python version", sys.version.replace('\n', ' '))

    # Torch
    ok, ver = check_module('torch')
    if ok:
        import torch
        cuda = torch.cuda.is_available()
        print_ok('torch', ver + (" (cuda available)" if cuda else " (cuda not available)"))
    else:
        print_ok('torch', 'not installed: ' + ver)

    # Demucs
    ok, ver = check_module('demucs')
    if ok:
        print_ok('demucs', ver)
    else:
        print_ok('demucs', 'not installed: ' + ver)

    # CLI check - try invoking `python -m demucs --help`
    try:
        r = subprocess.run([sys.executable, '-m', 'demucs', '--help'], capture_output=True, text=True, check=True)
        print_ok('demucs.cli', 'works')
    except Exception as e:
        print_ok('demucs.cli', f'failed: {e}')

    # ffmpeg
    ff = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
    print_ok('ffmpeg', ff or 'not found')
    # If ffmpeg not on PATH, check for imageio-ffmpeg binary
    if not ff:
        ok_im, ver_im = check_module('imageio_ffmpeg')
        if ok_im:
            try:
                import imageio_ffmpeg as imff
                ff_exe = None
                if hasattr(imff, 'get_ffmpeg_exe'):
                    ff_exe = imff.get_ffmpeg_exe()
                elif hasattr(imff, 'get_exe'):
                    ff_exe = imff.get_exe()
                if ff_exe:
                    print_ok('imageio-ffmpeg binary', ff_exe)
                else:
                    print_ok('imageio-ffmpeg binary', 'installed but exe retrieval failed')
            except Exception as e:
                print_ok('imageio-ffmpeg binary', 'installed but exe retrieval failed: ' + str(e))
        else:
            print_ok('imageio-ffmpeg', 'not installed')

    # librosa
    ok, ver = check_module('librosa')
    print_ok('librosa', ver if ok else 'not installed')

    # soundfile
    ok, ver = check_module('soundfile')
    print_ok('soundfile', ver if ok else 'not installed')

    # numpy
    ok, ver = check_module('numpy')
    print_ok('numpy', ver if ok else 'not installed')

    # Quick tips
    print('\nTip: install demucs into the active environment with:')
    print(f"  {sys.executable} -m pip install demucs")

if __name__ == '__main__':
    main()
