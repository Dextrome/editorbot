# AI Audio Editor

An AI-powered audio editor that transforms raw music recordings into polished, listenable songs.

## Features

- **Automatic Audio Analysis**: Detects tempo, key, loudness, and structural sections
- **Intelligent Editing**: AI-driven decisions for optimal audio processing
- **Multiple Presets**: Choose from balanced, warm, bright, or aggressive editing styles
- **Audio Effects**: Built-in EQ, compression, reverb, noise gate, and noise reduction
- **Batch Processing**: Process multiple files at once
- **Neural Enhancement**: Deep learning models for advanced audio enhancement

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- FFmpeg (for audio format support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-audio-editor.git
cd ai-audio-editor
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

If you prefer to automate venv and dependency installation on Windows, we provide a helper script `scripts/install_env_deps.ps1`:

```powershell
# Create venv, install requirements, and optionally install system ffmpeg (admin required)
powershell -ExecutionPolicy Bypass -File .\scripts\install_env_deps.ps1 -InstallSystemFFmpeg
```

### Optional: Install Demucs for stem separation (recommended)

If you intend to use Demucs for stem separation, install it into the same environment used to run the editor. Demucs leverages PyTorch and can take advantage of a GPU if available.

```bash
# Use the venv Python executable or activate the venv first
python -m pip install demucs
```

Note: This project no longer uses or supports Spleeter. Demucs is the recommended and supported stem separator. The `--stem-method` CLI option now accepts only `demucs`.

You can verify Demucs is available in your environment with:

```bash
# Should print Demucs CLI help
python -m demucs --help
```

If you are using a GPU, ensure you have the correct PyTorch wheel for your CUDA version:

```bash
# Example install for CUDA 12.4; choose the correct wheel for your GPU
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
# Then install demucs
python -m pip install demucs
```

### Environment validation
We include a helper script to validate the active Python environment:

```bash
python scripts/check_env.py
```
This will report installed versions and whether CUDA / `ffmpeg` are available.

If no system `ffmpeg` binary is present on PATH, the project includes `imageio-ffmpeg` in `requirements.txt` which provides a bundled ffmpeg executable that the Demucs wrapper can use as a fallback.

### Device selection

`DemucsSeparator` automatically selects CUDA when PyTorch detects an available GPU. If you need to force CPU mode, supply the `device` override when calling the separator from Python:

```python
from src.audio.demucs_wrapper import DemucsSeparator
separator = DemucsSeparator()
# Force CPU device to avoid GPU usage
separator.separate('input.wav', device='cpu')
```

You can also force the Demucs device from the CLI using the `--demucs-device` flag:

```bash
# Force Demucs to run on CPU
python main.py input.wav -o output.wav --demucs-device cpu

# Force Demucs to run on CUDA (GPU)
python main.py input.wav -o output.wav --demucs-device cuda
```



## Usage

### Command Line Interface

Process a single audio file:
```bash
python main.py input.wav -o output.wav -p balanced
```

Batch process a directory:
```bash
python main.py ./recordings -o ./edited -b -p warm
```

### Options

- `-o, --output`: Output file or directory path
- `-p, --preset`: Editing preset (balanced, warm, bright, aggressive)
- `-b, --batch`: Enable batch processing mode
- `-v, --verbose`: Enable verbose output
- `--demucs-device`: Force Demucs device selection (`cpu` or `cuda`). If omitted, device auto-detects.
- `--log-level`: Set logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL). Use `--log-level DEBUG` for verbose logs.
- `--remixatron-max-jump`: Limit allowed beat jump (in beats) for Remixatron to improve musical continuity. Smaller values keep transitions local and reduce creative jumps (e.g. `--remixatron-max-jump 8`).
- `--remixatron-max-jump`: Limit allowed beat jump (in beats) for Remixatron to improve musical continuity. Smaller values keep transitions local and reduce creative jumps (e.g. `--remixatron-max-jump 8`).

Remixatron behavior and blend stability
-------------------------------------
- The Remixatron adapter uses Stem Based blending (Demucs) and performs one Demucs pass per track (cached for the whole run). This improves blend continuity and prevents repeated separation overhead.
- Small positive gaps between beats are now detected and zero-padded before blending, avoiding audible silence breaks.
- Transitions are crossfaded using adaptive blend lengths based on the actual beat slice sizes and a RMS-level match to reduce clipping and level jumps.

Example: run with smoothing and Demucs controlled device (CPU):
```
python main.py data/samples/wartsnall12-V2.wav --remixatron --demucs-device cpu --remixatron-max-jump 8 --log-level DEBUG -o output/remix.wav
```

### Python API

```python
from src.ai.editor import AIEditor

# Initialize the editor
editor = AIEditor()

# Process a single file
result = editor.process_file("input.wav", "output.wav", preset="balanced")

# Access analysis results
print(f"Tempo: {result['analysis']['tempo']} BPM")
print(f"Key: {result['analysis']['key']}")

# Batch process
results = editor.batch_process("./recordings", "./edited", preset="warm")
```

### Low-Level Processing

```python
from src.audio.processor import AudioProcessor
from src.audio.analyzer import AudioAnalyzer
from src.audio.effects import AudioEffects

# Load audio
processor = AudioProcessor()
audio, sr = processor.load("input.wav")

# Analyze
analyzer = AudioAnalyzer()
tempo = analyzer.detect_tempo(audio)
key = analyzer.detect_key(audio)

# Apply effects
effects = AudioEffects()
audio = effects.apply_eq(audio, low_gain=2.0, high_gain=3.0)
audio = effects.apply_compression(audio, threshold=-18.0, ratio=4.0)
audio = effects.apply_reverb(audio, room_size=0.4, wet_level=0.2)

# Save
processor.save("output.wav", audio)
```

## Presets

| Preset | Description |
|--------|-------------|
| **balanced** | Natural sound with subtle enhancement |
| **warm** | Enhanced bass, smooth highs |
| **bright** | Crisp highs, reduced bass |
| **aggressive** | Heavy compression, punchy sound |

## Project Structure

```
ai-audio-editor/
├── src/
│   ├── audio/           # Audio processing modules
│   │   ├── processor.py # Core audio I/O
│   │   ├── analyzer.py  # Feature extraction
│   │   └── effects.py   # Audio effects
│   ├── ai/              # AI/ML modules
│   │   ├── editor.py    # Main AI editor
│   │   └── models.py    # Neural network models
│   └── utils/           # Utility functions
│       ├── helpers.py   # Helper functions
│       └── config.py    # Configuration management
├── tests/               # Unit tests
├── data/                # Sample audio files
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
└── pyproject.toml       # Project configuration
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [librosa](https://librosa.org/) - Audio analysis library
- [PyTorch](https://pytorch.org/) - Machine learning framework
- [soundfile](https://pysoundfile.readthedocs.io/) - Audio file I/O
