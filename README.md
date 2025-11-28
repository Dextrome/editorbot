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
