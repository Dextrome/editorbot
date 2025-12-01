# LoopRemix

A focused tool for remixing live jam recordings over loop pedals into proper songs.

## What it does

1. **Analyzes** a raw recording (10-30 minutes) to detect:
   - Tempo and beat grid
   - Loop/bar duration
   - Phrase boundaries (where musical ideas start/end)
   
2. **Segments** the recording into logical musical phrases

3. **Remixes** the best phrases into a cohesive song (2-9 minutes)

## Usage

```bash
python remix.py input.wav output.wav --target-duration 300
```

## Options

- `--target-duration` - Target length in seconds (default: 300 = 5 minutes)
- `--min-phrase-bars` - Minimum bars per phrase (default: 4)
- `--max-phrase-bars` - Maximum bars per phrase (default: 16)
- `--verbose` - Show detailed analysis info
