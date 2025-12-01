import numpy as np
from src.audio import AudioProcessor, AudioAnalyzer


def test_natten_based_detectors():
    processor = AudioProcessor(sample_rate=16000)
    audio, sr = processor.load("data/samples/wartsnall3xx.wav")
    analyzer = AudioAnalyzer(sample_rate=sr)

    # Tempo using NATTEN features
    tempo = analyzer.detect_tempo(audio, use_natten=True)
    assert isinstance(tempo, float)
    assert tempo > 0

    # Beats using NATTEN features
    beats = analyzer.detect_beats(audio, use_natten=True)
    assert beats.ndim == 1

    # Sections using NATTEN features
    sections, labels = analyzer.detect_sections(audio, num_segments=4, use_natten=True)
    assert sections.ndim >= 1
    assert len(labels) == len(sections)
