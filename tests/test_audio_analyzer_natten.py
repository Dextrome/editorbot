import numpy as np
import torch
from src.audio import AudioProcessor, AudioAnalyzer


def test_audio_analyzer_extract_attention_features():
    processor = AudioProcessor(sample_rate=16000)
    audio, sr = processor.load("data/samples/wartsnall3xx.wav")
    analyzer = AudioAnalyzer(sample_rate=sr)
    # Run with modest frame/hop to keep runtime reasonable
    feats = analyzer.extract_attention_features(audio, frame_size=256, hop_size=128, proj_dim=8, kernel_size=7)
    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 2
    assert feats.shape[1] == 8
    assert np.isfinite(feats).all()
