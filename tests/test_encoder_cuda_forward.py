import pytest
import torch
import numpy as np
from src.audio import AudioAnalyzer
from src.ai.natten_encoder import NattenFrameEncoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this test")
def test_encoder_forward_roundtrip_cuda(tmp_path):
    # Create small synthetic audio for quick GPU forward
    sr = 16000
    frame_size = 256
    hop_size = 128
    n_frames = 32
    # create audio long enough for n_frames
    audio = np.random.randn(frame_size + (n_frames - 1) * hop_size).astype(np.float32)

    analyzer1 = AudioAnalyzer(sample_rate=sr)
    encoder = NattenFrameEncoder(frame_size=frame_size, proj_dim=8, kernel_size=7, num_heads=1).to('cuda')
    analyzer1.set_encoder(encoder, frame_size=frame_size, proj_dim=8, device=torch.device('cuda'))

    # Forward with the original encoder on GPU
    feats1 = analyzer1.extract_attention_features(audio, frame_size=frame_size, hop_size=hop_size, proj_dim=8, kernel_size=7, device=torch.device('cuda'))

    # Save encoder
    out_file = tmp_path / "encoder_cuda.pt"
    analyzer1.save_encoder(out_file)

    # Load into new analyzer on GPU
    analyzer2 = AudioAnalyzer(sample_rate=sr)
    analyzer2.load_encoder(out_file, map_location=torch.device('cuda'))

    # Forward with loaded encoder
    feats2 = analyzer2.extract_attention_features(audio, frame_size=frame_size, hop_size=hop_size, proj_dim=8, kernel_size=7, device=torch.device('cuda'))

    assert feats1.shape == feats2.shape
    assert np.allclose(feats1, feats2, atol=1e-5)
