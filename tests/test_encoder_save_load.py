import numpy as np
import torch
from src.audio import AudioProcessor, AudioAnalyzer


def test_encoder_save_load_roundtrip(tmp_path):
    processor = AudioProcessor(sample_rate=16000)
    audio, sr = processor.load("data/samples/wartsnall3xx.wav")
    analyzer1 = AudioAnalyzer(sample_rate=sr)

    # Force CPU usage for determinism in test environment
    device = torch.device("cpu")

    # Instead of running the encoder forward (which may require CUDA-backed NATTEN),
    # create a NattenFrameEncoder instance, set it on the analyzer, and save its state.
    from src.ai.natten_encoder import NattenFrameEncoder

    encoder = NattenFrameEncoder(frame_size=256, proj_dim=8, kernel_size=7, num_heads=1)
    analyzer1.encoder = encoder
    analyzer1.encoder_frame_size = 256
    analyzer1.encoder_proj_dim = 8

    # Save encoder to disk
    out_file = tmp_path / "encoder.pt"
    analyzer1.save_encoder(out_file)

    # Create a new analyzer and load encoder
    analyzer2 = AudioAnalyzer(sample_rate=sr)
    analyzer2.load_encoder(out_file, map_location="cpu")

    # Compare state dicts of the saved and loaded encoder
    sd1 = {k: v.cpu().numpy() for k, v in analyzer1.encoder.state_dict().items()}
    sd2 = {k: v.cpu().numpy() for k, v in analyzer2.encoder.state_dict().items()}

    assert sd1.keys() == sd2.keys()
    for k in sd1.keys():
        assert np.allclose(sd1[k], sd2[k], atol=1e-6)
