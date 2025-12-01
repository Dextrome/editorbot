import os
import sys
import numpy as np
import pytest
from src.ai.editor import AIEditor


@pytest.mark.skipif(True, reason="E2E Demucs tests are heavy and optional - run manually")
def test_e2e_wartsnall_gap_heal_regression(tmp_path):
    # E2E regression test to ensure wartsnall12-V2.wav has no 200ms near-zero runs at amplitude threshold 0.01
    sample_file = os.path.join("data", "samples", "wartsnall12-V2.wav")
    assert os.path.exists(sample_file), "Sample file missing for E2E regression test"
    editor = AIEditor(demucs_device='cpu', remixatron_max_jump=None, remixatron_gap_heal_ms=250, remixatron_gap_heal_threshold=0.01, remixatron_gap_mode='heal')
    out_file = tmp_path / "healed.wav"
    result = editor.process_file(sample_file, out_file, preset='balanced', arrange=False, allow_rearrange=False)
    out_path = result.get('output_path')
    # Load result and detect near-zero runs of >=200ms length (threshold 0.01)
    import soundfile as sf
    arr, sr = sf.read(out_path)
    mono = np.mean(np.abs(arr), axis=1)
    is_small = mono <= 0.01
    runs = []
    start = None
    for i, v in enumerate(is_small):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(is_small) - start))
    # Ensure no runs >=200ms
    min_samples = int(0.200 * sr)
    assert not any(l >= min_samples for s, l in runs), f"Found large near-zero runs: {runs}"
