import subprocess
import sys
import soundfile as sf
import numpy as np
from pathlib import Path

pairs = [
    ("training_data/input/TheMachine-SolarCorona_synth_raw.wav", "training_data/desired_output/TheMachine-SolarCorona_synth_edit.wav"),
    ("training_data/input/Cream-Sunshine_synth_raw.wav", "training_data/desired_output/Cream-Sunshine_synth_edit.wav"),
    ("training_data/input/walkerjam_synth_raw.wav", "training_data/desired_output/walkerjam_synth_edit.wav"),
]

out_dir = Path("output/diag")
out_dir.mkdir(parents=True, exist_ok=True)

checkpoint = "models/modelV1/best.pt"

results = []
for inp, ref in pairs:
    inp_p = Path(inp)
    ref_p = Path(ref)
    out_p = out_dir / (inp_p.stem + "_edited" + inp_p.suffix)

    cmd = [
        sys.executable, "-m", "rl_editor.infer", str(inp_p),
        "--output", str(out_p),
        "--checkpoint", checkpoint,
        "--deterministic"
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Inference command failed with returncode={e.returncode}")
        raise

    # Load audio
    y_out, sr_out = sf.read(str(out_p))
    y_ref, sr_ref = sf.read(str(ref_p))

    # Resample if sample rates differ
    if sr_out != sr_ref:
        print(f"Sample rates differ: out={sr_out} ref={sr_ref} - skipping resample")

    # Match lengths
    min_len = min(len(y_out), len(y_ref))
    y_out = y_out[:min_len]
    y_ref = y_ref[:min_len]

    mse = float(np.mean((y_out - y_ref) ** 2))
    l2 = float(np.linalg.norm(y_out - y_ref))

    results.append((inp_p.name, float(mse), float(l2), str(out_p)))

print("\nDiagnostics results:")
for name, mse, l2, outp in results:
    print(f"{name}: MSE={mse:.6e}, L2={l2:.6f}, out={outp}")
