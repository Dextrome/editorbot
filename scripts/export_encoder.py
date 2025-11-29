#!/usr/bin/env python
"""
Export a NATTEN frame encoder to disk.

Usage examples:
  python scripts/export_encoder.py --out models/encoder.pt --frame-size 256 --proj-dim 8
  python scripts/export_encoder.py --out encoder.pt --frame-size 256 --proj-dim 16 --seed 42

The saved file matches the payload format used by `AudioAnalyzer.save_encoder`:
{"state_dict": ..., "frame_size": <int>, "proj_dim": <int>}
"""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Create and save a NattenFrameEncoder")
    parser.add_argument("--frame-size", type=int, required=True, help="Frame size in samples")
    parser.add_argument("--proj-dim", type=int, default=8, help="Projection dimension")
    parser.add_argument("--kernel-size", type=int, default=7, help="NATTEN kernel size")
    parser.add_argument("--num-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--out", required=True, help="Output path for encoder file")
    parser.add_argument("--device", default=None, help="Device to map encoder to when saving (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic init")

    args = parser.parse_args()

    if args.seed is not None:
        import random

        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Lazy import to avoid heavy deps at module import time
    from src.ai.natten_encoder import NattenFrameEncoder

    encoder = NattenFrameEncoder(frame_size=args.frame_size, proj_dim=args.proj_dim, kernel_size=args.kernel_size, num_heads=args.num_heads)

    # Map to device for saving if requested
    map_device = args.device
    if map_device is None:
        map_device = "cpu"

    # Prepare payload
    payload = {
        "state_dict": encoder.state_dict(),
        "frame_size": args.frame_size,
        "proj_dim": args.proj_dim,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with torch.save (CPU-friendly)
    torch.save(payload, str(out_path))
    print(f"Saved encoder to: {out_path}")


if __name__ == "__main__":
    main()
