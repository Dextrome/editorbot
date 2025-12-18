"""Dump policy logits/probs and action distributions for two checkpoints.

Loads checkpoints using a safe loader (fallback to full torch.load with
`torch.serialization.add_safe_globals` for legacy pickles), partially loads
policy weights into an `Agent`, runs a small batch of dummy states and
prints mean logits, softmax probabilities and sampled action counts.

Usage:
  python scripts/dump_policy_debug.py --before models/modelV1/checkpoint_epoch_24100.pt \
      --after models/modelV1/checkpoint_epoch_24200.pt
"""
import argparse
from pathlib import Path
import sys
import os
import torch
import numpy as np

# Ensure workspace root is on sys.path so `rl_editor` can be imported when the
# script is run directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_editor.config import get_default_config
from rl_editor.agent import Agent
from rl_editor.actions import N_ACTION_TYPES, N_ACTION_SIZES, N_ACTION_AMOUNTS


def safe_load(path):
    try:
        return torch.load(str(path), map_location='cpu')
    except Exception as e:
        # Fallback: allowlist numpy._core.multiarray.scalar (works in many numpy builds)
        try:
            with torch.serialization.add_safe_globals([np._core.multiarray.scalar]):
                return torch.load(str(path), map_location='cpu', weights_only=False)
        except Exception:
            # Re-raise original exception if fallback failed
            raise


def extract_policy_state(ckpt):
    # Common keys used in checkpoints
    for key in ('policy_state_dict', 'policy_net', 'policy'):
        if key in ckpt:
            return ckpt[key]
    # If checkpoint stores merged dict with 'policy_net' nested
    for k in ckpt.keys():
        if isinstance(ckpt[k], dict) and any('type_head' in x for x in ckpt[k].keys()):
            return ckpt[k]
    return None


def load_policy_into_agent(agent, ckpt_path, device='cpu'):
    ckpt = safe_load(ckpt_path)
    policy_state = extract_policy_state(ckpt)
    if policy_state is None:
        raise RuntimeError(f'No policy state found in {ckpt_path}')
    # Partial load allowed
    agent.policy_net.load_state_dict(policy_state, strict=False)
    return ckpt


def analyze(agent, device='cpu', n_samples=1024):
    agent.policy_net.to(device)
    agent.policy_net.eval()

    # Use a simple zero-state batch with input_dim inferred from encoder
    # Determine expected input size from first linear weight if possible
    # Fallback to 121
    input_dim = 121
    try:
        # encoder.input_projection weight shape: (hidden_dim, input_dim)
        w = agent.policy_net.encoder.input_projection.weight
        input_dim = w.shape[1]
    except Exception:
        pass

    batch = max(64, n_samples)
    # Policy expects 2D state (batch, input_dim) -> encoder will squeeze to seq
    states = torch.zeros(batch, input_dim, device=device)

    with torch.no_grad():
        type_logits, size_logits, amount_logits = agent.policy_net(states)

        type_probs = torch.softmax(type_logits, dim=-1)
        size_probs = torch.softmax(size_logits, dim=-1)
        amount_probs = torch.softmax(amount_logits, dim=-1)

        # Mean probabilities
        mean_type_prob = type_probs.mean(dim=0).cpu().numpy()
        mean_size_prob = size_probs.mean(dim=0).cpu().numpy()
        mean_amount_prob = amount_probs.mean(dim=0).cpu().numpy()

        # Sample actions many times to get empirical distribution
        n_draws = n_samples
        type_dist = torch.distributions.Categorical(probs=type_probs)
        draws = type_dist.sample((1,)).squeeze(0)
        # draws shape (batch,)
        vals, counts = torch.unique(draws, return_counts=True)
        type_counts = {int(k.item()): int(v.item()) for k, v in zip(vals, counts)}

    return {
        'input_dim': input_dim,
        'mean_type_prob': mean_type_prob.tolist(),
        'mean_size_prob': mean_size_prob.tolist(),
        'mean_amount_prob': mean_amount_prob.tolist(),
        'type_counts': type_counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-samples', type=int, default=1024)
    args = p.parse_args()

    device = args.device
    cfg = get_default_config()
    agent = Agent(cfg, input_dim=121)

    print('Loading before checkpoint:', args.before)
    before_ckpt = load_policy_into_agent(agent, args.before, device=device)
    before_stats = analyze(agent, device=device, n_samples=args.n_samples)

    print('Loading after checkpoint:', args.after)
    after_ckpt = load_policy_into_agent(agent, args.after, device=device)
    after_stats = analyze(agent, device=device, n_samples=args.n_samples)

    out = {
        'before': before_stats,
        'after': after_stats,
    }

    import json
    out_path = Path('output/policy_debug_summary.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print('Wrote', out_path)
    print('Before mean type probs:', before_stats['mean_type_prob'])
    print('After mean type probs :', after_stats['mean_type_prob'])
    print('Before type counts (sampled):', before_stats['type_counts'])
    print('After type counts (sampled) :', after_stats['type_counts'])


if __name__ == '__main__':
    main()
