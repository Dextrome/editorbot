"""Package entry: dump policy debug (same as top-level script but importable).

Run via: python -m rl_editor.scripts.dump_policy_debug --before <path> --after <path>
"""
import argparse
from pathlib import Path
import torch
import numpy as np

from rl_editor.config import get_default_config
from rl_editor.agent import Agent


def safe_load(path):
    try:
        return torch.load(str(path), map_location='cpu')
    except Exception:
        with torch.serialization.add_safe_globals([np._core.multiarray.scalar]):
            return torch.load(str(path), map_location='cpu', weights_only=False)


def extract_policy_state(ckpt):
    for key in ('policy_state_dict', 'policy_net', 'policy'):
        if key in ckpt:
            return ckpt[key]
    for k in ckpt.keys():
        if isinstance(ckpt[k], dict) and any('type_head' in x for x in ckpt[k].keys()):
            return ckpt[k]
    return None


def load_policy_into_agent(agent, ckpt_path):
    ckpt = safe_load(ckpt_path)
    policy_state = extract_policy_state(ckpt)
    if policy_state is None:
        raise RuntimeError(f'No policy state found in {ckpt_path}')
    agent.policy_net.load_state_dict(policy_state, strict=False)
    return ckpt


def analyze(agent, n_samples=1024):
    agent.policy_net.eval()
    # Try to infer input_dim from encoder
    input_dim = 121
    try:
        w = agent.policy_net.encoder.input_projection.weight
        input_dim = w.shape[1]
    except Exception:
        pass

    batch = max(64, n_samples)
    states = torch.zeros(batch, input_dim)

    with torch.no_grad():
        type_logits, size_logits, amount_logits = agent.policy_net(states)
        type_probs = torch.softmax(type_logits, dim=-1)
        size_probs = torch.softmax(size_logits, dim=-1)
        amount_probs = torch.softmax(amount_logits, dim=-1)

        mean_type_prob = type_probs.mean(dim=0).cpu().numpy()
        mean_size_prob = size_probs.mean(dim=0).cpu().numpy()
        mean_amount_prob = amount_probs.mean(dim=0).cpu().numpy()

        # Sample type actions from the batch distribution
        type_dist = torch.distributions.Categorical(probs=type_probs)
        draws = type_dist.sample().cpu()
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
    p.add_argument('--n-samples', type=int, default=1024)
    args = p.parse_args()

    cfg = get_default_config()
    agent = Agent(cfg, input_dim=121)

    print('Loading before checkpoint:', args.before)
    load_policy_into_agent(agent, args.before)
    before_stats = analyze(agent, n_samples=args.n_samples)

    print('Loading after checkpoint:', args.after)
    load_policy_into_agent(agent, args.after)
    after_stats = analyze(agent, n_samples=args.n_samples)

    out = {'before': before_stats, 'after': after_stats}
    import json
    out_path = Path('output/policy_debug_summary_pkg.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print('Wrote', out_path)
    print('Before mean type probs:', before_stats['mean_type_prob'])
    print('After mean type probs :', after_stats['mean_type_prob'])
    print('Before type counts (sampled):', before_stats['type_counts'])
    print('After type counts (sampled) :', after_stats['type_counts'])


if __name__ == '__main__':
    main()
