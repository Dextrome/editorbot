import os
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from rl_editor.infer import load_and_process_audio, get_default_config
from rl_editor.agent import Agent
from rl_editor.infer import run_inference

import torch
import random
import numpy as np


def entropy_from_counts(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def run_for_checkpoint(checkpoint_path: str, input_audio: str, outdir: Path, n_samples: int = 32):
    outdir.mkdir(parents=True, exist_ok=True)
    config = get_default_config()
    audio, sr, audio_state = load_and_process_audio(input_audio, config, max_beats=500, cache_dir=config.data.cache_dir)

    # Initialize agent
    from rl_editor.environment import AudioEditingEnvFactored
    env = AudioEditingEnvFactored(config, audio_state)
    obs, _ = env.reset()
    input_dim = len(obs)

    agent = Agent(config, input_dim=input_dim, beat_feature_dim=audio_state.beat_features.shape[1], use_auxiliary_tasks=False)
    agent.load(checkpoint_path)

    rows = []
    all_type_counts = defaultdict(int)
    all_size_counts = defaultdict(int)

    for i in range(n_samples):
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        actions, action_names, score = run_inference(agent, config, audio_state, deterministic=False, verbose=False)

        type_counts = Counter([a.action_type.name for a in actions])
        size_counts = Counter([a.action_size.name for a in actions])
        n_actions = len(actions)

        type_entropy = entropy_from_counts(type_counts)
        size_entropy = entropy_from_counts(size_counts)

        # aggregate
        for k,v in type_counts.items():
            all_type_counts[k] += v
        for k,v in size_counts.items():
            all_size_counts[k] += v

        rows.append({
            'sample': i,
            'seed': seed,
            'score': float(score),
            'n_actions': n_actions,
            'type_entropy_bits': type_entropy,
            'size_entropy_bits': size_entropy,
            'type_counts': json.dumps(dict(type_counts)),
            'size_counts': json.dumps(dict(size_counts)),
        })

    # write per-checkpoint CSV
    ck_name = Path(checkpoint_path).stem
    per_csv = outdir / f"{ck_name}_inference.csv"
    with open(per_csv, 'w', newline='') as f:
        fieldnames = ['sample','seed','score','n_actions','type_entropy_bits','size_entropy_bits','type_counts','size_counts']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # compute summary
    scores = [r['score'] for r in rows]
    type_entropies = [r['type_entropy_bits'] for r in rows]
    size_entropies = [r['size_entropy_bits'] for r in rows]
    n_actions = [r['n_actions'] for r in rows]

    def mean_std(arr):
        import statistics
        return statistics.mean(arr), statistics.pstdev(arr)

    summary = {
        'checkpoint': checkpoint_path,
        'n_samples': n_samples,
        'mean_score': mean_std(scores)[0],
        'std_score': mean_std(scores)[1],
        'mean_type_entropy': mean_std(type_entropies)[0],
        'std_type_entropy': mean_std(type_entropies)[1],
        'mean_size_entropy': mean_std(size_entropies)[0],
        'std_size_entropy': mean_std(size_entropies)[1],
        'mean_n_actions': mean_std(n_actions)[0],
        'total_type_counts': json.dumps(dict(all_type_counts)),
        'total_size_counts': json.dumps(dict(all_size_counts)),
    }

    return per_csv, summary


def main():
    checkpoints = [
        'models/modelV1/checkpoint_epoch_24200.pt',
        'models/modelV1/best.pt',
        'models/modelV1/checkpoint_epoch_22600.pt',
        'models/modelV1_resume_test2/final.pt',
    ]
    input_audio = 'training_data/input/TheMachine-SolarCorona_synth_raw.wav'
    outdir = Path('output/reports/batch_compare')
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for ck in checkpoints:
        print('Processing', ck)
        per_csv, summary = run_for_checkpoint(ck, input_audio, outdir, n_samples=32)
        print('Wrote', per_csv)
        summaries.append(summary)

    # write aggregate summary CSV
    agg_csv = outdir / 'all_checkpoints_summary.csv'
    with open(agg_csv, 'w', newline='') as f:
        fieldnames = ['checkpoint','n_samples','mean_score','std_score','mean_type_entropy','std_type_entropy','mean_size_entropy','std_size_entropy','mean_n_actions','total_type_counts','total_size_counts']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print('Wrote aggregate summary:', agg_csv)

if __name__ == '__main__':
    main()
