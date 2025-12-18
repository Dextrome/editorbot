import csv
import json
from pathlib import Path
from collections import Counter
import math

import sys, os
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from rl_editor.infer import load_and_process_audio, get_default_config  # get_default_config imported via module
from rl_editor.agent import Agent
from rl_editor.infer import run_inference, create_edited_audio

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


def main():
    input_audio = 'training_data/input/TheMachine-SolarCorona_synth_raw.wav'
    checkpoint = 'models/modelV1_resume_test2/final.pt'
    out_csv = Path('output/reports/resume_test2_inference.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    config = get_default_config()
    audio, sr, audio_state = load_and_process_audio(input_audio, config, max_beats=500, cache_dir=config.data.cache_dir)
    temp_env = None

    # Initialize agent
    temp_env = None
    from rl_editor.environment import AudioEditingEnvFactored
    env = AudioEditingEnvFactored(config, audio_state)
    obs, _ = env.reset()
    input_dim = len(obs)

    agent = Agent(config, input_dim=input_dim, beat_feature_dim=audio_state.beat_features.shape[1], use_auxiliary_tasks=False)
    agent.load(checkpoint)

    n_samples = 32
    base_seed = 0

    rows = []
    best_score = -1e9
    best_idx = None

    for i in range(n_samples):
        seed = base_seed + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        actions, action_names, score = run_inference(agent, config, audio_state, deterministic=False, verbose=False)

        type_counts = Counter([a.action_type.name for a in actions])
        size_counts = Counter([a.action_size.name for a in actions])
        n_actions = len(actions)
        n_segments = None

        # Compute entropies
        type_entropy = entropy_from_counts(type_counts)
        size_entropy = entropy_from_counts(size_counts)

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

        if score > best_score:
            best_score = score
            best_idx = i

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        fieldnames = ['sample','seed','score','n_actions','type_entropy_bits','size_entropy_bits','type_counts','size_counts']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Wrote', out_csv)
    print('Best sample:', best_idx, 'score', best_score)

if __name__ == '__main__':
    main()
