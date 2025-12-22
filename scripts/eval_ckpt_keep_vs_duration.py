#!/usr/bin/env python
"""Evaluate a checkpoint: report env.keep_ratio and edited-duration percent.

Usage:
  python scripts/eval_ckpt_keep_vs_duration.py --checkpoint models/hpo/hpo_trial_5_1766318427/best.pt --input training_data/input/Zappa-OrangeCounty_synth_raw.wav --max-beats 0 --n-samples 4
"""
import argparse
from pathlib import Path
import numpy as np
import logging
import torch

from rl_editor.config import get_default_config
from rl_editor.infer import load_and_process_audio, run_inference, create_edited_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--max-beats', type=int, default=0)
    p.add_argument('--n-samples', type=int, default=4)
    p.add_argument('--deterministic', action='store_true')
    args = p.parse_args()

    config = get_default_config()
    audio, sr, audio_state = load_and_process_audio(args.input, config, max_beats=args.max_beats, cache_dir=config.data.cache_dir)

    # Initialize agent like `rl_editor.infer` does (use observation dim)
    from rl_editor.agent import Agent
    from rl_editor.environment import AudioEditingEnvFactored
    temp_env = AudioEditingEnvFactored(config, audio_state)
    obs, _ = temp_env.reset()
    input_dim = len(obs)
    agent = Agent(config, input_dim=input_dim, beat_feature_dim=audio_state.beat_features.shape[1], use_auxiliary_tasks=False)
    # load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    agent.load(str(args.checkpoint))

    # Run multiple samples and report both metrics
    for i in range(max(1, args.n_samples)):
        actions, action_names, total_reward, aux_preds, final_keep_ratio = run_inference(agent, config, audio_state, deterministic=args.deterministic, verbose=False, collect_aux=False)
        edited = create_edited_audio(audio, sr, audio_state.beat_times, actions)
        orig_dur = len(audio) / sr
        edited_dur = len(edited) / sr
        edited_pct = 100.0 * edited_dur / orig_dur
        logger.info(f"Sample {i}: reward={total_reward:.4f} per-beat-keep_ratio={final_keep_ratio:.3f} edited_pct={edited_pct:.1f}% actions={len(actions)}")

if __name__ == '__main__':
    main()
