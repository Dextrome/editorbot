"""Optuna-based hyperparameter optimization for RL editor.

This script runs short training jobs via `scripts/train_with_config.py` with
suggested hyperparameters, then evaluates the produced checkpoint by running
`rl_editor.infer` on a small validation audio and parsing the edited-duration
and per-beat keep ratio. The objective returned to Optuna is a combined score
you can customize.

Note: install optuna in your environment (`pip install optuna`).
"""
import optuna
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
import argparse
import re
import os
import sys
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False
from rl_editor.config import get_default_config
from rl_editor.config import get_default_config


def run_train_trial(save_dir, data_dir, lr, clip_ratio, entropy_coeff, value_loss_coeff, max_grad_norm, batch_size, grad_acc_steps, n_envs, steps, subprocess_mode, epochs=1):
    cmd = [
        sys.executable, "scripts/train_with_config.py",
        "--data_dir", data_dir,
        "--save_dir", str(save_dir),
        "--epochs", str(epochs),
        "--n_envs", str(n_envs),
        "--steps", str(steps),
        "--lr", str(lr),
        "--clip_ratio", str(clip_ratio),
        "--entropy_coeff", str(entropy_coeff),
        "--value_loss_coeff", str(value_loss_coeff),
        "--max_grad_norm", str(max_grad_norm),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_acc_steps),
    ]
    if subprocess_mode:
        cmd.append("--subprocess")

    # Run training; ensure subprocess can import project packages by setting PYTHONPATH
    env = dict(**subprocess.os.environ)
    # Prefer existing PYTHONPATH but ensure current dir is first so `rl_editor` imports work
    env_py = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = ('.' + (os.pathsep + env_py if env_py else ''))

    # Run training; capture output for debugging
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60*30, env=env)
    out = proc.stdout

    # Save stdout to save_dir for later inspection
    try:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        logpath = save_dir_path / "train_stdout.txt"
        with open(logpath, 'w', encoding='utf-8') as f:
            f.write(out)
    except Exception:
        # Best-effort only
        pass

    return out


def run_infer_and_parse(ckpt_path, input_audio, deterministic=False, n_samples=4, max_beats: int = 0):
    cmd = [
        sys.executable, "-m", "rl_editor.infer",
        input_audio,
        "--checkpoint", str(ckpt_path),
        "--n-samples", str(n_samples),
    ]
    if deterministic:
        cmd.append("--deterministic")
    if max_beats and int(max_beats) > 0:
        cmd.extend(["--max-beats", str(int(max_beats))])

    # Ensure PYTHONPATH so subprocess can import project modules
    env = dict(**subprocess.os.environ)
    env_py = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = ('.' + (os.pathsep + env_py if env_py else ''))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300, env=env)
    out = proc.stdout

    # Parse edited-duration %, per-beat keep ratio and cumulative reward from output
    edited_pct = None
    best_keep_ratio = None
    best_reward = None

    # Parse per-sample lines like: "Sample 0: cumulative reward=123.45 per-beat-keep_ratio=0.512"
    sample_re = re.compile(r"Sample\s+(\d+):\s+cumulative reward=([-0-9\.]+)(?:\s+per-beat-keep_ratio=([0-9\.]+))?")
    for line in out.splitlines():
        # Edited percent
        if "Audio:" in line and "(" in line and "%" in line:
            m = re.search(r"\(([-0-9\.]+)%\)", line)
            if m:
                try:
                    edited_pct = float(m.group(1))
                except Exception:
                    pass

        # Sample cumulative reward lines
        m = sample_re.search(line)
        if m:
            try:
                r = float(m.group(2))
            except Exception:
                r = None
            kr = None
            if m.group(3):
                try:
                    kr = float(m.group(3))
                except Exception:
                    kr = None

            if r is not None:
                if best_reward is None or r > best_reward:
                    best_reward = r
                    best_keep_ratio = kr

    # Fallback: parse saved chosen sample line for keep_ratio if not present
    if best_keep_ratio is None:
        m2 = re.search(r"per-beat-keep_ratio=([0-9\.]+)", out)
        if m2:
            try:
                best_keep_ratio = float(m2.group(1))
            except Exception:
                best_keep_ratio = None

    return edited_pct, best_keep_ratio, best_reward, out


def objective(trial, args):
    # Expanded search space
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    clip_ratio = trial.suggest_float("clip_ratio", 0.05, 0.5)
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.0, 1.0)
    value_loss_coeff = trial.suggest_float("value_loss_coeff", 0.01, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    grad_acc = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8])
    # Train args sampled or fixed from CLI defaults
    n_envs = trial.suggest_categorical("n_envs", [2, 4, 8, 16, args.n_envs])
    steps = trial.suggest_categorical("steps", [64, 128, 256, 512, args.steps])
    subprocess_mode = args.subprocess

    # Use per-trial save dir
    save_dir = Path(args.base_save_dir) / f"hpo_trial_{trial.number}_{int(time.time())}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run training (short)
    try:
        train_log = run_train_trial(save_dir, args.data_dir, lr, clip_ratio, entropy_coeff, value_loss_coeff, max_grad_norm, batch_size, grad_acc, n_envs, steps, subprocess_mode, epochs=args.epochs)
    except subprocess.TimeoutExpired as e:
        trial.report(-1.0, step=0)
        raise

    # Find checkpoint (choose best.pt or last checkpoint)
    ckpt = None
    ckpt_best = save_dir / "best.pt"
    if ckpt_best.exists():
        ckpt = ckpt_best
    else:
        # fallback to any checkpoint_epoch_*.pt
        cands = sorted(save_dir.glob("checkpoint_epoch_*.pt"))
        if cands:
            ckpt = cands[-1]

    if ckpt is None:
        # Failed trial
        trial.report(-1.0, step=0)
        return -1.0

    # Run inference on a validation audio (single file) and parse
    # Use configured eval truncation if available
    try:
        cfg = get_default_config()
        eval_max_beats = int(getattr(cfg.training, 'eval_max_beats', 0))
    except Exception:
        eval_max_beats = 0

    edited_pct, keep_ratio, cum_reward, infer_out = run_infer_and_parse(
        ckpt, args.val_audio, deterministic=False, n_samples=4, max_beats=eval_max_beats
    )

    # Save inference stdout for debugging
    try:
        infer_log_path = Path(save_dir) / "infer_stdout.txt"
        with open(infer_log_path, 'w', encoding='utf-8') as f:
            f.write(infer_out or "")
    except Exception:
        pass

    # Also run a deterministic evaluation pass (single sample) using same truncation
    try:
        det_edited_pct, det_keep_ratio, det_cum_reward, det_out = run_infer_and_parse(
            ckpt, args.val_audio, deterministic=True, n_samples=1, max_beats=eval_max_beats
        )
        det_log_path = Path(save_dir) / "infer_deterministic_stdout.txt"
        with open(det_log_path, 'w', encoding='utf-8') as f:
            f.write(det_out or "")
    except Exception as e:
        det_edited_pct = None
        det_keep_ratio = None
        det_cum_reward = None
        try:
            with open(Path(save_dir) / "infer_deterministic_stdout.txt", 'w', encoding='utf-8') as f:
                f.write(f"deterministic eval failed: {e}\n")
        except Exception:
            pass

    # Write TensorBoard scalars for both stochastic and deterministic eval results
    try:
        tb_dir = Path(save_dir) / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        if edited_pct is not None:
            writer.add_scalar('eval/edited_pct', float(edited_pct), 0)
        if keep_ratio is not None:
            writer.add_scalar('eval/keep_ratio', float(keep_ratio), 0)
        if cum_reward is not None:
            writer.add_scalar('eval/cumulative_reward', float(cum_reward), 0)

        if det_edited_pct is not None:
            writer.add_scalar('eval/edited_pct_det', float(det_edited_pct), 0)
        if det_keep_ratio is not None:
            writer.add_scalar('eval/keep_ratio_det', float(det_keep_ratio), 0)
        if det_cum_reward is not None:
            writer.add_scalar('eval/cumulative_reward_det', float(det_cum_reward), 0)

        writer.close()
    except Exception:
        pass

    # Optional: log to W&B if available and enabled in config
    try:
        cfg = get_default_config()
        if HAS_WANDB and getattr(cfg.training, 'use_wandb', False):
            run = wandb.init(project=cfg.training.wandb_project, name=Path(save_dir).name, dir=str(save_dir), reinit=True)
            log_dict = {}
            if edited_pct is not None:
                log_dict['eval/edited_pct'] = float(edited_pct)
            if keep_ratio is not None:
                log_dict['eval/keep_ratio'] = float(keep_ratio)
            if cum_reward is not None:
                log_dict['eval/cumulative_reward'] = float(cum_reward)
            if det_edited_pct is not None:
                log_dict['eval/edited_pct_det'] = float(det_edited_pct)
            if det_keep_ratio is not None:
                log_dict['eval/keep_ratio_det'] = float(det_keep_ratio)
            if det_cum_reward is not None:
                log_dict['eval/cumulative_reward_det'] = float(det_cum_reward)
            if log_dict:
                wandb.log(log_dict)
            wandb.finish()
    except Exception:
        pass

    # Compose score: use deterministic evaluation if available (more robust),
    # otherwise fall back to stochastic best-of-N.
    primary_reward = None
    primary_keep = None
    if det_cum_reward is not None:
        primary_reward = det_cum_reward
        primary_keep = det_keep_ratio
    else:
        primary_reward = cum_reward
        primary_keep = keep_ratio

    if primary_reward is None:
        # failed inference or parse
        trial.set_user_attr("parsed_infer_output", infer_out)
        trial.set_user_attr("edited_pct", edited_pct)
        trial.set_user_attr("keep_ratio", keep_ratio)
        trial.set_user_attr("checkpoint", str(ckpt))
        return -1.0

    # Penalty for keep_ratio outside target range
    lower, upper = 0.35, 0.6
    penalty = 0.0
    if primary_keep is None:
        penalty = 100.0
    else:
        if primary_keep < lower:
            penalty = 200.0 * (lower - primary_keep)
        elif primary_keep > upper:
            penalty = 200.0 * (primary_keep - upper)

    score = float(primary_reward) - penalty

    # Report trial attributes for both stochastic and deterministic results
    trial.set_user_attr("edited_pct", edited_pct)
    trial.set_user_attr("keep_ratio", keep_ratio)
    trial.set_user_attr("cumulative_reward", cum_reward)
    trial.set_user_attr("edited_pct_det", det_edited_pct)
    trial.set_user_attr("keep_ratio_det", det_keep_ratio)
    trial.set_user_attr("cumulative_reward_det", det_cum_reward)
    trial.set_user_attr("checkpoint", str(ckpt))

    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--val_audio", type=str, default="training_data/input/Zappa-OrangeCounty_synth_raw.wav")
    parser.add_argument("--base_save_dir", type=str, default="models/hpo")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--subprocess", action="store_true")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--n_envs", type=int, default=4)
    args = parser.parse_args()

    Path(args.base_save_dir).mkdir(parents=True, exist_ok=True)

    # Use ASHA pruner to stop unpromising trials early (requires intermediate reports).
    # Note: pruning is most effective when objective reports intermediate results during training.
    # Use SuccessiveHalvingPruner (ASHA-like behavior) — correct Optuna class name
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    # Use lambda to capture args
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials, n_jobs=args.n_jobs)

    print("Best trial:", study.best_trial.params, "value=", study.best_value)


if __name__ == "__main__":
    main()
