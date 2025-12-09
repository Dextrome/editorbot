"""
Hyperparameter Finder for RL Audio Editor V2
Finds optimal: n_envs, steps_per_epoch, learning_rate

Based on:
1. Hardware capacity (GPU memory, CPU cores)
2. Empirical throughput measurements
3. Learning rate range test
"""

import os
import sys
import time
import psutil
import torch
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_editor.config import get_default_config
from rl_editor.data import PairedAudioDataset
from rl_editor.environment_v2 import AudioEditingEnvV2
from rl_editor.agent import Agent

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_hardware_info():
    """Get hardware capabilities."""
    info = {
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_available': torch.cuda.is_available(),
    }
    
    if info['gpu_available']:
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['gpu_compute'] = torch.cuda.get_device_capability(0)
    
    return info


def estimate_optimal_n_envs(hw_info, use_subprocess=True):
    """Estimate optimal number of parallel environments."""
    
    if use_subprocess:
        # Subprocess mode: limited by CPU cores (each env is a process)
        # Rule: n_envs = cpu_cores - 2 (leave some for main process and OS)
        # But also limited by GPU memory for inference
        cpu_based = max(4, hw_info['cpu_cores'] - 2)
        
        if hw_info['gpu_available']:
            # Each env needs ~200MB GPU for inference
            # Leave 4GB for policy network and gradients
            gpu_based = int((hw_info['gpu_memory_gb'] - 4) / 0.2)
            optimal = min(cpu_based, gpu_based)
        else:
            optimal = cpu_based
    else:
        # Thread mode: can have more envs (GIL limits but less overhead)
        cpu_based = hw_info['cpu_threads']
        optimal = min(cpu_based, 32)  # Cap at 32 for memory
    
    # Round to power of 2 for efficiency
    powers = [4, 8, 16, 24, 32, 48, 64]
    optimal = min(powers, key=lambda x: abs(x - optimal))
    
    return optimal


def estimate_optimal_steps(n_envs, avg_episode_length=200):
    """Estimate optimal steps per epoch."""
    
    # Rule of thumb: steps should be multiple of n_envs
    # And should allow several complete episodes per epoch
    
    # Target: 8-16 complete episodes per epoch for good variance
    target_episodes = 12
    base_steps = target_episodes * avg_episode_length
    
    # Round up to multiple of n_envs * 64
    chunk = n_envs * 64
    optimal = ((base_steps + chunk - 1) // chunk) * chunk
    
    # Clamp to reasonable range
    optimal = max(1024, min(8192, optimal))

    return optimal


def _create_audio_state_from_item(item, max_beats=500):
    """Create an AudioState from a dataset item dict."""
    from rl_editor.environment_v2 import AudioState
    
    raw_data = item["raw"]
    beat_times = raw_data["beat_times"].numpy()
    beat_features = raw_data["beat_features"].numpy()
    
    # Get ground truth edit labels
    edit_labels = item.get("edit_labels")
    if edit_labels is not None:
        edit_labels = edit_labels.numpy() if hasattr(edit_labels, 'numpy') else np.array(edit_labels)
    
    # Limit to max_beats
    if len(beat_times) > max_beats:
        start_idx = np.random.randint(0, len(beat_times) - max_beats)
        beat_times = beat_times[start_idx:start_idx + max_beats]
        beat_features = beat_features[start_idx:start_idx + max_beats]
        if edit_labels is not None and len(edit_labels) > max_beats:
            edit_labels = edit_labels[start_idx:start_idx + max_beats]
        beat_times = beat_times - beat_times[0]
    
    return AudioState(
        beat_index=0,
        beat_times=beat_times,
        beat_features=beat_features,
        tempo=raw_data["tempo"].item() if hasattr(raw_data["tempo"], 'item') else raw_data["tempo"],
        target_labels=edit_labels,
    )

def measure_throughput(config, dataset, n_envs, steps, use_subprocess=False):
    """Measure actual training throughput."""
    from rl_editor.train_v2 import ParallelPPOTrainerV2
    
    logger.info(f"Measuring throughput: n_envs={n_envs}, steps={steps}")
    
    try:
        trainer = ParallelPPOTrainerV2(
            config,
            n_envs=n_envs,
            total_epochs=5,
            use_subprocess=use_subprocess,
        )
        
        # Get audio states from dataset items (dict format)
        audio_states = [_create_audio_state_from_item(dataset[i % len(dataset)]) for i in range(n_envs)]
        
        # Initialize agent using observation dimension
        from rl_editor.actions_v2 import ActionSpaceV2
        trainer.vec_env.set_audio_states(audio_states)
        obs_list, _ = trainer.vec_env.reset_all()
        input_dim = obs_list[0].shape[0]
        trainer._init_agent(input_dim, ActionSpaceV2.N_ACTIONS)
        
        # Measure rollout time
        start = time.time()
        rollout_data = trainer.collect_rollouts_parallel(audio_states, steps)
        rollout_time = time.time() - start
        
        # Measure update time
        start = time.time()
        metrics = trainer.update(rollout_data)
        update_time = time.time() - start
        
        total_steps = n_envs * steps
        throughput = total_steps / (rollout_time + update_time)
        
        # Cleanup
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'throughput': throughput,
            'rollout_time': rollout_time,
            'update_time': update_time,
            'total_steps': total_steps,
            'steps_per_second': throughput,
        }
    
    except Exception as e:
        logger.error(f"Throughput measurement failed: {e}")
        return None




def find_optimal_n_envs_empirical(config, dataset, candidates=None, steps_per_test=256):
    if candidates is None:
        candidates = [4, 8, 12, 16, 24, 32]
    
    results = []
    best_throughput = 0
    best_n_envs = candidates[0]
    
    print(f"  Testing {len(candidates)} configurations...")
    
    for n_envs in candidates:
        print(f"    n_envs={n_envs}: ", end='', flush=True)
        
        try:
            result = measure_throughput(config, dataset, n_envs, steps_per_test, use_subprocess=False)
            
            if result:
                throughput = result['throughput']
                results.append({
                    'n_envs': n_envs,
                    'throughput': throughput,
                    'rollout_time': result['rollout_time'],
                    'update_time': result['update_time'],
                })
                rt = result['rollout_time']
                ut = result['update_time']
                print(f"{throughput:.1f} steps/sec (rollout={rt:.2f}s, update={ut:.2f}s)")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_n_envs = n_envs
            else:
                print("FAILED")
                
        except Exception as e:
            print(f"ERROR: {e}")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(1)
    
    return best_n_envs, results


def find_optimal_steps_empirical(config, dataset, n_envs, candidates=None):
    if candidates is None:
        candidates = [512, 1024, 2048, 4096, 8192]
    
    results = []
    best_efficiency = 0
    best_steps = candidates[0]
    
    print(f"  Testing {len(candidates)} step sizes...")
    
    for steps in candidates:
        print(f"    steps={steps}: ", end='', flush=True)
        
        try:
            result = measure_throughput(config, dataset, n_envs, steps, use_subprocess=False)
            
            if result:
                throughput = result['throughput']
                overhead_ratio = result['update_time'] / (result['rollout_time'] + result['update_time'])
                efficiency = throughput * (1 - overhead_ratio * 0.5)
                
                results.append({
                    'steps': steps,
                    'throughput': throughput,
                    'efficiency': efficiency,
                    'overhead_ratio': overhead_ratio,
                })
                pct = overhead_ratio * 100
                print(f"{throughput:.1f} steps/sec, efficiency={efficiency:.1f}, overhead={pct:.1f}%")
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_steps = steps
            else:
                print("FAILED")
                
        except Exception as e:
            print(f"ERROR: {e}")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(1)
    
    return best_steps, results


def run_lr_range_test(config, dataset, n_envs, steps, n_iterations=20):
    """Find optimal learning rate using range test."""
    from rl_editor.train_v2 import ParallelPPOTrainerV2
    from rl_editor.actions_v2 import ActionSpaceV2
    
    logger.info(f"Running LR range test: {n_iterations} iterations")
    
    min_lr, max_lr = 1e-6, 1e-2
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), n_iterations)
    
    results = []
    
    trainer = ParallelPPOTrainerV2(
        config,
        n_envs=n_envs,
        total_epochs=n_iterations,
        use_subprocess=False,  # Faster for short test
    )
    
    audio_states = [_create_audio_state_from_item(dataset[i % len(dataset)]) for i in range(n_envs)]
    trainer.vec_env.set_audio_states(audio_states)
    obs_list, _ = trainer.vec_env.reset_all()
    input_dim = obs_list[0].shape[0]
    trainer._init_agent(input_dim, ActionSpaceV2.N_ACTIONS)
    
    for i, lr in enumerate(lrs):
        # Set LR
        for param_group in trainer.policy_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in trainer.value_optimizer.param_groups:
            param_group['lr'] = lr
        
        try:
            rollout_data = trainer.collect_rollouts_parallel(audio_states, steps // 4)
            metrics = trainer.update(rollout_data)
            
            loss = metrics.get('total_loss', float('inf'))
            results.append({'lr': lr, 'loss': loss, 'metrics': metrics})
            
            logger.info(f"LR: {lr:.2e} | Loss: {loss:.4f}")
            
            # Early stop if loss explodes
            if loss > 100000 or np.isnan(loss):
                logger.warning(f"Loss exploded at LR={lr:.2e}, stopping")
                break
                
        except Exception as e:
            logger.error(f"LR test failed at {lr:.2e}: {e}")
            break
    
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


def find_optimal_lr(lr_results):
    """Find optimal LR from range test results."""
    if not lr_results:
        return 3e-4  # Default
    
    # Find LR with steepest loss decrease (not minimum loss)
    losses = [r['loss'] for r in lr_results]
    lrs = [r['lr'] for r in lr_results]
    
    # Smooth losses
    smoothed = []
    for i in range(len(losses)):
        window = losses[max(0, i-2):i+3]
        smoothed.append(np.mean(window))
    
    # Find steepest descent
    best_idx = 0
    best_descent = 0
    for i in range(1, len(smoothed) - 1):
        descent = smoothed[i-1] - smoothed[i+1]
        if descent > best_descent:
            best_descent = descent
            best_idx = i
    
    # Return LR slightly before the steepest point (more conservative)
    optimal_idx = max(0, best_idx - 1)
    return lrs[optimal_idx]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find optimal hyperparameters for RL Audio Editor')
    parser.add_argument('--empirical', action='store_true', help='Run empirical throughput tests (slower but more accurate)')
    parser.add_argument('--quick', action='store_true', help='Quick mode - skip LR test')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RL Audio Editor - Hyperparameter Finder")
    if args.empirical:
        print("  Mode: EMPIRICAL (measuring actual throughput)")
    else:
        print("  Mode: HEURISTIC (hardware-based estimates)")
    print("=" * 60)
    
    # 1. Hardware analysis
    print("\n[1/5] Analyzing hardware...")
    hw_info = get_hardware_info()
    
    print(f"  CPU Cores: {hw_info['cpu_cores']} physical, {hw_info['cpu_threads']} logical")
    print(f"  RAM: {hw_info['ram_gb']:.1f} GB")
    if hw_info['gpu_available']:
        print(f"  GPU: {hw_info['gpu_name']}")
        print(f"  GPU Memory: {hw_info['gpu_memory_gb']:.1f} GB")
    else:
        print("  GPU: Not available (CPU training)")
    
    # Load dataset early for empirical tests
    config = get_default_config()
    dataset = None
    
    if args.empirical:
        print("\n[2/5] Loading dataset for benchmarks...")
        try:
            dataset = PairedAudioDataset(
                "./training_data",
                config,
                cache_dir=config.data.cache_dir,
                include_reference=True,
            )
            print(f"  Loaded {len(dataset)} training samples")
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            print("  Falling back to heuristic mode")
            args.empirical = False
    
    # 2. Find optimal n_envs
    if args.empirical and dataset:
        print("\n[3/5] Finding optimal n_envs (empirical)...")
        optimal_n_envs, n_envs_results = find_optimal_n_envs_empirical(
            config, dataset, 
            candidates=[4, 8, 12, 16, 20, 24],
            steps_per_test=128
        )
        print(f"  Best n_envs: {optimal_n_envs}")
    else:
        print("\n[2/5] Estimating optimal environments (heuristic)...")
        use_subprocess = True
        optimal_n_envs = estimate_optimal_n_envs(hw_info, use_subprocess)
        print(f"  Recommended n_envs: {optimal_n_envs}")
    
    # 3. Find optimal steps
    if args.empirical and dataset:
        print("\n[4/5] Finding optimal steps (empirical)...")
        optimal_steps, steps_results = find_optimal_steps_empirical(
            config, dataset, optimal_n_envs,
            candidates=[512, 1024, 2048, 4096]
        )
        print(f"  Best steps: {optimal_steps}")
    else:
        print("\n[3/5] Estimating optimal steps per epoch (heuristic)...")
        optimal_steps = estimate_optimal_steps(optimal_n_envs)
        print(f"  Recommended steps: {optimal_steps}")
    
    # 4. Run LR test
    if args.quick:
        print("\n  Skipping LR test (quick mode)")
        optimal_lr = 3e-4
        print(f"  Using default learning rate: {optimal_lr:.2e}")
    else:
        step_num = "[5/5]" if args.empirical else "[4/4]"
        print(f"\n{step_num} Running learning rate range test...")
        
        # Load dataset if not already loaded
        if dataset is None:
            try:
                dataset = PairedAudioDataset(
                    "./training_data",
                    config,
                    cache_dir=config.data.cache_dir,
                    include_reference=True,
                )
                print(f"  Loaded {len(dataset)} training samples")
            except Exception as e:
                print(f"  Could not load dataset: {e}")
                dataset = None
        
        if dataset is not None:
            try:
        
                # Run LR range test
                lr_results = run_lr_range_test(
                    config, dataset,
                    n_envs=min(8, optimal_n_envs),
                    steps=512,
                    n_iterations=15
                )
                
                optimal_lr = find_optimal_lr(lr_results)
                print(f"  Recommended learning rate: {optimal_lr:.2e}")
                
            except Exception as e:
                print(f"  Could not run LR test: {e}")
                optimal_lr = 3e-4
                print(f"  Using default learning rate: {optimal_lr:.2e}")
        else:
            optimal_lr = 3e-4
            print(f"  Using default learning rate: {optimal_lr:.2e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RECOMMENDED HYPERPARAMETERS")
    print("=" * 60)
    print(f"  --n_envs {optimal_n_envs}")
    print(f"  --steps {optimal_steps}")
    print(f"  --lr {optimal_lr:.2e}")
    print(f"  --subprocess  (recommended for stability)")
    print()
    print("Example command:")
    print(f"  python -m rl_editor.train_v2 --n_envs {optimal_n_envs} --steps {optimal_steps} --lr {optimal_lr:.2e} --subprocess")
    print("=" * 60)
    
    return {
        'n_envs': optimal_n_envs,
        'steps': optimal_steps,
        'lr': optimal_lr,
    }


if __name__ == "__main__":
    main()
