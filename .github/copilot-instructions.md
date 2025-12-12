<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Audio Editor Project (V2)

This is a Python project for an AI-powered audio editor that processes raw music recordings and transforms them into polished, listenable songs using reinforcement learning.

## Current Architecture: V2 Section-Level RL

The V2 system uses **section-level actions** (phrase/bar/beat) with **episode-end rewards** for true multi-step credit assignment.

```
Input Audio  [RL Agent: Section-level actions]  Output
                    ↓
            Actions: KEEP_PHRASE, CUT_BAR, LOOP_BEAT, REORDER, JUMP_BACK, etc.
                    ↓
            Reward: Episode-end audio quality metrics (Monte Carlo)
```

### V2 Action Space (16 actions)
- **Beat-level**: KEEP_BEAT, CUT_BEAT
- **Bar-level** (4 beats): KEEP_BAR, CUT_BAR
- **Phrase-level** (8 beats): KEEP_PHRASE, CUT_PHRASE
- **Looping**: LOOP_BEAT, LOOP_BAR, LOOP_PHRASE
- **Reordering**: REORDER_BEAT, REORDER_BAR, REORDER_PHRASE
- **Navigation**: JUMP_BACK_4, JUMP_BACK_8
- **Transitions**: MARK_SOFT_TRANSITION, MARK_HARD_CUT

### State Representation
- **Current position** in track (beat index)
- **Audio features** (121 dimensions per beat):
  - Spectral features (onset, centroid, rolloff, bandwidth, flatness, contrast)
  - Timbral features (13 MFCCs + deltas)
  - Harmonic features (12 chroma bins)
  - Rhythmic features (tempo, beat phase)
  - Stem features (drums, bass, vocals, other - 4 stems × 3 features)
- **Edit history** - What's been kept/cut/looped so far
- **Target constraints** - Duration remaining, target keep ratio (~45%)
- **Global context** - Via NATTEN local attention encoder

### V2 Reward Design (Episode-End)
Monte Carlo rewards computed at episode end:

**Penalty Components** (discourage bad behavior):
- **Keep ratio penalty** - Must cut at least 30%, can't cut more than 80%
- **Loop budget** - Base allowance of 2 loops + 1 per 3 cuts (prevent loop spam)
- **Excessive jumps** - Only penalize if >25-30% of actions are jumps
- **Action diversity** - Mild penalties for using <4 action types

**Reward Components** (encourage good behavior):
- **Cut ratio score** - Reward for hitting target cut ratio (~45%)
- **Section coherence** - Reward for keeping consecutive beats together
- **Phrase alignment** - Reward for cutting at phrase boundaries
- **Edit structure** - Reward for clean edit patterns
- **Audio quality** - Click detection, energy consistency
- **Cut quality** - Reward for cutting at phrase boundaries
- **Flow score** - Smooth energy transitions between kept sections
- **Action diversity bonus** - Small bonus for using varied actions
- **Reordering quality** - Reward for effective use of jumps/loops (20 points max)

**Key Insight**: Balance penalties and rewards carefully. Too many penalties cause the model to collapse to "safe" actions (e.g., CUT_BAR spam). Add positive rewards for creative actions you want to encourage.

### Training Approach
- **Monte Carlo returns** - Episode-end rewards only, mean-baseline advantages
- **PPO with high entropy** - entropy_coeff=0.75 to encourage exploration
- **Auxiliary tasks** - Energy prediction, phrase boundary detection (multi-task learning)
- **NATTEN encoder** - Neighborhood attention for local context with global pooling
- **Gradient accumulation** - For larger effective batch sizes

## Project Structure
```
rl_editor/
  config.py          - All hyperparameters (dataclass pattern)
  train_v2.py        - V2 PPO training with parallel envs
  infer_v2.py        - V2 inference script
  environment_v2.py  - V2 Gym environment (section actions, episode rewards)
  actions_v2.py      - V2 action space (16 actions)
  agent.py           - Policy/Value networks with NATTEN encoder
  state.py           - State representation
  reward.py          - Reward calculation utilities
  data.py            - PairedAudioDataset, feature extraction
  features.py        - BeatFeatureExtractor (121-dim features)
  augmentation.py    - Audio augmentation (pitch, noise, gain, EQ)
  auxiliary_tasks.py - Multi-task learning targets
  cache.py           - Feature caching system
  utils.py           - Audio processing utilities
  infer_utils.py     - Inference utilities (load_and_process_audio)
  logging_utils.py   - TensorBoard logging
  subprocess_vec_env.py - Subprocess-based parallel environments
training_data/       - Raw and edited audio pairs
models_v4*/          - Saved checkpoints
shared/              - Demucs wrapper for stem separation
```

## Key Commands
```bash
# Training
python -m rl_editor.train_v2 --save_dir models_v4d --epochs 30000 --steps 512 --n_envs 16 --lr 5e-5 --subprocess

# Inference
python -m rl_editor.infer_v2 "input.wav" --checkpoint "models_v4c/checkpoint.pt" --output "output.wav"

# Resume from checkpoint
python -m rl_editor.train_v2 --checkpoint models_v4c/checkpoint_v2_epoch_1000.pt --save_dir models_v4c
```

## Key Technologies
- Python 3.10+
- PyTorch with CUDA
- librosa - Audio analysis
- NATTEN - Neighborhood attention
- Demucs - Stem separation
- gymnasium - RL environment interface

## Development Guidelines
- **ALWAYS use CUDA** if available (never fall back to CPU silently)
- **V2 only** - All new development should use V2 architecture
- **Monte Carlo rewards** - Minimize step rewards, maximize episode-end rewards
- **High entropy** - Keep entropy_coeff high (0.5-0.75) until policy stabilizes
- **Action masking** - Prevent invalid actions based on current state
- **Feature caching** - Always cache features and stems for faster iteration
- **Gradient accumulation** - Use for larger effective batch sizes
- **Mixed precision** - Disabled due to large value losses causing FP16 overflow
- **Subprocess parallelism** - Use `--subprocess` flag for true multiprocessing on Windows

### Model Architecture Constraints
- **policy_hidden_dim must be divisible by natten_n_heads** (e.g., 512 with 8 heads)
- **NATTEN kernel_size must be odd** (e.g., 31, 33)

### PPO Hyperparameter Tips
- **clip_ratio**: 0.2-0.5 (higher = more exploration)
- **entropy_coeff**: Start high (0.5-0.75), decay over training
- **value_loss_coeff**: Keep low (0.05-0.1) to balance with policy loss
- **target_kl**: 0.01-0.05 for early stopping
- **gradient_accumulation_steps**: 2-8 for larger effective batch

### Debugging Tips
- Log reward breakdowns every N epochs to diagnose issues
- Check action distribution - should be diverse, not collapsed
- Monitor policy loss - near-zero means policy is too confident
- Watch for loop/jump spam - indicates reward exploitation
- **Watch for action collapse** - If one action dominates (>90%), reward structure needs rebalancing
- Focus on performance gains over time, not just raw reward numbers
- Use epoch timing breakdown to identify bottlenecks (data_sample, rollout, update, scheduler)
- **Run inference tests** periodically to see actual action distribution on real audio

### Known Issues & Solutions

#### Mixed Precision NaNs
- **Problem**: FP16 overflow with large value losses (>65504)
- **Solution**: Keep `use_mixed_precision=False` in config, or:
  - Clip returns to [-1000, 1000] before value loss
  - Reduce `trajectory_reward_scale` 
  - Use normalized returns for value targets

#### GPU Underutilization
- **Symptoms**: Low GPU usage (30-40%), spiky utilization pattern
- **Causes**: CPU-bound environment stepping, IPC overhead
- **Solutions**:
  - Use `--subprocess` flag for true multiprocessing (subprocesses don't do much though - needs improvement)
  - Increase `batch_size` (1024+) for better GPU saturation
  - Ensure all features are pre-cached (check `feature_cache/` directory)
  - Profile with epoch timing breakdown to find bottlenecks

#### Reward Normalization
- **Value network** should be trained on unnormalized (or lightly clipped) returns
- **Advantages** should be normalized (mean=0, std=1) for stable policy gradients
- Use Welford's online algorithm for running reward statistics

#### Subprocess Environments
- Workers may show 0% CPU if waiting for main process (IPC bound)
- Main process batches actions → sends to workers → waits for results
- Bottleneck is often in the main process model inference, not env stepping

#### Action Collapse / CUT_BAR Spam
- **Symptoms**: Model uses one action type 90%+ of the time (typically CUT_BAR)
- **Causes**: Penalties too harsh for creative actions, no positive reward for reordering
- **Solutions**:
  - Relax loop/jump penalties (allow more before penalizing)
  - Add **positive rewards** for using jumps/loops effectively (Component 8: Reordering Quality)
  - Lower diversity penalties (-15 instead of -25 for low diversity)
  - Ensure reordering actions have a path to positive reward, not just avoidance of penalty

### Performance Tuning
- **Batch size**: 512-1024 for RTX 4070 Ti (12GB VRAM)
- **n_envs**: 16-24 parallel environments
- **n_steps**: 512 per epoch (total samples = n_envs × n_steps)
- **Learning rate**: 4e-5 to 5e-5 with cosine decay
- **Target episode reward**: Look for consistent improvement over epochs, not absolute values

### Inference Testing
```bash
# Test action distribution on real audio
python -m rl_editor.infer_v2 "test_input.wav" --checkpoint "models_v4c/checkpoint.pt" --output "test_output.wav"
```
- Check action distribution in output - should see mix of actions, not just CUT_BAR
- Good distribution: 60-70% cutting actions, 15-25% keeping, 5-15% reordering (jumps/loops)
- Bad distribution: >90% one action type = reward structure problem
