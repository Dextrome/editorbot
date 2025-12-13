<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Audio Editor Project

This is a Python project for an AI-powered audio editor that processes raw music recordings and transforms them into polished, listenable songs using reinforcement learning.

## Current Architecture: Factored Action Space

The system uses a **3-head factored action space** for efficient policy learning:
- **Type head** (18 outputs): What action to take (KEEP, CUT, LOOP, FADE, GAIN, etc.)
- **Size head** (5 outputs): How many beats (BEAT, BAR, PHRASE, TWO_BARS, TWO_PHRASES)
- **Amount head** (5 outputs): Intensity/direction (-3dB to +3dB, back 4/8 beats, etc.)

This gives **450 possible action combinations from just 28 network outputs** (vs 450 for discrete).

```
Input Audio  [RL Agent: Factored actions]  Output
                    ↓
            Policy outputs: (type_head, size_head, amount_head)
                    ↓
            Combined: (KEEP, PHRASE, NEUTRAL) → Keep 8 beats
            Combined: (GAIN, BAR, +3dB) → Boost 4 beats by 3dB
            Combined: (JUMP_BACK, BEAT, -8beats) → Jump back 8 beats
                    ↓
            Reward: Episode-end audio quality metrics (Monte Carlo)
```

### Factored Action Space (18 types × 5 sizes × 5 amounts)
**Action Types** (18):
- **Core**: KEEP, CUT, LOOP, REORDER
- **Navigation**: JUMP_BACK, SKIP
- **Volume**: FADE_IN, FADE_OUT, GAIN
- **Time**: DOUBLE_TIME, HALF_TIME, REVERSE
- **EQ**: EQ_LOW, EQ_HIGH
- **Effects**: DISTORTION, REVERB
- **Structure**: REPEAT_PREV, SWAP_NEXT

**Action Sizes** (5): BEAT (1), BAR (4), PHRASE (8), TWO_BARS (8), TWO_PHRASES (16)

**Action Amounts** (5): NEG_LARGE, NEG_SMALL, NEUTRAL, POS_SMALL, POS_LARGE

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

### Reward Design (Episode-End)
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
- **Flow score** - Smooth energy transitions between kept sections
- **Action diversity bonus** - Small bonus for using varied actions
- **Reordering quality** - Reward for effective use of jumps/loops

### Training Approach
- **Monte Carlo returns** - Episode-end rewards only, mean-baseline advantages
- **PPO with high entropy** - entropy_coeff=0.75 to encourage exploration
- **3-head policy** - Separate log_probs combined: log P(type) + log P(size|type) + log P(amount|type)
- **Auxiliary tasks** - Energy prediction, phrase boundary detection (multi-task learning)
- **NATTEN encoder** - Neighborhood attention for local context with global pooling
- **Gradient accumulation** - For larger effective batch sizes

## Project Structure
```
rl_editor/
  config.py            - All hyperparameters (dataclass pattern)
  train.py             - PPO training with parallel envs (threading + subprocess)
  infer.py             - Inference script
  environment.py       - Gym environment (factored actions, episode rewards)
  actions.py           - Factored action space (18 types × 5 sizes × 5 amounts)
  agent.py             - Agent, PolicyNetwork (3-head), ValueNetwork, HybridNATTENEncoder
  state.py             - State representation
  reward.py            - Reward calculation utilities
  data.py              - PairedAudioDataset, feature extraction
  features.py          - BeatFeatureExtractor (121-dim features)
  augmentation.py      - Audio augmentation (pitch, noise, gain, EQ)
  auxiliary_tasks.py   - Multi-task learning targets
  cache.py             - Feature caching system
  utils.py             - Audio processing utilities
  infer_utils.py       - Inference utilities
  logging_utils.py     - TensorBoard logging
  subprocess_vec_env.py - Subprocess-based parallel environments
training_data/         - Raw and edited audio pairs
models*/               - Saved checkpoints
shared/                - Demucs wrapper for stem separation
```

## Key Commands
```bash
# Training (threading mode - default)
python -m rl_editor.train --save_dir models --epochs 30000 --steps 512 --n_envs 16 --lr 5e-5

# Training (subprocess mode - true multiprocessing)
python -m rl_editor.train --save_dir models --epochs 30000 --steps 512 --n_envs 16 --lr 5e-5 --subprocess

# Inference
python -m rl_editor.infer "input.wav" --checkpoint "models/best.pt" --output "output.wav"

# Resume from checkpoint
python -m rl_editor.train --checkpoint models/best.pt --save_dir models
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
- **Factored actions** - All development uses factored 3-head action space
- **Monte Carlo rewards** - Minimize step rewards, maximize episode-end rewards
- **High entropy** - Keep entropy_coeff high (0.5-0.75) until policy stabilizes
- **Action masking** - Prevent invalid actions based on current state (3 masks: type, size, amount)
- **Feature caching** - Always cache features and stems for faster iteration
- **Gradient accumulation** - Use for larger effective batch sizes
- **Mixed precision** - Disabled due to large value losses causing FP16 overflow
- **Subprocess parallelism** - Use `--subprocess` flag for true multiprocessing on Windows

### Model Architecture Constraints
- **policy_hidden_dim must be divisible by natten_n_heads** (e.g., 512 with 8 heads)
- **NATTEN kernel_size must be odd** (e.g., 31, 33)
- **3-head policy**: type_head (18), size_head (5, conditioned on type), amount_head (5, conditioned on type)

### PPO Hyperparameter Tips
- **clip_ratio**: 0.2-0.5 (higher = more exploration)
- **entropy_coeff**: Start high (0.5-0.75), decay over training
- **value_loss_coeff**: Keep low (0.05-0.1) to balance with policy loss
- **target_kl**: 0.01-0.05 for early stopping
- **gradient_accumulation_steps**: 2-8 for larger effective batch

### Debugging Tips
- Log reward breakdowns every N epochs to diagnose issues
- Check action distribution - should be diverse across types, sizes, and amounts
- Monitor policy loss - near-zero means policy is too confident
- Watch for type collapse (e.g., only CUT) - indicates reward structure problem
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
  - Use `--subprocess` flag for true multiprocessing
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

#### Action Type Collapse
- **Symptoms**: Model uses one action type 90%+ of the time (typically CUT)
- **Causes**: Penalties too harsh for creative actions, no positive reward for variety
- **Solutions**:
  - Relax loop/jump penalties (allow more before penalizing)
  - Add **positive rewards** for using varied action types effectively
  - Lower diversity penalties
  - Ensure creative actions have a path to positive reward

### Performance Tuning
- **Batch size**: 512-1024 for RTX 4070 Ti (12GB VRAM)
- **n_envs**: 16-24 parallel environments
- **n_steps**: 512 per epoch (total samples = n_envs × n_steps)
- **Learning rate**: 4e-5 to 5e-5 with cosine decay
- **Target episode reward**: Look for consistent improvement over epochs, not absolute values

### Inference Testing
```bash
# Test action distribution on real audio
python -m rl_editor.infer "test_input.wav" --checkpoint "models/best.pt" --output "test_output.wav"
```
- Check type/size/amount distribution in output
- Good distribution: Mix of types (60-70% keep/cut, 15-25% loop/effects, 5-15% navigation)
- Bad distribution: >90% one action type = reward structure problem
