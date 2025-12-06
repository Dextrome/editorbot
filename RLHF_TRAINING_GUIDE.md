# RLHF Training Integration Guide

## Overview

The AI audio editor now supports full RLHF (Reinforcement Learning from Human Feedback) training with the learned reward model. This connects the pre-trained LearnedRewardModel with the PPO/DPO policy training loop.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   RLHF Training Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Training Data                                                │
│       ↓                                                        │
│  AudioEditingEnv (samples trajectories)                       │
│       ↓                                                        │
│  Trajectory Features                                           │
│       ↓  ┌──────────────────────────────────────────┐        │
│       ├─→│ Dense Rewards (environment metrics)      │        │
│       │  │ - Tempo consistency                      │        │
│       │  │ - Energy flow                            │        │
│       │  │ - Phrase completeness                    │        │
│       │  │ - Transition quality                     │        │
│       │  └──────────────────────────────────────────┘        │
│       │           Weight: 0.2                                 │
│       │           ↓                                           │
│       │    Combined Reward                                    │
│       │           ↑                                           │
│       │  Weight: 0.8                                          │
│       │  ┌──────────────────────────────────────────┐        │
│       └─→│ Learned Reward Model (v8 - trained)     │        │
│          │ - Input: Beat features (125 dim)         │        │
│          │ - Model: 3-layer transformer (4 heads)   │        │
│          │ - Output: Scalar preference score         │        │
│          └──────────────────────────────────────────┘        │
│                                                               │
│  Combined Reward                                              │
│       ↓                                                        │
│  PPO/DPO Policy Update                                        │
│       ↓                                                        │
│  Updated Policy Network                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. LearnedRewardIntegration (`learned_reward_integration.py`)

**Purpose**: Bridges trained reward model with training pipeline

**Key Classes**:
- `LearnedRewardModel`: Transformer-based reward model
- `LearnedRewardIntegration`: Integration manager
- `LearnedRewardConfig`: Configuration dataclass

**Key Methods**:
- `load_model()`: Load checkpoint from disk
- `compute_learned_reward()`: Compute scalar reward for beat sequence
- `compute_trajectory_reward()`: Combine learned + dense rewards
- `get_model_state()` / `set_model_state()`: Checkpointing

### 2. RLHFTrainer (`train_rlhf.py`)

**Purpose**: Main training loop that combines PPO with learned rewards

**Key Features**:
- Loads training data from disk
- Collects rollouts with both dense and learned rewards
- Updates policy using PPO algorithm
- Supports DPO as alternative algorithm
- Checkpoints policy and training history
- Tracks reward components throughout training

**Key Methods**:
- `train()`: Main training loop
- `_collect_rollouts_with_learned_rewards()`: Augment rollouts with learned signal
- `save_training_history()` / `load_training_history()`: Persistence

### 3. Integration with Existing Code

**Modified**: None (fully backward compatible)

**New Dependencies**:
- `learned_reward_integration.py` (new module)
- `train_rlhf.py` (new training script)

**Interfaces**:
- PPOTrainer: No changes required (uses existing methods)
- AudioEditingEnv: No changes required (provides rollouts)
- Agent: No changes required (policy network unchanged)

## Configuration

### Default Reward Weights (in LearnedRewardConfig)

```python
learned_reward_weight: float = 0.8      # 80% from preference model
dense_reward_weight: float = 0.2        # 20% from environment metrics

clamp_reward: Tuple = (-10.0, 10.0)     # Bound reward range
reward_scale: float = 1.0               # Scaling factor
reward_offset: float = 0.0              # Offset (for centering)
```

### Model Loading

Default checkpoint path: `models/reward_model_v8_long/reward_model_final.pt`

Override by passing `--reward_model` argument:

```bash
python train_rlhf.py \
    --data_dir ./training_data \
    --reward_model ./custom/path/model.pt \
    --total_steps 100000
```

## Usage

### Basic PPO Training with Learned Rewards

```bash
python train_rlhf.py \
    --data_dir ./training_data \
    --total_steps 100000 \
    --algorithm ppo \
    --save_interval 10000
```

### DPO Training with Learned Rewards

```bash
python train_rlhf.py \
    --data_dir ./training_data \
    --total_steps 100000 \
    --algorithm dpo \
    --save_interval 10000
```

### Resume from Checkpoint

```bash
python train_rlhf.py \
    --data_dir ./training_data \
    --checkpoint ./models/checkpoint_50000.pt \
    --total_steps 200000
```

### Custom Configuration

```bash
python train_rlhf.py \
    --data_dir ./training_data \
    --reward_model ./models/reward_v9.pt \
    --total_steps 500000 \
    --device cuda:0 \
    --save_interval 5000 \
    --log_level DEBUG
```

## Training Process

### Step-by-Step Flow

1. **Initialize**
   - Load training dataset
   - Load learned reward model checkpoint
   - Create PPO trainer and agent

2. **For each episode**:
   a. Sample random training audio
   b. Create AudioState for that audio
   c. Initialize environment
   d. **Collect rollouts** (nsteps=2048)
      - Environment samples trajectories
      - Computes dense rewards (tempo, energy, etc.)
   e. **Augment with learned rewards**
      - Extract beat features from trajectory
      - Feed to learned reward model
      - Get scalar preference score
   f. **Combine reward signals**
      - `combined_reward = 0.8 * learned + 0.2 * dense`
   g. **PPO update**
      - Compute advantages
      - Update policy network (5 epochs)
      - Update value network
   h. **Save checkpoint** (every 10k steps)
      - Policy weights
      - Optimizer state
      - Training history

3. **Logging**
   - Episode-level metrics (every 10 episodes)
   - Step-level tracking
   - Training history (JSON)

## Outputs

### Checkpoints

```
models/
├── policy_best.pt           # Best policy weights
├── policy_final.pt          # Final policy weights
├── checkpoint_10000.pt      # Periodic checkpoints
├── checkpoint_20000.pt
└── ...
```

### Logs

```
logs/
├── training_history.json    # Complete metrics history
└── training.log             # Text log file
```

### Training History Format

```json
{
  "episode": [0, 1, 2, ...],
  "step": [0, 2048, 4096, ...],
  "total_reward": [0.15, 0.23, 0.31, ...],
  "policy_loss": [1.2, 0.9, 0.7, ...],
  "value_loss": [0.8, 0.5, 0.3, ...],
  "learned_reward": [0.1, 0.2, 0.15, ...],
  "dense_reward": [0.05, 0.08, 0.06, ...]
}
```

## Reward Signals Explained

### Learned Reward (80% weight)

**Source**: LearnedRewardModel trained on 5,000 human preference pairs

**What it captures**: 
- Overall preference quality (learned from data)
- Edit coherence and musicality (implicit in training data)
- Preference patterns from training set

**Strengths**:
- Generalizes to new music
- Captures subtle quality aspects
- Data-driven (no hand-coded rules)

**Limitations**:
- Only as good as training data
- Requires good preference annotations

### Dense Reward (20% weight)

**Components**:
1. **Tempo Consistency** (25% of dense)
   - Penalty for BPM changes > 5 BPM
   - Range: [0, 1]

2. **Energy Flow** (25% of dense)
   - Lower is better (smoother dynamics)
   - Normalized by original energy variance
   - Range: [0, 1]

3. **Phrase Completeness** (25% of dense)
   - Penalty for aggressive edits at phrase boundaries
   - Assumes 8-beat phrases
   - Range: [0, 1]

4. **Transition Quality** (25% of dense)
   - Penalty for duration mismatch with beat grid
   - Range: [0, 1]

**Final Dense Reward**: Average of 4 components = [0, 1]

### Combined Reward

```
reward = 0.8 * learned_reward + 0.2 * dense_reward
```

**Intuition**: 
- Learned reward provides primary signal (80%)
- Dense rewards provide stability and known-good metrics (20%)

## Advanced Configuration

### Adjusting Reward Weights

Edit `LearnedRewardConfig` in `train_rlhf.py`:

```python
reward_config = LearnedRewardConfig(
    learned_reward_weight=0.7,  # Reduce to 70%
    dense_reward_weight=0.3,    # Increase to 30%
)
```

### Using DPO (Direct Preference Optimization)

```bash
python train_rlhf.py \
    --algorithm dpo \
    --data_dir ./training_data \
    --total_steps 100000
```

DPO is an alternative to PPO that directly optimizes for preferences without explicit reward model. However, since we have a trained reward model, PPO is recommended.

### Custom Reward Scaling

```python
reward_config = LearnedRewardConfig(
    reward_scale=2.0,      # Amplify reward signal by 2x
    reward_offset=-1.0,    # Center around -1
    clamp_reward=(-20.0, 20.0),  # Allow wider range
)
```

## Monitoring Training

### Real-time Monitoring

```bash
tail -f logs/training.log
```

### Plot Training History

```python
import json
import matplotlib.pyplot as plt

with open('logs/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history['step'], history['total_reward'])
plt.xlabel('Step')
plt.ylabel('Total Reward')
plt.title('Total Reward Over Time')

plt.subplot(2, 2, 2)
plt.plot(history['step'], history['policy_loss'])
plt.xlabel('Step')
plt.ylabel('Policy Loss')
plt.title('Policy Loss Over Time')

plt.subplot(2, 2, 3)
plt.plot(history['step'], history['learned_reward'], label='Learned')
plt.plot(history['step'], history['dense_reward'], label='Dense')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward Components')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history['step'], history['value_loss'])
plt.xlabel('Step')
plt.ylabel('Value Loss')
plt.title('Value Loss Over Time')

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150)
plt.show()
```

## Next Steps

1. **Prepare Training Data**
   ```bash
   # Place raw audio in training_data/train/
   python -m rl_editor.precache_stems training_data/
   ```

2. **Start RLHF Training**
   ```bash
   python train_rlhf.py \
       --data_dir ./training_data \
       --total_steps 100000 \
       --algorithm ppo
   ```

3. **Monitor Progress**
   ```bash
   tail -f logs/training.log
   ```

4. **Evaluate Policy**
   ```bash
   python -m rl_editor.evaluation \
       --checkpoint models/policy_best.pt \
       --test_dir ./test_audio/
   ```

5. **Collect Human Feedback**
   - Generate edits with policy
   - Get human preference annotations
   - Retrain reward model (optional refinement)

## Troubleshooting

### Reward Model Fails to Load

```
ERROR: Failed to load reward model
```

**Solution**:
1. Check checkpoint path is correct
2. Verify file exists: `ls models/reward_model_v8_long/reward_model_final.pt`
3. Ensure file is not corrupted (test with `test_report.py`)

### CUDA Memory Error

```
RuntimeError: CUDA out of memory
```

**Solution**:
1. Reduce batch size in config
2. Reduce trajectory length (`n_steps`)
3. Use CPU: `--device cpu`

### Policy Not Improving

**Possible causes**:
1. Learned reward model not well-calibrated
2. Reward weights need tuning
3. Training data distribution mismatch

**Solutions**:
1. Adjust `learned_reward_weight` (try 0.5)
2. Add more diverse training data
3. Train new reward model with fresh human feedback

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **RLHF**: [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- **DPO**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
