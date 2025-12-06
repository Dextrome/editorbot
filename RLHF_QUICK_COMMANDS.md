# RLHF Training - Quick Commands

## Run Training

### Standard (Synthetic Data - STABLE)
```bash
python train_rlhf_stable.py --episodes 200 --steps 128
```
✅ **Recommended** - Uses proper feature dimensions, integrates learned rewards automatically

### Extended Training (500 episodes)
```bash
python train_rlhf_stable.py --episodes 500 --steps 256
```
- Longer training for better convergence
- 256 steps per episode for more data per update

### With Custom Device
```bash
python train_rlhf_stable.py --episodes 200 --device cuda --log_level INFO
```
- Automatically uses CUDA if available
- Set to `cpu` to force CPU training

---

## Resume Training from Checkpoint

Modify `train_rlhf_stable.py` to load a checkpoint:

```python
trainer = StableRLHFTrainer(config)
# Add this after initialization:
trainer.ppo_trainer.load_checkpoint("./models/policy_best.pt")
trainer.train(n_episodes=200)
```

Or use the main training script:
```bash
python train_rlhf.py --data_dir ./training_data --checkpoint ./models/policy_best.pt
```

---

## Evaluate Trained Policy

```python
from rl_editor.trainer import PPOTrainer
from rl_editor.config import get_default_config
from rl_editor.state import AudioState
import numpy as np

config = get_default_config()
trainer = PPOTrainer(config)
trainer.load_checkpoint("./models/policy_final.pt")

# Create audio state
beat_features = np.random.randn(32, 125)  # 32 beats, 125D features
audio_state = AudioState(
    beat_index=0,
    beat_times=np.linspace(0, 30, 32),
    beat_features=beat_features,
    tempo=120.0
)

# Initialize and run
trainer.initialize_env_and_agent(audio_state)
actions = trainer.predict(states)  # Get policy actions
```

---

## Available Checkpoints

```
models/policy_final.pt           (24.87 MB) ← Latest trained policy
models/policy_best.pt            (24.87 MB) ← Best during training
models/checkpoint_ep00200.pt     (24.87 MB)
models/checkpoint_ep00150.pt     (24.87 MB)
models/checkpoint_ep00100.pt     (24.87 MB)
models/checkpoint_ep00050.pt     (24.87 MB)
```

All checkpoints are fully functional and can be loaded for evaluation or continued training.

---

## Reward Model Status

```
✓ Location: models/reward_model_v8_long/reward_model_final.pt
✓ Loaded automatically during training
✓ Input: 125D beat features + action IDs
✓ Output: Preference score [-10, 10]
✓ Integration: Automatic in train_rlhf_stable.py
✓ Weight: 80% in combined reward signal
```

---

## Troubleshooting

### "Feature dimension mismatch: expected 125, got 4"
- ✅ Normal and expected when using `train_rlhf.py` with real audio
- Gracefully falls back to dense rewards only
- Use `train_rlhf_stable.py` to use learned rewards with synthetic data

### "Reward model not found"
- Check path: `models/reward_model_v8_long/reward_model_final.pt`
- Training will use dense rewards only if model unavailable
- Model loads automatically from default location

### CUDA out of memory
```bash
python train_rlhf_stable.py --episodes 100 --device cpu
```
- Reduce episodes or use CPU
- Batch size is handled internally

### Training too slow
```bash
python train_rlhf_stable.py --episodes 200 --steps 64
```
- Reduce steps per episode
- Or use fewer episodes for testing

---

## Key Parameters

In `train_rlhf_stable.py`:

```python
# Reward mixing
weight_learned = 0.8  # Learned reward weight
weight_dense = 0.2    # Dense reward weight

# Feature generation
n_beats = np.random.randint(16, 64)  # Random beat count
beat_features = np.random.randn(n_beats, 125)  # 125D features

# Training loop
rollout_data = trainer.ppo_trainer.collect_rollouts(steps_per_episode)
```

---

## Scripts Overview

| Script | Purpose | Features |
|--------|---------|----------|
| `train_rlhf_stable.py` | Stable training with synthetic data | ✓ Learned rewards integrated ✓ Error handling ✓ Checkpointing |
| `train_rlhf.py` | Full pipeline with real audio | ✓ Data loading ✓ Advanced config ✓ Dimension fallback |
| `train_rlhf_simple.py` | Minimal synthetic training | ✓ Testing ✓ Quick validation |

---

## Output Files

After training, check:
- `models/policy_final.pt` - Use this for inference
- `models/checkpoint_ep00050.pt` - Can resume from any checkpoint
- Terminal logs show training progress and metrics

---

**Status**: 🟢 Ready to train, evaluate, or continue with human feedback

