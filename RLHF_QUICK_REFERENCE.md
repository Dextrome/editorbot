# RLHF Quick Reference

## TL;DR - Start Training in 30 Seconds

```bash
# 1. Ensure training data is prepared
ls training_data/train/*.wav | head -5

# 2. Start RLHF training
python train_rlhf.py --data_dir ./training_data --total_steps 100000

# 3. Watch progress
tail -f logs/training.log
```

## Command Examples

### Basic PPO Training
```bash
python train_rlhf.py --data_dir ./training_data
```

### Full Configuration
```bash
python train_rlhf.py \
    --data_dir ./training_data \
    --total_steps 500000 \
    --algorithm ppo \
    --reward_model models/reward_model_v8_long/reward_model_final.pt \
    --device cuda \
    --save_interval 10000 \
    --log_level INFO
```

### Resume from Checkpoint
```bash
python train_rlhf.py \
    --data_dir ./training_data \
    --checkpoint models/checkpoint_50000.pt \
    --total_steps 200000
```

### DPO Training (Alternative)
```bash
python train_rlhf.py --algorithm dpo --data_dir ./training_data
```

### CPU-Only (No GPU)
```bash
python train_rlhf.py --device cpu --data_dir ./training_data
```

## Key Files

| File | Purpose | Size |
|------|---------|------|
| `rl_editor/learned_reward_integration.py` | Core integration module | 530 lines |
| `train_rlhf.py` | Main training script | 280 lines |
| `test_rlhf_integration.py` | Integration tests | 165 lines |
| `RLHF_TRAINING_GUIDE.md` | Comprehensive guide | 500+ lines |
| `RLHF_INTEGRATION_COMPLETE.md` | Detailed documentation | 600+ lines |

## Core API

### Load Reward Model
```python
from rl_editor.learned_reward_integration import LearnedRewardIntegration
from rl_editor.config import get_default_config

config = get_default_config()
reward = LearnedRewardIntegration(config)
reward.load_model()  # Returns True if successful
```

### Compute Reward
```python
import numpy as np

beat_features = np.random.randn(32, 125)  # 32 beats
action_ids = np.zeros(32, dtype=int)      # All KEEP actions
reward_value = reward.compute_learned_reward(beat_features, action_ids)
print(f"Reward: {reward_value:.3f}")  # Range: [-10, 10]
```

### Train Policy
```python
from train_rlhf import RLHFTrainer

trainer = RLHFTrainer(config)
results = trainer.train(
    data_dir="./training_data",
    total_timesteps=100000
)
```

## Reward Components

```
Learned Reward (80%)
├─ Input: Beat features (125D) + action IDs
├─ Model: 3-layer transformer
└─ Output: Scalar [-10, 10]

Dense Reward (20%)
├─ Tempo consistency (25%)
├─ Energy flow (25%)
├─ Phrase completeness (25%)
└─ Transition quality (25%)

Combined = 0.8 * learned + 0.2 * dense
```

## Output Files

```
models/
├─ policy_best.pt      # Best policy weights
├─ policy_final.pt     # Final policy weights
├─ checkpoint_10000.pt # Periodic checkpoints
└─ checkpoint_20000.pt

logs/
├─ training_history.json  # Metrics (JSON)
└─ training.log            # Log file (text)
```

## Monitoring Training

### Real-time Log
```bash
tail -f logs/training.log
```

### Plot Rewards
```python
import json
import matplotlib.pyplot as plt

with open('logs/training_history.json') as f:
    h = json.load(f)

plt.plot(h['step'], h['total_reward'])
plt.xlabel('Step')
plt.ylabel('Reward')
plt.show()
```

### Key Metrics
- `total_reward`: Combined reward signal
- `learned_reward`: Preference model score
- `dense_reward`: Environment metrics
- `policy_loss`: Policy network loss
- `value_loss`: Value network loss

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model won't load | Check path: `models/reward_model_v8_long/reward_model_final.pt` |
| CUDA out of memory | Add `--device cpu` or reduce batch size |
| Reward always -10 | Normal for untrained features; will improve during training |
| Training too slow | Use GPU: `--device cuda` |
| Loss not decreasing | Check learning rate, increase patience |

## Configuration Parameters

```bash
--data_dir              Training data directory (required)
--checkpoint            Resume from checkpoint (optional)
--reward_model          Path to reward model (default: v8)
--total_steps           Total training steps (default: 100000)
--algorithm             PPO or DPO (default: ppo)
--device                cuda or cpu (default: cuda)
--save_interval         Save every N steps (default: 10000)
--log_level             DEBUG, INFO, WARNING (default: INFO)
```

## Environment Variables

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Run training
python train_rlhf.py --data_dir ./training_data
```

## Testing

```bash
# Run integration tests
python test_rlhf_integration.py

# Expected output: ✓ ALL TESTS PASSED
```

## Performance Tips

1. **Larger Batches**: Increase `n_steps` in config (more parallel rollouts)
2. **Mixed Precision**: Already enabled for speed
3. **Gradient Accumulation**: Already supported in trainer
4. **Multi-GPU**: Set `CUDA_VISIBLE_DEVICES` for multiple GPUs
5. **Data Loading**: Use SSD for training data, not HDD

## Next Steps After Training

```bash
# 1. Evaluate best policy
python -m rl_editor.evaluation \
    --checkpoint models/policy_best.pt \
    --test_dir ./test_audio/

# 2. Generate edits for human feedback
python -m rl_editor.infer \
    --checkpoint models/policy_best.pt \
    --audio test_song.wav \
    --output edited_song.wav

# 3. Collect human preferences (future)
# 4. Train new reward model (future)
# 5. Iterate RLHF (future)
```

## Project Structure

```
editorbot/
├── rl_editor/
│   ├── learned_reward_integration.py  ← NEW
│   ├── train_reward_model.py
│   ├── trainer.py
│   ├── agent.py
│   ├── environment.py
│   └── ...
├── train_rlhf.py                      ← NEW
├── test_rlhf_integration.py            ← NEW
├── RLHF_TRAINING_GUIDE.md              ← NEW
├── RLHF_INTEGRATION_COMPLETE.md        ← NEW
└── training_data/
    └── train/
        ├── song1.wav
        ├── song2.wav
        └── ...
```

## Status

✅ **Ready for RLHF Training**

- ✅ Reward model loaded (v8 - trained & verified)
- ✅ Integration complete (tested with 8-part suite)
- ✅ Training pipeline ready
- ✅ No breaking changes
- ✅ Backward compatible
- ✅ Documentation complete

## Need Help?

1. See `RLHF_TRAINING_GUIDE.md` for detailed documentation
2. See `RLHF_INTEGRATION_COMPLETE.md` for architecture details
3. Run `python test_rlhf_integration.py` to verify setup
4. Check logs: `tail -f logs/training.log`

---

**Last Updated**: December 6, 2025
**Version**: 1.0
**Status**: ✅ Production Ready
