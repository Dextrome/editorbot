# RL Editor Python Commands

## Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
pip install librosa pydub soundfile numpy gymnasium stable-baselines3 tensorboard wandb
```

## Training Commands

### Train RL Agent (PPO)
```bash
# Basic training
python -m rl_editor.trainer

# With custom config
python -m rl_editor.trainer --config configs/ppo_config.yaml

# Resume from checkpoint
python -m rl_editor.trainer --checkpoint models/checkpoint_1000.pt

# With TensorBoard logging
python -m rl_editor.trainer --log-dir runs/experiment_1
```

### Train Reward Model (RLHF)
```bash
# Train reward model from human preferences
python -m rl_editor.reward_model --mode train --data training_data/preferences.json

# Evaluate reward model
python -m rl_editor.reward_model --mode eval --checkpoint models/reward_model.pt
```

## Data Preparation

### Generate Training Data
```bash
# Extract features from raw audio
python -m rl_editor.data.preprocess --input training_data/raw --output training_data/processed

# Create preference pairs for reward model
python -m rl_editor.data.create_preferences --input training_data/edit_pairs
```

## Inference / Editing

### Generate Edits
```bash
# Edit a single audio file
python -m rl_editor.inference --input song.wav --output edited.wav --model models/policy.pt

# Batch editing
python -m rl_editor.inference --input-dir songs/ --output-dir edited/ --model models/policy.pt

# With visualization
python -m rl_editor.inference --input song.wav --output edited.wav --visualize --save-plot edit_viz.png
```

## Evaluation

### Evaluate Agent Performance
```bash
# Evaluate on test set
python -m rl_editor.evaluate --test-dir test_data/ --model models/policy.pt

# Compare multiple models
python -m rl_editor.evaluate --models models/v1.pt models/v2.pt models/v3.pt

# Generate metrics report
python -m rl_editor.evaluate --model models/policy.pt --output-report metrics.json
```

## Utilities

### Check CUDA Availability
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Visualize Training Progress
```bash
# Launch TensorBoard
tensorboard --logdir runs/

# Launch Weights & Biases
wandb login
python -m rl_editor.trainer --use-wandb
```

### Test Environment
```bash
# Test Gym environment
python -m rl_editor.environment --test --episodes 10

# Visualize action space
python -m rl_editor.actions --visualize
```

## Development / Debugging

### Run Unit Tests
```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_environment.py

# With coverage
pytest --cov=rl_editor tests/
```

### Profile Performance
```bash
# Profile training loop
python -m cProfile -o profile.stats -m rl_editor.trainer
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
python -m memory_profiler rl_editor/trainer.py
```

### Export Model
```bash
# Export to ONNX for deployment
python -m rl_editor.export --model models/policy.pt --format onnx --output models/policy.onnx

# Export to TorchScript
python -m rl_editor.export --model models/policy.pt --format torchscript --output models/policy_traced.pt
```

## Interactive Mode

### Launch Interactive Editor
```bash
# Interactive Python shell with RL editor
python -i -m rl_editor.interactive

# IPython notebook
jupyter notebook notebooks/rl_editor_demo.ipynb
```
## V2 Specific Commands

### Training V2 Agent
```bash
# Train V2 RL agent
python -m rl_editor.v2.train

# Train with custom hyperparameters
python -m rl_editor.v2.train --learning-rate 3e-4 --batch-size 64 --epochs 100

# Resume V2 training
python -m rl_editor.v2.train --resume --checkpoint models/v2_checkpoint.pt
```

### V2 Inference
```bash
# Edit audio with V2 agent
python -m rl_editor.v2.inference --input song.wav --output edited.wav

# V2 batch processing
python -m rl_editor.v2.inference --input-dir raw_songs/ --output-dir v2_edits/
```

### V2 Evaluation
```bash
# Evaluate V2 agent
python -m rl_editor.v2.evaluate --test-dir test_data/ --model models/v2_policy.pt

# Compare V2 with other versions
python -m rl_editor.evaluate --models models/v2.pt models/v9.pt models/v13.pt
```

### V2 Environment Testing
```bash
# Test V2 environment
python -m rl_editor.v2.environment --test --episodes 5

# Visualize V2 action distributions
python -m rl_editor.v2.visualize --checkpoint models/v2_policy.pt
```