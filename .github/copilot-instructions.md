<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Audio Editor Project

This is a Python project for an AI-powered audio editor that processes raw music recordings and transforms them into polished, listenable songs.

## Current Goal: True RL Editing (End-to-End Learned)

Instead of pipeline approach:
```
Input Audio  [ML: Score beats]  [Rules: Threshold, Merge, Concat]  Output
```

We're building full RL-based editing:
```
Input Audio  [RL Agent: Choose actions]  Output
                    
            Actions: KEEP, CUT, LOOP, CROSSFADE, REORDER
                    
            Reward: Human feedback / Audio quality metrics
```

## Key Components Needed

### 1. Action Space - What the agent can do
- **KEEP(beat_i)** - Include this beat in output
- **CUT(beat_i)** - Remove this beat from output
- **LOOP(beat_i, n_times)** - Repeat a beat or section n times
- **CROSSFADE(beat_i, beat_j, duration_s)** - Blend smoothly between sections
- **REORDER(section_a, section_b)** - Move sections around (non-linear editing)

### 2. State - What the agent sees
- **Current position** in track (beat index)
- **Audio features** of current beat and surrounding context
  - Stem-separated tracks (drums, bass, vocals, other)
  - Mel-spectrogram features
  - Beat-level and phrase-level descriptors
- **Edit history** - What's been kept/cut so far
- **Target constraints** - Duration remaining, target keep ratio (~35%)
- **Global features** - Tempo curve, energy contour, key, overall energy level

### 3. Reward Signal - How to train
Three complementary signals:
- **Sparse rewards** - Human rates final edit 1-10 (ground truth, but slow)
- **Dense rewards** - Automatic metrics:
  - Tempo consistency (keep similar BPM throughout)
  - Energy flow (smooth dynamics, no jarring drops)
  - Phrase completeness (respect musical phrase boundaries)
  - Transition quality (beats align well at boundaries)
- **Learned reward model** - Train from human preference pairs (like RLHF for LLMs):
  - Input: Two edit versions of same song
  - Output: Which one is better (and by how much)
  - Use this to train dense reward function

### 4. Training Approach
- **Offline RL**: Learn from existing rawedit pairs as demonstrations
  - Imitation learning warmup (supervised learning from human edits)
  - Then refine with learned reward model
- **Online RL**: Generate edits, collect human feedback, improve
- **RLHF Pipeline**: 
  1. Train reward model on human preference data
  2. Use learned reward in PPO/DPO training
  3. Generate candidate edits with different policies
  4. Collect human feedback to refine reward model (loop)

## Project Structure
- `AIeditor/` - Existing models (V9, V13, hybrid editors)
- `rl_editor/` - New RL-based editing framework
  - `config.py` - Hyperparameters and settings
  - `environment.py` - Gym-style editing environment
  - `agent.py` - Policy and value networks (transformer-based)
  - `reward_model.py` - Learned reward from human preferences
  - `trainer.py` - PPO training loop
  - `actions.py` - Action definitions and masking
  - `utils.py` - Audio processing utilities
- `training_data/` - Raw and edited audio pairs for training
- `models/` - Saved policy and reward model checkpoints

## Key Technologies
- Python 3.10+
- librosa - Audio analysis and feature extraction
- torch/torchaudio - Deep learning and RL training
- pydub - Audio manipulation
- soundfile - Audio file I/O
- numpy - Numerical operations
- gymnasium - RL environment interface
- stable-baselines3 or custom PPO/DPO - RL algorithms

## Development Guidelines
- Use type hints for all functions
- Follow PEP 8 style guidelines
- Write unit tests for new features
- Document all public APIs
- Use dataclass Config pattern for all hyperparameters
- Gymnasium-compatible environment interface
- Checkpoint/resume training capability
- Parralel Processing for faster data loading
- Logging with TensorBoard/Weights & Biases
- Version control with Git and clear commit messages
- Optimize for GPU training
- Modular code structure for easy experimentation
- Multiple Worker Optimization
- Gradient Accumulation for large batch sizes
- Mixed Precision Training for speed and memory efficiency
- Clear separation of concerns (data, model, training, utils)
- Visualization of training progress and metrics
- Visualize audio edits for debugging
