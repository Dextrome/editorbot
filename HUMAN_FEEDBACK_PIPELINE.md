# Human Feedback Pipeline for RLHF

## Overview
The human feedback loop is the core of Reinforcement Learning from Human Feedback (RLHF). Instead of waiting for perfect ground-truth labels, we collect **preference pairs** and train a reward model to learn what humans find good.

```
Raw Audio  →  [Policy: Generate 2 Edits]  →  [Human: Which is better?]  →  [Reward Model: Learn Pattern]
                                                                                        ↓
                                                        [Policy: Improve based on learned preferences]
```

---

## The Three-Step Cycle

### Step 1: Generate Multiple Edit Candidates

The trained policy creates several different edits of the same song:

```python
from rl_editor.trainer import PPOTrainer
import numpy as np

policy = PPOTrainer(config)
policy.load_checkpoint("./models/policy_final.pt")

# For each song, generate 2-4 different edit versions
raw_audio = load_audio("song.wav")
beat_features = extract_features(raw_audio)

# Version A: Conservative (keeps more beats)
policy.set_temperature(0.1)  # Low temperature = more deterministic
edit_a = policy.generate_edit(beat_features)

# Version B: Aggressive (cuts more, tighter edit)
policy.set_temperature(0.7)  # High temperature = more exploration
edit_b = policy.generate_edit(beat_features)

# Version C: Different strategy
policy.set_temperature(0.5)
edit_c = policy.generate_edit(beat_features)

save_audio("edit_a.wav", edit_a)
save_audio("edit_b.wav", edit_b)
save_audio("edit_c.wav", edit_c)
```

**Output**: 3 different edits of the same song, each with different characteristics.

---

### Step 2: Collect Human Preferences

Humans listen and rate the edits. This is much faster than labeling raw data!

#### Option A: Pairwise Comparison (Easiest for Humans)

Present 2 edits at a time:

```
🎵 EDIT A: [2:30] Conservative, keeps more vocal
🎵 EDIT B: [2:15] Tighter, more energetic

Which edit do you prefer?
  A (Better) | B (Better) | Equally Good | Equally Bad

How much better? 
  Slightly | Moderately | Significantly
```

**Advantages**:
- ✓ Faster to judge (A vs B is easier than rating 1-10)
- ✓ More reliable preferences
- ✓ Only need ~200-500 pairs to train good reward model

**Output**: Preference pair + strength (e.g., "A is moderately better than B")

#### Option B: Absolute Rating (If you prefer)

```
EDIT A: [2:30]
🎵 [Audio player]

Rate this edit: 
  ⭐ Poor (jarring, bad cuts)
  ⭐⭐ Okay (acceptable)
  ⭐⭐⭐ Good (well edited)
  ⭐⭐⭐⭐ Excellent (professional quality)

Notes: [Optional feedback]
```

**Advantages**:
- ✓ Absolute scale helpful for distribution
- ✓ Can collect more data per edit

**Disadvantages**:
- ✗ Slower to judge
- ✗ More inconsistent (one person's 4-star = another's 3-star)

---

## Data Collection Format

### Preference Pair Dataset
```json
[
  {
    "song_id": "song_001",
    "edit_a": {
      "path": "outputs/song_001_edit_a.wav",
      "actions": [0, 1, 0, 1, 2, 0, ...],
      "duration_seconds": 215
    },
    "edit_b": {
      "path": "outputs/song_001_edit_b.wav", 
      "actions": [0, 0, 1, 0, 2, 1, ...],
      "duration_seconds": 230
    },
    "preference": "a",
    "strength": 2,
    "rater": "user_123",
    "timestamp": "2025-12-06T15:30:00",
    "notes": "Edit A is tighter and more energetic"
  },
  {
    "song_id": "song_002",
    "edit_a": {...},
    "edit_b": {...},
    "preference": "b",
    "strength": 1,
    "rater": "user_456",
    "timestamp": "2025-12-06T15:35:00",
    "notes": "Both are similar, slight preference for B's flow"
  }
]
```

**Key fields**:
- `preference`: "a" | "b" | "tie"
- `strength`: 1 (slight) | 2 (moderate) | 3 (strong)
- `rater`: Who provided the feedback
- `notes`: Optional qualitative feedback

---

## Step 3: Train Reward Model on Preferences

Once you have ~200+ preference pairs, retrain the reward model:

```python
import torch
from torch.utils.data import DataLoader

# Load preference data
preference_data = load_json("feedback/preferences.json")

# Convert to Bradley-Terry format
# P(edit_a > edit_b) = sigmoid(score_a - score_b)
training_pairs = []
for pair in preference_data:
    edit_a_features = extract_features(pair["edit_a"]["path"])
    edit_b_features = extract_features(pair["edit_b"]["path"])
    
    # Convert preference to target score
    if pair["preference"] == "a":
        target = pair["strength"]  # 1, 2, or 3
    elif pair["preference"] == "b":
        target = -pair["strength"]  # -1, -2, or -3
    else:
        target = 0  # Tie
    
    training_pairs.append({
        "features_a": edit_a_features,
        "features_b": edit_b_features,
        "target": target  # How much better A is than B
    })

# Create dataloader
dataloader = DataLoader(training_pairs, batch_size=32, shuffle=True)

# Train reward model
reward_model = LearnedRewardModel(input_dim=125)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)

for epoch in range(10):
    for batch in dataloader:
        # Forward pass
        score_a = reward_model(batch["features_a"])
        score_b = reward_model(batch["features_b"])
        
        # Bradley-Terry loss: penalize if scores don't match preferences
        preference_loss = torch.nn.functional.mse_loss(
            score_a - score_b,  # Predicted preference
            batch["target"]      # Actual preference
        )
        
        # Backprop
        optimizer.zero_grad()
        preference_loss.backward()
        optimizer.step()

torch.save(reward_model.state_dict(), "models/reward_model_v9_with_feedback.pt")
```

**Loss function**: Bradley-Terry model
- If human says "A is moderately better (strength=2)", train model so `score_a - score_b ≈ 2`
- If "tie", train so `score_a ≈ score_b`

---

## Full Feedback Loop

```
┌─────────────────────────────────────────────────────────┐
│ ITERATION 1: Initial Training (Done ✓)                 │
├─────────────────────────────────────────────────────────┤
│ ✓ Train policy with dense rewards (audio quality)      │
│ ✓ Train reward model v8 (synthetic preference data)    │
│ ✓ Train policy v14 with learned rewards                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ ITERATION 2: Human Feedback Loop (Next)                │
├─────────────────────────────────────────────────────────┤
│ 1. Generate 500 song edits with policy v14             │
│    - Sample 50 songs, generate 10 edits each           │
│    - Vary temperature: [0.1, 0.3, 0.5, 0.7, 0.9]       │
│                                                          │
│ 2. Collect human preferences                            │
│    - Pairwise: Pick 2-3 best/worst from each song      │
│    - Get 10-20 pairwise comparisons per song           │
│    - 500-1000 total preference pairs                    │
│                                                          │
│ 3. Train reward model v9                                │
│    - Fine-tune v8 on human preferences                 │
│    - New model learns human tastes                      │
│                                                          │
│ 4. Train policy v15                                     │
│    - Use reward_model_v9 (human preferences)           │
│    - Reach alignment with human preferences            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ ITERATION 3: Second Round (Optional)                   │
├─────────────────────────────────────────────────────────┤
│ - Generate edits with policy v15                        │
│ - Collect feedback on new edits                         │
│ - Train reward_model_v10                               │
│ - Repeat as needed for convergence                      │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

### Phase 1: Generate Candidates (~1 hour)
```python
# scripts/generate_eval_set.py

from rl_editor.trainer import PPOTrainer
from pathlib import Path
import json

policy = PPOTrainer(config)
policy.load_checkpoint("./models/policy_final.pt")

eval_songs = list(Path("data/test_songs").glob("*.wav"))[:50]  # 50 songs
outputs = []

for song_path in eval_songs:
    beat_features = extract_features(song_path)
    
    # Generate 3-5 candidates per song
    candidates = []
    for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
        edit = policy.generate_edit(beat_features, temperature=temp)
        output_path = f"eval_outputs/{song_path.stem}_temp{temp}.wav"
        save_audio(output_path, edit)
        
        candidates.append({
            "temperature": temp,
            "path": output_path,
            "duration": len(edit) / sr
        })
    
    outputs.append({
        "song": str(song_path),
        "candidates": candidates
    })

with open("eval_outputs/manifest.json", "w") as f:
    json.dump(outputs, f)
```

### Phase 2: Collect Feedback (~2-5 hours depending on annotators)
```python
# Interface for collecting preferences

"""
EVALUATION INTERFACE:

For each song:
  🎵 Original Audio: [Play button]
  
  Candidate A (Temperature 0.1): [Play] Duration: 2:15
  Candidate B (Temperature 0.5): [Play] Duration: 2:08
  Candidate C (Temperature 0.9): [Play] Duration: 2:22
  
  Pairwise Comparisons:
  ☐ A vs B: [A] [B] [Tie]  Strength: [1-5]
  ☐ A vs C: [A] [C] [Tie]  Strength: [1-5]
  ☐ B vs C: [B] [C] [Tie]  Strength: [1-5]
  
  Notes: [text field]
  
  Next Song »
"""
```

Tools to use:
- **Simple**: Google Forms + audio links
- **Better**: Custom web interface (Flask/React)
- **Best**: Dedicated annotation platform (Scale AI, Labelbox, etc.)

### Phase 3: Train Reward Model on Feedback (~30 minutes)
```python
# train_reward_model_from_feedback.py

from rl_editor.learned_reward_integration import LearnedRewardModel
import torch

preferences = load_json("feedback/preferences.json")
training_data = convert_to_training_pairs(preferences)

model = LearnedRewardModel(input_dim=125)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Fine-tune from v8
model.load_state_dict(
    torch.load("models/reward_model_v8_long/reward_model_final.pt")
)

# Train on preferences
for epoch in range(20):
    total_loss = train_epoch(model, optimizer, training_data)
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

torch.save(model.state_dict(), "models/reward_model_v9_feedback.pt")
```

### Phase 4: Fine-tune Policy (~1-2 hours)
```bash
python train_rlhf_stable.py \
  --episodes 500 \
  --reward_model models/reward_model_v9_feedback.pt \
  --log_level INFO
```

---

## Key Insights

### Why This Works

1. **Preference Signal is Stronger**
   - Instead of absolute ratings (subjective), humans give relative preferences (consistent)
   - "A is better than B" is more reliable than "A gets 7/10"

2. **Reward Model Learns Pattern**
   - Model doesn't need to see every possible edit
   - Learns: "More beat alignment" → higher score
   - Learns: "Smooth transitions" → higher score

3. **Policy Improves Automatically**
   - Policy trained with v8 rewards → generates edits that v8 likes
   - V8 trained on human preferences → generates edits humans like
   - Policy trained with v9 rewards → generates even better edits

### Data Requirements

| Feedback Type | Amount | Time to Collect | Model Quality |
|---------------|--------|-----------------|---------------|
| Synthetic (v8) | ~1000 pairs | 0 (generated) | Good (baseline) |
| Human preferences | 200-500 pairs | 2-5 hours | Excellent |
| Human preferences | 1000+ pairs | 10-20 hours | Outstanding |

### Practical Timeline

```
Hour 0:    Have policy_final.pt ✓
Hour 0-1:  Generate 500 edits (50 songs × 10 variants)
Hour 1-6:  Collect human feedback (200-500 pairs)
Hour 6-6.5: Train reward_model_v9
Hour 6.5-8: Fine-tune policy v15
Hour 8+:   Evaluate improved policy
```

---

## What You Could Do This Week

### Minimal Effort (~2 hours)
1. Generate 10-20 edit pairs
2. Rate them yourself (quick preference data)
3. Retrain reward model
4. See if policy improves

### Good Effort (~4 hours)
1. Generate 50-100 edit pairs
2. Get 2-3 people to rate them
3. Combine feedback
4. Train reward model
5. Fine-tune policy

### Professional (~1 week)
1. Generate 500+ edit pairs systematically
2. Use online annotation platform
3. Get 5+ raters for quality
4. Calculate inter-rater agreement
5. Train multiple versions of reward model
6. A/B test policies
7. Iterate based on results

---

## Visualization

```
                    HUMAN RATERS
                   /    |    \
                  ↓     ↓     ↓
    EDIT A vs B vs C → Preferences → Bradley-Terry
                              ↓
                       Preference Scores
                       (A: +2.3 vs B)
                              ↓
                       Reward Model v9
                       (trained on human data)
                              ↓
                       Policy v15 trained
                       (aligns with humans)
                              ↓
                       Better Edits!
```

---

## Success Metrics

After one human feedback iteration, measure:

```python
# Compare edits
def evaluate_improvement():
    policy_old = PPOTrainer(config)
    policy_old.load_checkpoint("models/policy_v14.pt")
    
    policy_new = PPOTrainer(config)
    policy_new.load_checkpoint("models/policy_v15.pt")
    
    test_songs = load_songs("data/test_set")
    
    # For each song, generate edits with both policies
    win_count = 0
    for song in test_songs:
        edit_old = policy_old.generate_edit(song)
        edit_new = policy_new.generate_edit(song)
        
        # A/B test with held-out raters
        preference = human_rate(edit_old, edit_new)
        if preference == "new":
            win_count += 1
    
    improvement = win_count / len(test_songs)
    print(f"New policy wins: {improvement*100:.1f}%")
    # Target: >60% win rate = significant improvement
```

---

## Conclusion

The human feedback loop is what makes RLHF special:

1. **Generate** diverse edits with trained policy
2. **Collect** human preferences (much easier than raw labeling)
3. **Train** reward model on preferences
4. **Improve** policy with learned rewards
5. **Repeat** until satisfied

Each iteration gets better because the reward model learns human preferences from real data.
