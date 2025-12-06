# Complete RLHF Workflow: From Training to Human Feedback

## The Full Loop

```
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: INITIAL TRAINING (Completed ✓)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: Generate dense reward signals                           │
│  ├─ Tempo consistency: beats align throughout edit             │
│  ├─ Energy flow: smooth dynamics without jarring drops         │
│  ├─ Phrase completeness: respect musical structure             │
│  └─ Transition quality: beats align well at boundaries          │
│                                                                   │
│  Step 2: Collect synthetic preference pairs                      │
│  └─ Bradley-Terry training: model learns to score edits         │
│     → Result: reward_model_v8 (2.63M parameters)                │
│                                                                   │
│  Step 3: Train policy with learned rewards                       │
│  └─ PPO agent learns: 80% learned rewards + 20% dense           │
│     → Result: policy_final.pt (889 state dims, 9 actions)       │
│                                                                   │
│  Status: ✓ COMPLETE (200 episodes trained)                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: HUMAN FEEDBACK COLLECTION (Your Turn Now!)             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: Generate Candidate Edits (~1 hour)                     │
│  ├─ Use trained policy with different "temperatures"            │
│  ├─ Temperature 0.1 = conservative (keeps more beats)           │
│  ├─ Temperature 0.9 = aggressive (cuts more, tighter)           │
│  └─ For each song: 5 versions with different styles             │
│                                                                   │
│  Command:                                                        │
│  $ python generate_eval_candidates.py \                          │
│      --songs_dir data/test_songs \                              │
│      --n_songs 10 \                                             │
│      --candidates_per_song 5                                    │
│                                                                   │
│  Output: eval_outputs/evaluation_manifest.json                   │
│  │       + 50 audio files (10 songs × 5 versions each)           │
│  │       + Pairwise comparison tasks ready for humans            │
│                                                                   │
│  Step 2: Collect Human Preferences (~2-5 hours)                 │
│  ├─ Listen to edit pairs                                        │
│  ├─ Choose: Which is better? A / B / Tie                        │
│  ├─ Rate strength: Slightly / Moderately / Significantly        │
│  └─ Optional: Add notes explaining preference                   │
│                                                                   │
│  Example Annotation:                                             │
│  {                                                              │
│    "song_id": "song_001",                                       │
│    "edit_a_id": "temp_0.1",                                     │
│    "edit_b_id": "temp_0.5",                                     │
│    "preference": "a",                                           │
│    "strength": 2,                                               │
│    "reasoning": "Tighter without losing vocal hook"            │
│  }                                                              │
│                                                                   │
│  Format: JSON (see FEEDBACK_DATA_FORMAT.md)                      │
│  File: feedback/preferences.json                                 │
│                                                                   │
│  Step 3: Train Reward Model on Feedback (~30 minutes)            │
│  ├─ Load v8 (pre-trained on synthetic data)                    │
│  ├─ Fine-tune on human preferences                              │
│  ├─ Bradley-Terry loss: reward(A) - reward(B) ≈ preference      │
│  └─ Save: reward_model_v9_feedback_final.pt                     │
│                                                                   │
│  Command:                                                        │
│  $ python train_from_feedback.py \                               │
│      --feedback feedback/preferences.json \                      │
│      --pretrained models/reward_model_v8_long/... \              │
│      --epochs 20                                                │
│                                                                   │
│  Output: reward_model_v9_feedback_final.pt                       │
│                                                                   │
│  Status: Ready to start (you are here!)                         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3: POLICY REFINEMENT (Next Step!)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: Fine-tune Policy with Human Preferences (~1-2 hours)   │
│  ├─ Load policy_final.pt (trained on v8)                        │
│  ├─ Use reward_model_v9 (trained on human feedback)             │
│  ├─ Continue PPO training: now chasing human preferences        │
│  └─ Result: policy_v15 (even better edits!)                     │
│                                                                   │
│  Command:                                                        │
│  $ python train_rlhf_stable.py \                                 │
│      --episodes 500 \                                           │
│      --reward_model models/reward_model_v9_feedback_final.pt    │
│                                                                   │
│  Output: models/policy_final.pt (updated)                        │
│                                                                   │
│  Step 2: Evaluate Improvements                                  │
│  ├─ Generate new edits with policy_v15                          │
│  ├─ Have humans rate: "How much better?"                        │
│  └─ Measure win rate: target >60% improvement                   │
│                                                                   │
│  Status: Ready after feedback collection                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4: ITERATION (Optional - Repeat for Excellence)           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Cycle 2: More Human Feedback                                   │
│  └─ Generate edits with policy_v15                              │
│     Collect 200+ more preference pairs                          │
│     Train reward_model_v10                                      │
│     Fine-tune policy_v16                                        │
│     → Further improvement!                                      │
│                                                                   │
│  Cycle 3: Active Learning                                       │
│  └─ Identify "hard cases" where models disagree                │
│     Collect feedback on disagreements                           │
│     Train reward_model_v11                                      │
│     → Resolves ambiguities!                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Guide

### You Are Here (Human Feedback Phase)

#### Timeline: 2-5 hours depending on effort

**Task 1: Generate Candidates (1 hour)**
```bash
python generate_eval_candidates.py \
  --songs_dir data/test_songs \
  --output_dir eval_outputs \
  --n_songs 10 \
  --candidates_per_song 5
```

Output: `eval_outputs/`
- `evaluation_manifest.json` - List all tasks
- `song_001/` - Audio files for song 1
- `song_002/` - Audio files for song 2
- etc.

**Task 2: Collect Feedback (2-4 hours)**

1. Open `eval_outputs/evaluation_manifest.json`
2. For each song, get the candidate audio files
3. Listen to each pair
4. For each comparison, choose:
   - **Preference**: Which is better? (A / B / Tie)
   - **Strength**: How much? (1=Slight, 2=Moderate, 3=Strong)
   - **Notes**: Why? (optional)

Save as `feedback/preferences.json`

**Task 3: Train Reward Model (30 minutes)**
```bash
python train_from_feedback.py \
  --feedback feedback/preferences.json \
  --epochs 20
```

Output:
- `models/reward_model_v9_feedback_final.pt`
- Training history with metrics

**Task 4: Fine-tune Policy (1-2 hours)**
```bash
python train_rlhf_stable.py \
  --episodes 500 \
  --reward_model models/reward_model_v9_feedback_final.pt
```

Output:
- `models/policy_final.pt` (updated with human preferences)
- Training logs showing improvements

---

## Key Concepts

### Temperature (Controls Edit Aggressiveness)

```
Temperature 0.1  →  Very Conservative  →  Keeps most beats  →  ~180 sec
Temperature 0.3  →  Moderately Keep    →  Removes few beats  →  ~168 sec
Temperature 0.5  →  Balanced           →  Mixed strategy     →  ~150 sec
Temperature 0.7  →  Moderately Cut     →  Removes many beats →  ~132 sec
Temperature 0.9  →  Very Aggressive    →  Only keeps best    →  ~90 sec
```

The policy generates different edits based on temperature. Humans prefer some approaches over others.

### Preference Signal

Instead of rating absolute quality (1-10, which is subjective), we collect **relative preferences**:

```
❌ Bad: "Edit A is 7/10"  (subjective, inconsistent between raters)
✅ Good: "A is better than B"  (objective, consistent across raters)
```

Bradley-Terry model learns the pattern:
- If humans say "A is better than B" (strength 2), train model so:
  - `reward(A) - reward(B) ≈ 2`

---

## Example: Full Workflow

### Scenario: You collected 50 preference pairs

**File: `feedback/preferences.json`**
```json
[
  {
    "song_id": "song_001",
    "edit_a_id": "temp_0.1",
    "edit_b_id": "temp_0.5",
    "preference": "b",
    "strength": 1,
    "reasoning": "Cut version is punchier"
  },
  {
    "song_id": "song_001",
    "edit_a_id": "temp_0.3",
    "edit_b_id": "temp_0.7",
    "preference": "a",
    "strength": 2,
    "reasoning": "0.3 keeps vocals better than 0.7"
  },
  ... (48 more pairs)
]
```

**Step 1: Train Reward Model**
```bash
$ python train_from_feedback.py --feedback feedback/preferences.json
```

Output:
```
✓ Loaded 50 preference pairs
  Train set: 45 pairs
  Val set: 5 pairs

TRAINING REWARD MODEL FROM PREFERENCES
Epoch  1/20 | Train Loss: 1.2341 | Val Loss: 1.1893
Epoch  2/20 | Train Loss: 0.8764 | Val Loss: 0.9234
Epoch  3/20 | Train Loss: 0.6234 | Val Loss: 0.7845
...
Epoch 20/20 | Train Loss: 0.2341 | Val Loss: 0.3123

Best epoch: 18
✓ Saved reward_model_v9_feedback_final.pt
```

**Step 2: Fine-tune Policy**
```bash
$ python train_rlhf_stable.py --episodes 500 --reward_model models/reward_model_v9_feedback_final.pt
```

Output:
```
RLHF TRAINING - SYNTHETIC DATA
Device: cuda
Episodes: 500

Episode  10/500 | Dense: 0.123 | Learned: 0.456 | Combined: 0.234 | Loss: 0.1234
Episode  20/500 | Dense: 0.145 | Learned: 0.523 | Combined: 0.267 | Loss: 0.1021
...
Episode 500/500 | Dense: 0.198 | Learned: 0.687 | Combined: 0.342 | Loss: 0.0456
  ✓ New best: 0.342

✓ Training complete!
  Total episodes: 500
  Best combined reward: 0.342
```

**Step 3: Evaluate Improvement**

Generate edits with new policy and compare with old:
```
Old policy (v14): 45% win rate with new humans
New policy (v15): 67% win rate with new humans
Improvement: +22 percentage points! 🎉
```

---

## What Happens Behind the Scenes

### Bradley-Terry Learning

The reward model learns through preference pairs:

```python
# If human says "Edit A is better (strength=2)"
preference_target = 2

# Train the model to output:
score_a = 5.2
score_b = 3.1
difference = 5.2 - 3.1 = 2.1  ✓ Matches target!

# Loss = (2.1 - 2.0)² = 0.01 (small loss, good!)
```

After 50+ preferences, the model learns:
- "More beat alignment" → higher score
- "Smoother transitions" → higher score
- "Tighter edits" → sometimes higher
- "Keeping vocals" → important to humans

### Policy Gradient Update

The policy learns to chase the learned reward:

```python
# Policy generates action:
action = KEEP_BEAT

# Reward model scores it:
reward = 0.45  (human preferences say this is good!)

# Policy gradient:
gradient = ∇ log π(action | state) × reward
update = policy_parameters - learning_rate × gradient

# Policy learns: for this state, KEEP_BEAT is good!
```

Over 500 episodes, policy becomes expert at:
- Making edits humans like
- Choosing good beats to keep
- Creating smooth transitions
- Balancing tightness with musicality

---

## Success Metrics to Track

### After First Feedback Iteration

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Feedback pairs collected | 50+ | Count annotations |
| Inter-rater agreement | 80%+ | Have 2+ people rate same songs |
| Reward model loss | <0.5 | Check training logs |
| Policy improvement | +10% | A/B test with new humans |

### After Second Iteration

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Total feedback pairs | 200+ | Accumulate across iterations |
| Policy consistency | 85%+ | Hold-out test set |
| Human alignment | 70%+ | Does policy match human taste? |

---

## Tips for Success

### For Collecting Feedback

✓ **Take breaks** - Ear fatigue is real, take 5-min break every 30 mins
✓ **Be consistent** - If you rate song A as "tight and good", remember this for song B
✓ **Trust your instinct** - First impression is often best
✓ **Note-taking** - Write quick notes explaining choices (helps debugging later)
✓ **Multiple raters** - 2-3 people is better than 1 (catch biases)

### For Model Training

✓ **Start with confident feedback** - Only use annotations with confidence >0.7
✓ **Balance preferences** - Mix of "A better", "B better", "Tie" is healthy
✓ **Track convergence** - Plot training loss, watch for plateaus
✓ **Validate on held-out** - Don't train/test on same feedback

### For Policy Refinement

✓ **Warm-start** - Load policy_final.pt, don't start from scratch
✓ **Higher learning rate** - Fine-tuning, not pre-training
✓ **Monitor for overfitting** - Stop if policy learning diverges
✓ **Save frequently** - Save checkpoint every 50 episodes

---

## Troubleshooting

### "I have only 10 preference pairs - is that enough?"
- **No**, minimum ~30-50 for meaningful training
- **But**: Better than nothing! Train and observe
- **Next**: Collect 100+ pairs for reliable model

### "The new policy is worse than the old one"
- Could be reward model is learning wrong pattern
- Try: Lower learning rate, more epochs
- Or: Check feedback annotations for errors/contradictions

### "I keep changing my mind about preferences"
- Normal! Musical taste has ambiguity
- Solution: Get multiple people to rate same songs
- Average their preferences to reduce noise

### "Training is slow"
- Check GPU usage: `nvidia-smi`
- Reduce batch size if running out of memory
- Speed up: Fewer episodes (100 vs 500) for testing

---

## Next Steps After Feedback

### Option 1: Iterate (Recommended)
1. Collect 100+ more preference pairs
2. Train reward_model_v10
3. Fine-tune policy_v16
4. Measure improvement

### Option 2: Scale Up
1. Use annotation platform (Scale AI, Labelbox)
2. Collect 500+ pairs from diverse raters
3. Train production-quality models

### Option 3: Active Learning
1. Generate hard cases where models disagree
2. Collect focused feedback on disagreements
3. Resolve ambiguities
4. Converge faster

---

## Files You'll Create

```
feedback/
├─ preferences.json                    (your annotations)
├─ annotations_batch_001.csv          (optional CSV format)
└─ quality_report.json                (optional validation results)

models/
├─ reward_model_v9_feedback_best.pt    (from train_from_feedback.py)
├─ reward_model_v9_feedback_final.pt   (best model to use)
└─ policy_final.pt                     (updated from train_rlhf_stable.py)

eval_outputs/
├─ evaluation_manifest.json            (from generate_eval_candidates.py)
├─ song_001/
│  ├─ song_001_temp_0.1.wav
│  ├─ song_001_temp_0.3.wav
│  ├─ song_001_temp_0.5.wav
│  ├─ song_001_temp_0.7.wav
│  └─ song_001_temp_0.9.wav
└─ song_002/
   └─ ...
```

---

## You've Now Completed

✅ Initial RLHF training (200 episodes)
✅ Learned reward model (v8, synthetic data)
✅ Basic policy (policy_final.pt)

## Next: Human Feedback Loop

Now you can:
1. Generate diverse edit candidates
2. Collect human preferences
3. Train model that aligns with humans
4. Improve policy quality

**Time estimate**: 2-5 hours for one feedback iteration
**Expected result**: 15-25% improvement in policy quality

Ready to start? → Run `python generate_eval_candidates.py` 🚀
