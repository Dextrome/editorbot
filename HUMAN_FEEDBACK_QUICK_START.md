# Human Feedback - Quick Start (TL;DR)

## In 30 Seconds

**RLHF = Teach AI from human preferences**

```
Step 1: Make 5 versions of each song
Step 2: Humans pick: "I like version B better"
Step 3: AI learns the pattern
Step 4: AI makes better edits
Step 5: Repeat!
```

---

## The Commands

### 1. Generate Candidates (10 min)
```bash
python generate_eval_candidates.py --n_songs 10
```
Creates: `eval_outputs/` with 50 audio files ready for human evaluation

### 2. Collect Feedback (2-4 hours)

Listen to pairs, pick the better one, save to `feedback/preferences.json`:

```json
[
  {
    "song_id": "song_001",
    "edit_a_id": "temp_0.1",
    "edit_b_id": "temp_0.5",
    "preference": "a",           ← Which is better? a / b / tie
    "strength": 2,              ← How much? 1 = slight, 2 = moderate, 3 = strong
    "reasoning": "Tighter"      ← Why? (optional)
  },
  ...
]
```

### 3. Train on Feedback (30 min)
```bash
python train_from_feedback.py --feedback feedback/preferences.json
```
Creates: `models/reward_model_v9_feedback_final.pt` (learns your taste!)

### 4. Fine-tune Policy (1 hour)
```bash
python train_rlhf_stable.py \
  --episodes 500 \
  --reward_model models/reward_model_v9_feedback_final.pt
```
Updates: `models/policy_final.pt` (now follows human preferences!)

---

## What's Actually Happening

### Before Feedback
```
Policy = Random educated guesses about what's good
Generates: edits, some good, some bad
```

### After Collecting Feedback
```
Human: "Edit A is better than Edit B"
Reward Model learns: "A scored higher because X,Y,Z reasons"
```

### After Training on Feedback
```
Reward Model: "I predict humans will like edits with these features"
Policy: "I should generate edits with those features to get high rewards"
Result: Much better edits! 🎉
```

---

## Why This Works

❌ **Hard way**: Tell AI exactly what to do
- "Cut beats 3, 7, and 9"
- "Keep vocals, remove drums"
- Requires too many rules

✅ **Easy way** (RLHF): Show AI examples humans like
- Human rates: "I prefer edit A"
- AI learns: "What made A better?"
- AI generalizes: "I should do things like A does"

---

## Expected Results

### One Feedback Cycle

| Metric | Result |
|--------|--------|
| Time investment | 2-5 hours |
| Feedback pairs | 50-200 |
| Policy improvement | +15% to +25% |
| Effort | Medium |

### After 3 Cycles

| Metric | Result |
|--------|--------|
| Total time | 1-2 weeks |
| Feedback pairs | 500+ |
| Policy improvement | +50% or better |
| Effort | High but worth it |

---

## Data Format (Simplest)

**CSV (if you prefer)**:
```csv
song_id,edit_a,edit_b,preference,strength
song_001,temp_0.1,temp_0.5,a,2
song_001,temp_0.3,temp_0.7,b,1
song_001,temp_0.1,temp_0.9,tie,0
song_002,temp_0.2,temp_0.6,a,3
```

Convert to JSON:
```python
import pandas as pd
import json

df = pd.read_csv("feedback.csv")
data = []
for _, row in df.iterrows():
    data.append({
        "song_id": row["song_id"],
        "edit_a_id": row["edit_a"],
        "edit_b_id": row["edit_b"],
        "preference": row["preference"],
        "strength": int(row["strength"])
    })

with open("feedback/preferences.json", "w") as f:
    json.dump(data, f, indent=2)
```

---

## Common Questions

**Q: How many feedback pairs do I need?**
- A: Start with 30+, aim for 100+, excellent at 500+

**Q: How long does each person take to rate?**
- A: ~3-5 min per pair (listen + decide + note)
- 30 pairs = 1.5-2.5 hours per person

**Q: What if I disagree with myself?**
- A: Normal! Get multiple people, average their votes

**Q: Will the policy improve guaranteed?**
- A: If feedback is consistent and high-quality, yes!
- Bad/contradictory feedback can confuse the model

**Q: Can I use an annotation platform?**
- A: Yes! Scale AI, Labelbox, etc. make it easier for many raters

---

## Files Reference

| File | What It Is | What To Do |
|------|-----------|-----------|
| `generate_eval_candidates.py` | Creates edit pairs | Run to generate candidates |
| `train_from_feedback.py` | Trains reward model | Run after collecting feedback |
| `HUMAN_FEEDBACK_PIPELINE.md` | Deep explanation | Read for understanding |
| `FEEDBACK_DATA_FORMAT.md` | Data format details | Reference when collecting |
| `RLHF_WORKFLOW_COMPLETE.md` | Full workflow | Read for big picture |

---

## Checklist

- [ ] Run `generate_eval_candidates.py`
- [ ] Listen to 10-20 song pairs (pick audio files from eval_outputs/)
- [ ] Create `feedback/preferences.json` with 30+ annotations
- [ ] Run `train_from_feedback.py`
- [ ] Run `train_rlhf_stable.py` with new reward model
- [ ] Compare edits: old policy vs new policy
- [ ] Measure improvement (yes/no, better, how much)
- [ ] (Optional) Collect more feedback → repeat

---

## Success Indicators

✓ **Working well if:**
- Reward model loss decreases during training
- Policy loss decreases during training
- New edits sound better to your ears
- Humans rate new policy >60% better than old

✗ **Something wrong if:**
- Losses stay flat or increase
- New policy sounds worse
- Training crashes with errors
- Feedback contradicts itself

---

## Pro Tips

**Collecting Feedback:**
- Take breaks (ear fatigue is real!)
- Listen to full edits, not just clips
- Write quick notes (helps explain choices)
- Be consistent (if A is tight in song 1, remember that)
- Get 2+ people to rate (catches weird preferences)

**Training:**
- Start with high learning rate (1e-4) → decreases over time
- Train until validation loss plateaus
- Save best model (validation loss, not just final)
- Monitor GPU: `nvidia-smi`

**Iterating:**
- Compare v14 (old) vs v15 (new) with fresh raters
- Measure: % of humans who prefer new
- Target: >60% prefer new for "good iteration"
- Do 2-3 cycles for major improvement

---

## One More Thing

**The magic of RLHF:**
- You don't need to be a machine learning expert
- Just need to know what you like musically
- The system learns from you automatically
- Each feedback cycle makes it smarter

**Your role:**
- Generate diverse edits ✓ (AI does this)
- Judge which is better ✓ (You do this!)
- Let AI learn the pattern ✓ (Automatic)
- Get better edits ✓ (Result!)

---

## Start Now! 🚀

Ready? Open terminal and run:

```bash
python generate_eval_candidates.py --n_songs 5
```

Then:
1. Open `eval_outputs/` folder
2. Listen to the generated audio files
3. Pick your favorite pair
4. Note down: which you prefer, how much, why
5. Repeat for 20+ pairs
6. Save as `feedback/preferences.json`
7. Run the training scripts

**Total time: 3-5 hours for one complete cycle**

That's it! The system handles the rest. 🎵
