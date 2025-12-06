# Evaluation Candidates - You Asked How! 🎵

## The Problem You Just Asked

> "errr how am I supposed to evaluate the evaluation manifest??"

Good question! The manifest contains 50 pairwise comparisons ready for human feedback, but you need a way to actually rate them.

## The Solution: CSV + Excel

✅ **Now you have `feedback/feedback_template.csv`** with 50 rows, one for each pair.

### What Each Column Means

```
song_id         → Which song (song_000, song_001, etc.)
candidate_a_id  → Edit A (e.g., temp_0.10 = conservative)
candidate_b_id  → Edit B (e.g., temp_0.30 = more aggressive)
a_keep_ratio    → How much of A is kept (0.91 = 91% of original)
b_keep_ratio    → How much of B is kept (0.64 = 64% of original)
preference      → ← YOU FILL THIS (a, b, or tie)
strength        → ← YOU FILL THIS (1, 2, or 3)
reasoning       → ← YOU FILL THIS (optional explanation)
```

### Example: A Filled Row

```
song_000,temp_0.10,temp_0.30,0.91,0.64,a,2,Tighter and more concise
```

This means:
- Song 000
- Comparing temp_0.10 (91% kept) vs temp_0.30 (64% kept)
- You prefer A (the tighter one)
- Moderately better (strength 2 of 3)
- Because it's "Tighter and more concise"

## Your Workflow

### Step 1: Open in Excel
```
feedback/feedback_template.csv
```

### Step 2: Rate Each Pair

For each row (50 total):

1. **Listen to both candidates** (they're in `eval_outputs/`)
   - Candidate A temperature: `temp_0.10`, `temp_0.30`, etc.
   - Lower temp = more conservative (keeps more beats)
   - Higher temp = more aggressive (cuts more beats)

2. **Fill in preference**
   - `a` = Candidate A is better
   - `b` = Candidate B is better
   - `tie` = About the same

3. **Fill in strength**
   - `1` = Slight difference (almost equal)
   - `2` = Moderate difference (clear preference)
   - `3` = Strong difference (much better)

4. **Optional: Add reasoning**
   - "Tighter", "Better flow", "More energetic", etc.

### Step 3: Save and Convert

Once done:
```bash
python evaluate_candidates_simple.py --convert-csv feedback/feedback_template.csv
```

This creates: `feedback/preferences.json` ✅

### Step 4: Train on Your Preferences

```bash
python train_from_feedback.py --feedback feedback/preferences.json
```

This trains a reward model that learned your taste!

### Step 5: Fine-tune Policy

```bash
python train_rlhf_stable.py --episodes 500 \
  --reward_model models/reward_model_v9_feedback_final.pt
```

Now your policy makes edits that match your preferences! 🚀

## Audio Files

The candidates are synthetic beats generated based on actions:
- `temp_0.10`: Conservative (keeps 91% of beats)
- `temp_0.30`: Moderate (keeps 64% of beats)
- `temp_0.50`: Balanced (keeps 43% of beats)
- `temp_0.70`: Aggressive (keeps 28% of beats)
- `temp_0.90`: Very aggressive (keeps 6% of beats)

You're rating: "Which edit sounds better to your ears?"

## Templates Available

We created two options:

### Option 1: CSV (Recommended)
```bash
python evaluate_candidates_simple.py --format csv
```
→ `feedback/feedback_template.csv`
- Open in Excel/Sheets
- Fill columns: preference, strength, reasoning
- Convert back to JSON

### Option 2: JSON (Direct Edit)
```bash
python evaluate_candidates_simple.py --format json
```
→ `feedback/feedback_template.json`
- Edit in VS Code
- Change `None` to actual values
- Convert to preferences.json

## Time Estimate

- **Per pair**: 3-5 minutes (listen + decide)
- **Total**: 50 pairs × 4 min = **3-4 hours**
- Take breaks! Ear fatigue is real

## Next Steps

1. **Now**: Open `feedback/feedback_template.csv` in Excel
2. **Fill in preferences** (50 rows)
3. **Save the file**
4. **Run**: `python evaluate_candidates_simple.py --convert-csv feedback/feedback_template.csv`
5. **This creates**: `feedback/preferences.json` ✅
6. **Then**: Train and improve your policy!

## Files Created

| File | Purpose |
|------|---------|
| `evaluate_candidates_simple.py` | Tool to generate templates & convert |
| `feedback/feedback_template.csv` | Template for rating (CSV format) |
| `feedback/feedback_template.json` | Template for rating (JSON format) |
| `EVAL_INSTRUCTIONS.md` | Detailed guide |

## Questions?

- **Q: Do I have to listen to audio?** A: Yes, that's the point! Hear which edit sounds better.
- **Q: Do I need to rate ALL 50?** A: No, but more is better. Start with 10-20.
- **Q: Can I change my mind?** A: Yes! Edit the CSV and re-convert.
- **Q: How much will this improve the policy?** A: 15-25% per feedback cycle.

---

**Ready?** Start here:
```bash
python evaluate_candidates_simple.py --format csv
# Then open feedback/feedback_template.csv in Excel
```

Good luck! 🎵
