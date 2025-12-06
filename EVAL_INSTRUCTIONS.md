# How to Evaluate the Audio Candidates

You have **50 comparisons** to rate (5 songs × 10 pairs each).

## The Easiest Way: Use the CSV Template

### Step 1: Generate the Template
```bash
python evaluate_candidates_simple.py --format csv
```
This creates: `feedback/feedback_template.csv`

### Step 2: Open in Excel
Open `feedback/feedback_template.csv` in Excel or Google Sheets

You'll see:
```
song_id,candidate_a_id,candidate_b_id,a_keep_ratio,b_keep_ratio,preference,strength,reasoning
song_000,temp_0.10,temp_0.30,0.91,0.64,?,?,
song_000,temp_0.10,temp_0.50,0.91,0.43,?,?,
song_000,temp_0.10,temp_0.70,0.91,0.28,?,?,
...
```

### Step 3: Fill in Your Preferences

For each row:
1. **Listen to both candidates** (see audio files below)
2. **Pick a preference**:
   - `a` = Candidate A is better
   - `b` = Candidate B is better  
   - `tie` = About the same
3. **Rate the strength**:
   - `1` = Slight difference (almost equal)
   - `2` = Moderate (clear difference)
   - `3` = Strong (much better)
4. **Optional**: Add reasoning in the last column
   - "Tighter", "Better flow", "More energetic", etc.

Example of filled row:
```
song_000,temp_0.10,temp_0.30,0.91,0.64,a,2,Tighter edit, better flow
song_000,temp_0.10,temp_0.50,0.91,0.43,b,1,Slightly more energetic
```

### Step 4: Convert to Feedback Format

Once you've filled in the CSV:
```bash
python evaluate_candidates_simple.py --convert-csv feedback/feedback_template.csv
```

This creates: `feedback/preferences.json` ✅

## Alternative: Use the JSON Template

If you prefer editing JSON directly:

### Generate Template
```bash
python evaluate_candidates_simple.py --format json
```
Creates: `feedback/feedback_template.json`

### Fill It In
Open the JSON and change entries from:
```json
{
  "preference": None,
  "strength": None,
  "reasoning": ""
}
```

To:
```json
{
  "preference": "a",
  "strength": 2,
  "reasoning": "Tighter edit, better flow"
}
```

### Convert
```bash
python evaluate_candidates_simple.py --convert-json feedback/feedback_template.json
```

Creates: `feedback/preferences.json` ✅

## Understanding the Candidates

Each comparison shows:
- **candidate_a_id** / **candidate_b_id**: Temperature setting (0.1=conservative, 0.9=aggressive)
- **a_keep_ratio** / **b_keep_ratio**: % of beats kept (how tight)
  - 0.91 = 91% of beats kept (conservative, full song)
  - 0.06 = 6% of beats kept (aggressive, very tight)
- **temperature**: Raw setting (0.1, 0.3, 0.5, 0.7, 0.9)

## What You're Listening For

Listen for:
- **Tightness**: Does it feel well-paced or bloated?
- **Flow**: Do transitions feel natural?
- **Energy**: Is it engaging throughout?
- **Completeness**: Are musical phrases complete?

Don't worry about:
- Perfect audio quality (synthetic beats are simple)
- Absolute preferences (relative comparison is what matters)
- Consistency (it's okay if you change your mind)

## Time Estimate

- ~3-5 minutes per pair (listen + decide)
- 50 pairs = **2.5 to 4 hours total**
- Take breaks! Ear fatigue is real

## Next Steps After Evaluation

Once you have `feedback/preferences.json`:

```bash
# Train reward model on your preferences
python train_from_feedback.py --feedback feedback/preferences.json

# Fine-tune policy with your learned preferences
python train_rlhf_stable.py --episodes 500 \
  --reward_model models/reward_model_v9_feedback_final.pt
```

The AI will learn to match YOUR taste! 🎵

---

## Quick Reference: What to Fill

| Column | Values | Example |
|--------|--------|---------|
| preference | a, b, or tie | a |
| strength | 1, 2, or 3 | 2 |
| reasoning | Free text | "Tighter edit" |

That's it! Start with: `python evaluate_candidates_simple.py --format csv`
