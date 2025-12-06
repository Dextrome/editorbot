# Human Feedback Data Format

## Collection Template

Use this format to record human preference annotations:

### JSON Format (Recommended)

```json
[
  {
    "annotation_id": "ann_001",
    "song_id": "song_001",
    "song_path": "data/test_songs/song_001.wav",
    "original_duration_sec": 180,
    
    "edit_a": {
      "id": "a_temp_0.3",
      "policy_version": "v14",
      "temperature": 0.3,
      "beat_actions": [0, 0, 1, 0, 2, 0, ...],
      "n_beats_kept": 28,
      "n_beats_total": 45,
      "estimated_duration_sec": 168,
      "path": "eval_outputs/song_001_temp_0.3.wav"
    },
    
    "edit_b": {
      "id": "b_temp_0.7",
      "policy_version": "v14",
      "temperature": 0.7,
      "beat_actions": [0, 1, 0, 1, 2, ...],
      "n_beats_kept": 22,
      "n_beats_total": 45,
      "estimated_duration_sec": 132,
      "path": "eval_outputs/song_001_temp_0.7.wav"
    },
    
    "comparison": {
      "preference": "a",
      "strength": 2,
      "reasoning": "Edit A is tighter and more energetic without losing the hook"
    },
    
    "rater_info": {
      "rater_id": "annotator_001",
      "rater_name": "Alice",
      "musical_background": "Producer"
    },
    
    "metadata": {
      "timestamp": "2025-12-06T15:30:00Z",
      "duration_seconds_to_complete": 180,
      "confidence": 0.95,
      "notes": "Both are good but A keeps the vocal runs better"
    }
  },
  {
    "annotation_id": "ann_002",
    "song_id": "song_001",
    "edit_a": { ... },
    "edit_b": { ... },
    "comparison": {
      "preference": "tie",
      "strength": 0,
      "reasoning": "Both are equally good, just different vibes"
    },
    ...
  }
]
```

---

## Field Descriptions

### annotation_id
- **Type**: string
- **Required**: Yes
- **Description**: Unique identifier for this annotation
- **Example**: `"ann_001"`, `"feedback_20251206_001"`

### song_id
- **Type**: string
- **Required**: Yes
- **Description**: ID of the song being edited
- **Example**: `"song_001"`, `"my_favorite_song"`

### edit_a / edit_b
- **Type**: object
- **Required**: Yes
- **Description**: Metadata about each edit candidate
- **Fields**:
  - `id`: Candidate identifier (e.g., policy version + temperature)
  - `temperature`: Policy temperature used to generate edit (0.1-0.9)
  - `beat_actions`: List of action IDs for each beat [0=KEEP, 1=CUT, 2=LOOP, etc.]
  - `n_beats_kept`: Number of beats in final edit
  - `n_beats_total`: Number of beats in original
  - `estimated_duration_sec`: Approximate final duration
  - `path`: Path to audio file for listening

### preference
- **Type**: string
- **Required**: Yes
- **Allowed values**: `"a"` | `"b"` | `"tie"`
- **Description**: Which edit is preferred?
- **Example**: `"a"` means Edit A is better

### strength
- **Type**: integer
- **Required**: Yes
- **Allowed values**: 0 | 1 | 2 | 3
- **Mapping**:
  - `0`: Tie (equal quality)
  - `1`: Slightly better
  - `2`: Moderately better
  - `3`: Significantly better
- **Description**: How much better is the preferred edit?
- **Example**: `2` means preference is "moderate"

### reasoning
- **Type**: string
- **Required**: No
- **Description**: Why did you prefer this edit?
- **Example**: `"Tighter without losing the hook"`, `"Smoother transitions"`
- **Purpose**: Help understand model preferences

### rater_info
- **Type**: object
- **Required**: Recommended
- **Fields**:
  - `rater_id`: Unique identifier (for consistency tracking)
  - `rater_name`: Name of annotator
  - `musical_background`: e.g., "Producer", "Listener", "Engineer"
- **Purpose**: Track inter-rater agreement and bias

### timestamp
- **Type**: ISO 8601 string
- **Required**: Recommended
- **Format**: `"2025-12-06T15:30:00Z"`
- **Purpose**: Track when feedback was collected

### confidence
- **Type**: float
- **Required**: No
- **Range**: 0.0 - 1.0
- **Description**: How confident are you in this preference?
- **Example**: `0.95` means very confident, `0.6` means uncertain

---

## Batch File Example

### File: `feedback/preferences_batch_001.json`

```json
[
  {
    "annotation_id": "batch001_ann001",
    "song_id": "song_001",
    "edit_a": {
      "id": "v14_temp0.1",
      "temperature": 0.1,
      "path": "eval/song_001_v14_temp0.1.wav",
      "estimated_duration_sec": 180
    },
    "edit_b": {
      "id": "v14_temp0.5",
      "temperature": 0.5,
      "path": "eval/song_001_v14_temp0.5.wav",
      "estimated_duration_sec": 156
    },
    "comparison": {
      "preference": "b",
      "strength": 1,
      "reasoning": "Cut version is punchier"
    },
    "rater_info": {
      "rater_id": "alice",
      "musical_background": "Producer"
    },
    "timestamp": "2025-12-06T15:30:00Z",
    "confidence": 0.8
  },
  {
    "annotation_id": "batch001_ann002",
    "song_id": "song_001",
    "edit_a": {
      "id": "v14_temp0.3",
      "temperature": 0.3,
      "path": "eval/song_001_v14_temp0.3.wav",
      "estimated_duration_sec": 168
    },
    "edit_b": {
      "id": "v14_temp0.7",
      "temperature": 0.7,
      "path": "eval/song_001_v14_temp0.7.wav",
      "estimated_duration_sec": 132
    },
    "comparison": {
      "preference": "a",
      "strength": 2,
      "reasoning": "Moderately cut is better - too aggressive at 0.7"
    },
    "rater_info": {
      "rater_id": "bob",
      "musical_background": "Listener"
    },
    "timestamp": "2025-12-06T15:35:00Z",
    "confidence": 0.9
  }
]
```

---

## Collection Workflow

### Step 1: Generate Candidates
```bash
python generate_eval_candidates.py \
  --songs_dir data/test_songs \
  --output_dir eval_outputs \
  --n_songs 10 \
  --candidates_per_song 5
```

Creates files:
- `eval_outputs/evaluation_manifest.json` - Index of all tasks
- `eval_outputs/song_001/` - Candidate audio files
- `eval_outputs/song_002/` - etc.

### Step 2: Listen & Annotate

For each song, compare edit candidates:

```
🎵 SONG: song_001.wav (Original)
├─ 🎧 Play Original
│
├─ COMPARISON 1: Edit A vs Edit B
│  ├─ A (temp=0.1): [Play]  | Which is better? [A] [B] [Tie]
│  ├─ B (temp=0.5): [Play]  | Strength: [1] [2] [3]
│  └─ Notes: ________________
│
├─ COMPARISON 2: Edit A vs Edit C
│  └─ ...
│
└─ COMPARISON 3: Edit B vs Edit C
   └─ ...
```

### Step 3: Save Annotations

Save to `feedback/preferences_batch_001.json`

### Step 4: Train Model
```bash
python train_from_feedback.py \
  --feedback feedback/preferences_batch_001.json \
  --pretrained models/reward_model_v8_long/reward_model_final.pt \
  --epochs 20
```

### Step 5: Fine-tune Policy
```bash
python train_rlhf_stable.py \
  --reward_model models/reward_model_v9_feedback_final.pt \
  --episodes 500
```

---

## Quick Annotation Template

Use this CSV format for quick collection:

### File: `feedback/quick_annotations.csv`

```csv
song_id,edit_a_id,edit_b_id,preference,strength,notes,rater_id
song_001,temp_0.1,temp_0.5,b,1,Cut version punchier,alice
song_001,temp_0.3,temp_0.7,a,2,Too aggressive at 0.7,bob
song_001,temp_0.1,temp_0.9,tie,0,Different vibes same quality,charlie
song_002,temp_0.2,temp_0.6,a,3,Keeps vocals better,alice
song_002,temp_0.4,temp_0.8,b,2,Smoother transitions,bob
```

Convert to JSON:
```python
import pandas as pd
import json

df = pd.read_csv("feedback/quick_annotations.csv")
annotations = []

for _, row in df.iterrows():
    annotations.append({
        "song_id": row["song_id"],
        "edit_a": {"id": row["edit_a_id"]},
        "edit_b": {"id": row["edit_b_id"]},
        "comparison": {
            "preference": row["preference"],
            "strength": int(row["strength"]),
            "reasoning": row["notes"]
        },
        "rater_info": {"rater_id": row["rater_id"]}
    })

with open("feedback/preferences.json", "w") as f:
    json.dump(annotations, f, indent=2)
```

---

## Quality Guidelines

### For Raters

**Do**:
- ✓ Listen to both edits fully
- ✓ Consider overall flow and energy
- ✓ Think about what works musically
- ✓ Be consistent in preferences
- ✓ Record confidence level

**Don't**:
- ✗ Choose arbitrarily
- ✗ Rush through comparisons
- ✗ Get fatigued (take breaks!)
- ✗ Contradict yourself (track consistency)
- ✗ Let technical details override musicality

### For Collectors

**Track**:
- Number of annotations per rater
- Inter-rater agreement (do multiple raters agree?)
- Confidence levels (only train on confident feedback)
- Rater background (Producer vs Listener)

**Calculate**:
```python
def inter_rater_agreement(annotations_rater1, annotations_rater2):
    # For same song pairs, % agreement
    agreements = 0
    for pair1, pair2 in zip(annotations_rater1, annotations_rater2):
        if pair1["preference"] == pair2["preference"]:
            agreements += 1
    return agreements / len(annotations_rater1)
```

---

## Example: Full Feedback Cycle

### Batch 1 (Manual Feedback)
- 10 songs × 3 comparisons = 30 pairs
- 3 raters, Inter-rater agreement: 87%
- Training: `reward_model_v9`

### Batch 2 (Refined Feedback)
- 20 songs × 5 comparisons = 100 pairs
- 5 raters, Inter-rater agreement: 92%
- Training: `reward_model_v10`

### Batch 3 (Active Learning)
- Generate hardest cases from v10
- 15 songs × 2 comparisons = 30 pairs
- Train: `reward_model_v11`

### Result
- Policy improves each iteration
- Reward model better aligns with humans
- Edits more professional quality

---

## Tools for Collection

### Simple (Spreadsheet)
- Google Sheets with audio links
- CSV file with preferences
- Convert to JSON for training

### Better (Web Interface)
- Flask/FastAPI with audio player
- Side-by-side waveform visualization
- Real-time validation

### Professional (Annotation Platform)
- [Scale AI](https://scale.com)
- [Labelbox](https://labelbox.com)
- [Prodigy](https://prodi.gy)
- [Amazon Mechanical Turk](https://www.mturk.com)

---

## Validation

Before training, validate feedback:

```python
def validate_feedback(annotations):
    issues = []
    
    for ann in annotations:
        # Check required fields
        if "preference" not in ann["comparison"]:
            issues.append(f"{ann['annotation_id']}: missing preference")
        
        if ann["comparison"]["preference"] not in ["a", "b", "tie"]:
            issues.append(f"{ann['annotation_id']}: invalid preference")
        
        if ann["comparison"]["strength"] not in [0, 1, 2, 3]:
            issues.append(f"{ann['annotation_id']}: invalid strength")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        return False
    
    print(f"✓ All {len(annotations)} annotations valid")
    return True
```

---

## Success Metrics

Track across feedback iterations:

```
Iteration 1 (v8 synthetic)  → Policy Accuracy: 65%
Iteration 2 (v9 human)      → Policy Accuracy: 78% (+13%)
Iteration 3 (v10 human)     → Policy Accuracy: 85% (+7%)
Iteration 4 (v11 active)    → Policy Accuracy: 89% (+4%)
```

Target: 85%+ agreement with held-out human raters
