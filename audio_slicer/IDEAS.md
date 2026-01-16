# Audio Slicer - Ideas & Insights

## Core Insight: Why Previous Approaches Failed

### 1. Mel-to-Mel (Direct Transformation)
**Problem**: Tried to learn frame-to-frame mapping when raw/edited have different lengths.
- Raw: 10 minutes → Edited: 2 minutes (80% cut)
- Frame 100 of raw ≠ Frame 100 of edited
- Model learned garbage mapping

**Lesson**: Can't do 1:1 mapping when lengths differ.

### 2. Label-Based (super_editor)
**Problem**: Predefined DSP effects can't replicate human editing.
- DSP LOOP ≠ Human "loop this section"
- DSP FADE ≠ Human "smoothly transition here"
- Labels oversimplify complex edits

**Lesson**: Don't try to discretize continuous transformations.

### 3. RL/PPO (Phase 2)
**Problem**: Policy collapsed to single action, entropy died.
- Action space too large (20×5×5 = 500)
- Reward signal too sparse
- No good supervision signal

**Lesson**: RL struggles without dense reward or good initialization.

## Why FaceSwap-Style Might Work

### The Key Insight
FaceSwap doesn't predict "what to change" - it learns:
- What face A looks like (raw audio)
- What face B looks like (edited audio)
- How to transform content from A-style to B-style

### Applied to Audio
Instead of predicting "cut here, loop there", we learn:
- What raw audio sounds like (all takes, varied quality)
- What edited audio sounds like (best takes, consistent quality)
- How well content transforms between styles

### Selection by Transformation Quality
- Good content transforms cleanly (low cycle error)
- Bad content transforms poorly (high cycle error)
- This implicitly learns what a human editor would keep/cut

## Alternative Architectures to Try

### 1. VAE with KL Divergence
```python
class VAEEncoder:
    # Output mu and log_var
    # Sample z = mu + exp(log_var/2) * noise
    # KL loss forces latent to be Gaussian
    # Might help with generalization
```

### 2. VQ-VAE (Discrete Latent)
```python
class VQEncoder:
    # Quantize latent to discrete codebook
    # Forces encoder to learn discrete features
    # Could align with musical structure (beats, bars)
```

### 3. Transformer Autoencoder
```python
class TransformerAE:
    # Replace conv with self-attention
    # Better long-range dependencies
    # More expensive but potentially better
```

### 4. Multi-Scale Dual AE
```python
class MultiScaleDualAE:
    # Process at multiple time scales
    # Beat level (128 frames = 1.5s)
    # Bar level (512 frames = 6s)
    # Phrase level (2048 frames = 24s)
    # Combine scores across scales
```

## Scoring Methods to Try

### Current: Cycle Consistency
```
raw → encode → decode_edited → re-encode → decode_raw → compare to raw
```
**Pros**: Measures round-trip fidelity
**Cons**: Might be too forgiving

### Alternative 1: Direct Reconstruction Error
```
raw → encode → decode_edited → compare to ???
```
**Problem**: What's the target? We don't have aligned edited version.

### Alternative 2: Latent Distance
```
raw → encode → z_raw
edited_samples → encode → z_edited (average)
score = similarity(z_raw, z_edited_center)
```
**Idea**: Good raw segments should be close to edited cluster.

### Alternative 3: Discriminator Score
```
raw → encode → decode_edited → discriminator → "real edited" vs "fake edited"
```
**Idea**: Train discriminator to tell real edited from transformed raw.

### Alternative 4: Ensemble
```
score = 0.3 * cycle_consistency + 0.3 * latent_distance + 0.4 * discriminator
```
**Idea**: Combine multiple signals for robustness.

## Data Augmentation Ideas

### 1. Temporal Jitter
- Slight time stretching (0.95x - 1.05x)
- Makes encoder invariant to tempo variations

### 2. Pitch Shifting
- ±2 semitones
- Makes encoder invariant to pitch

### 3. Mix Augmentation
- Mix two segments at random ratio
- Forces encoder to disentangle content

### 4. Contrastive Pairs
- Same segment, different augmentations = positive pair
- Different segments = negative pair
- SimCLR-style contrastive learning

## Evaluation Ideas

### 1. Human Evaluation
- Show original, auto-edited, human-edited
- Rate preference

### 2. Keep Ratio vs Quality
- Plot % kept vs subjective quality
- Find optimal threshold

### 3. Segment Overlap with Human Edits
- Compare auto-selected segments to human-selected
- Measure IoU (intersection over union)

### 4. A/B Testing
- Auto-edit some tracks, human-edit others
- See if listeners can tell the difference

## Future Directions

### 1. Conditional Generation
Instead of just selecting, generate new content:
- Fill gaps between selected segments
- Generate transitions
- Create variations

### 2. Interactive Editing
- User adjusts threshold in real-time
- Preview before committing
- Fine-tune selections

### 3. Style Transfer
- Learn from multiple editors' styles
- Transfer style from one editor to another
- "Edit this like [professional editor]"

### 4. Multi-Track Editing
- Edit multiple instruments together
- Maintain musical coherence
- Stem-aware editing

## What We Tried & Results (January 2026)

### Approach 1: FaceSwap-Style Dual Autoencoder
**Idea**: Shared encoder + separate decoders for raw/edited. Score by cycle consistency.

**Result**: FAILED
- All segments scored similarly (0.81-0.89)
- Both decoders learned equally good reconstruction
- Cycle consistency doesn't discriminate when both styles reconstruct well

**Why it failed**: Unlike FaceSwap where Face A and Face B are visually distinct people, raw and edited audio share too much similarity (same melody, key, tempo). The only difference is structural cuts.

### Approach 2: Binary Classifier (edited=good, raw=bad)
**Idea**: Train classifier to distinguish edited from raw segments.

**Result**: FAILED
- 66% accuracy, but almost all segments scored >0.9
- Classifier learned "music-like features" not "edited vs unedited"

**Why it failed**: Edited audio is a subset of raw audio. "Good" content exists in both. The classifier can't discriminate because:
- Segments that appear in both datasets → ambiguous
- Only "cut" segments exist uniquely in raw → minority class

### Approach 3: Anomaly Detection (autoencoder on edited only)
**Idea**: Train autoencoder ONLY on edited (good) audio. Bad segments have high reconstruction error.

**Result**: SUCCESS
- Clear error distribution (0.004 - 0.028 MSE)
- 40th percentile threshold → 49% retention (close to actual 42%)
- Anomalies correctly identified at transitions and problematic areas

**Why it works**: The model only sees "good" audio. When presented with "bad" audio it hasn't seen, reconstruction error is higher. This is classic anomaly detection.

### Key Insight
The problem isn't "classify good vs bad" but "model the distribution of good, then detect outliers." This reframing made the approach work.

## Known Issues & Solutions

### Issue: Training Too Slow
**Cause**: Large segments, complex model
**Solutions**:
- Smaller segment_frames (64 instead of 128)
- Fewer hidden_dims
- Mixed precision (already enabled)

### Issue: Both Decoders Learn Same Thing
**Cause**: Not enough difference between raw/edited
**Solutions**:
- Stronger augmentation on raw only
- Add contrastive loss
- Explicit style embedding

### Issue: Scores All Similar
**Cause**: Model learned identity mapping
**Solutions**:
- More bottleneck compression
- Add noise to latent
- Stronger regularization

### Issue: Cuts Too Aggressive
**Cause**: Threshold too high, or model too strict
**Solutions**:
- Lower threshold
- Ensemble with other signals
- Minimum segment length constraint

---

## NEXT APPROACH: Pointer-Based Sequence Model (Planned)

### The Problem with Current Approach

The anomaly detection approach has a fundamental limitation: **it doesn't actually learn to edit**.

What it does:
- Model learns what "good" audio sounds like
- Scores segments by reconstruction error
- ALL editing logic is manual code (where to cut, min length, transitions)

What we want:
- Model learns WHERE humans cut
- Model learns WHAT to keep/repeat
- Model learns HOW to structure the edit

### The Pointer-Based Solution

Instead of predicting quality scores, predict a **sequence of pointers** into the raw audio:

```
Input:  Raw audio + Edited audio (as reference)
Output: Sequence of frame indices [1000, 1001, ..., 2000, 1000, 1001, ...]
```

The output sequence says "play frame 1000, then 1001, ... then jump back to 1000" - allowing for:
- Cutting (skip frames)
- Keeping (include frames)
- Looping/repeating (point to same frames multiple times)
- Reordering (change sequence)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    POINTER NETWORK                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Raw Audio ────→ [Encoder] ────→ Raw Embeddings (N frames) │
│                                        ↓                    │
│  Edited Audio ─→ [Encoder] ────→ Edit Embeddings           │
│                                        ↓                    │
│                              [Cross-Attention]              │
│                                        ↓                    │
│                              [Pointer Decoder]              │
│                                        ↓                    │
│                    Output: [ptr_1, ptr_2, ..., ptr_M]      │
│                    (sequence of indices into raw)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The cross-attention learns alignment: "which parts of raw appear in edited?"
The pointer decoder outputs the sequence of frame selections.

### Training

**Ground truth creation:**
1. Align raw ↔ edited audio (fingerprinting, cross-correlation, or let model learn)
2. For each frame in edited, find corresponding frame in raw
3. This gives us the target pointer sequence

**Loss function:**
- Cross-entropy on pointer predictions
- Optional: contrastive loss on alignments

**Key insight:** The model must implicitly learn alignment to solve the task.

### Limitations & Caveats

1. **Can only copy, not create**
   - Pointers select from existing raw audio
   - Cannot generate new content, effects, or transformations
   - No EQ, compression, pitch correction, fades
   - Workaround: Add a post-processing effects module

2. **Variable output length**
   - Model must decide when to stop generating
   - How does it know edit should be 2 min vs 5 min?
   - Solutions:
     - Special [STOP] token
     - Predict length first, then fill
     - Conditioning on target duration

3. **Long sequence challenge**
   - 10 min audio @ 86 frames/sec = ~52,000 frames
   - Pointing into 52k options is a huge search space
   - Attention is O(n²) - expensive for long sequences
   - Solutions:
     - Hierarchical attention (attend to segments, then frames)
     - Sparse attention patterns
     - Process in chunks with overlap
     - Downsample to beat/bar level first

4. **Training signal / alignment**
   - Need ground truth pointer sequences for supervised training
   - Requires aligning raw ↔ edited to create labels
   - Options:
     - Audio fingerprinting (Chromaprint, etc.)
     - Cross-correlation
     - Let model learn alignment end-to-end (harder)
     - DTW on spectrograms

5. **Multiple valid edits**
   - No single "correct" edit - different editors make different choices
   - Model may struggle with this ambiguity
   - Solutions:
     - Probabilistic outputs (sample different edits)
     - VAE-style latent for edit "style"
     - GAN training for diversity

6. **Autoregressive generation is slow**
   - Generating one pointer at a time
   - 10,000 output frames = 10,000 forward passes
   - Solutions:
     - Non-autoregressive decoding
     - Parallel decoding with refinement
     - Cache key/value states

7. **Musical coherence**
   - Must learn that random jumping sounds bad
   - Needs to understand phrases, structure, continuity
   - Solutions:
     - Multi-scale processing (frame + beat + bar + phrase)
     - Add structure prediction auxiliary task
     - Music-aware positional encoding

### Possible Improvements

1. **Hierarchical Pointers**
   ```
   Level 1: Select which bars to keep (coarse)
   Level 2: Select which beats within bars (medium)
   Level 3: Select exact frames within beats (fine)
   ```
   Reduces search space dramatically.

2. **Edit Operations Instead of Pointers**
   ```
   Output: [(COPY, start=1000, end=2000),
            (LOOP, start=500, end=600, times=2),
            (COPY, start=3000, end=4000)]
   ```
   More structured, captures editing intent.

3. **Conditioning on Style**
   - Provide example edited tracks as style reference
   - Model learns to edit "in the style of" the reference
   - Allows different editing styles without retraining

4. **Reinforcement Learning Fine-tuning**
   - Pre-train with supervised pointer prediction
   - Fine-tune with RL using audio quality rewards
   - Rewards: spectral continuity, beat alignment, listener preference

5. **Hybrid with Generation**
   - Pointer network for selection
   - Small generative model for transitions
   - Combines best of both: copy when possible, generate when needed

6. **Multi-track Awareness**
   - Process stems separately but jointly
   - Ensure cuts align across all instruments
   - Maintain mix coherence

### Implementation Plan

**Phase 1: Alignment**
- Implement audio fingerprinting to align raw ↔ edited
- Create ground truth pointer sequences from alignment
- Validate alignment quality on test pairs

**Phase 2: Basic Pointer Network**
- Encoder: CNN or Transformer on mel spectrograms
- Decoder: Autoregressive transformer with pointer attention
- Train on aligned pairs
- Evaluate: does output match edited audio?

**Phase 3: Improvements**
- Add hierarchical processing (bar → beat → frame)
- Add [STOP] token or length prediction
- Optimize for long sequences

**Phase 4: Evaluation**
- Compare to human edits (IoU of selected segments)
- Listening tests
- Ablation studies on architecture choices
