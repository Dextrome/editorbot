# Super Editor - Next Steps & Options

## Current Status (2026-01-15)

**Phase 1 (Reconstruction):** FULLY WORKING with hard-coded DSP effects
- KEEP (1): Perfect passthrough
- CUT (0): Perfect silence
- LOOP (2): Repeats previous frame throughout segment
- FADE_IN (3): Linear ramp 0→1 within segment
- FADE_OUT (4): Linear ramp 1→0 within segment
- EFFECT (5): Low-pass filter (reduces high frequencies)
- TRANSITION (6): Partial fade to 50%

**Implementation:** All effects are now deterministic DSP operations in `decoder.py`

**Phase 2 (RL Label Prediction):** Ready to train
- Previous attempt failed because Phase 1 didn't produce meaningful outputs
- With hard-coded effects, each action now produces predictable results
- RL agent should be able to learn which effects work best where

---

## Options for Moving Forward

### Option 1: Simplify to KEEP/CUT Only

**Approach:** Reduce label vocabulary to just 2 labels (KEEP=1, CUT=0). Phase 2 learns WHERE to cut.

**Pros:**
- Guaranteed to work - Phase 1 handles these perfectly
- Simple, interpretable
- Fast to train

**Cons:**
- Limited editing capability (only cuts, no effects)
- No fades, loops, or other creative edits

**Implementation:** Minimal changes - just use labels 0 and 1 only.

---

### Option 2: Supervised Label Prediction (Not RL)

**Approach:** Replace PPO with supervised classification. Train a model to predict edit labels directly from (raw_mel) using DTW-inferred labels as ground truth.

**Pros:**
- More stable than RL
- Faster convergence
- Direct supervision signal

**Cons:**
- Limited by quality of inferred labels (DTW alignment can be noisy)
- Less flexibility than RL
- Still relies on Phase 1 to execute the labels meaningfully

**Implementation:**
```python
class LabelPredictor(nn.Module):
    def forward(self, raw_mel):
        # Encode raw mel
        features = self.encoder(raw_mel)
        # Predict label distribution per frame
        logits = self.classifier(features)  # (B, T, n_labels)
        return logits

# Train with cross-entropy loss against DTW-inferred labels
loss = F.cross_entropy(pred_logits, target_labels)
```

---

### Option 3: Hard-Code Effects in Decoder (RECOMMENDED)

**Approach:** Implement FADE_IN, FADE_OUT, LOOP, etc. as deterministic DSP operations in the decoder, similar to how KEEP/CUT use hard overrides.

**Pros:**
- Effects are guaranteed to work correctly
- Phase 2 has meaningful, predictable actions to learn
- Interpretable - each label does exactly what it says
- Builds on what already works (hard overrides)

**Cons:**
- Less "learned" behavior, more hand-crafted
- May not capture subtle editing styles from training data
- Need to define each effect manually

**Implementation:**
```python
def forward(self, latent, raw_mel, edit_labels):
    # ... existing code ...

    # Hard overrides for ALL label types
    if edit_labels is not None:
        B, T, M = raw_mel.shape

        # CUT (0) -> silence
        cut_mask = (edit_labels == 0).unsqueeze(-1).float()

        # KEEP (1) -> passthrough
        keep_mask = (edit_labels == 1).unsqueeze(-1).float()

        # FADE_IN (3) -> multiply by ramp 0->1
        fade_in_mask = (edit_labels == 3).unsqueeze(-1).float()
        fade_in_ramp = torch.linspace(0, 1, T, device=raw_mel.device).view(1, T, 1)
        fade_in_output = raw_mel * fade_in_ramp

        # FADE_OUT (4) -> multiply by ramp 1->0
        fade_out_mask = (edit_labels == 4).unsqueeze(-1).float()
        fade_out_ramp = torch.linspace(1, 0, T, device=raw_mel.device).view(1, T, 1)
        fade_out_output = raw_mel * fade_out_ramp

        # LOOP (2) -> repeat previous frame (or small window)
        loop_mask = (edit_labels == 2).unsqueeze(-1).float()
        # Implementation: shift and repeat

        # EFFECT (5) -> apply simple effect (e.g., reduce high freq)
        effect_mask = (edit_labels == 5).unsqueeze(-1).float()
        # Could do: low-pass filter, reverb-like smear, etc.

        # Combine all
        silence = torch.zeros_like(raw_mel)
        output = (
            cut_mask * silence +
            keep_mask * raw_mel +
            fade_in_mask * fade_in_output +
            fade_out_mask * fade_out_output +
            loop_mask * loop_output +
            effect_mask * effect_output +
            other_mask * base_output  # fallback for unknown labels
        )

    return output
```

---

### Option 4: Direct Mel-to-Mel (No Labels)

**Approach:** Skip the label intermediate representation. Train an encoder-decoder that directly maps raw_mel -> edited_mel.

**Pros:**
- Simplest architecture
- End-to-end learning
- No label inference needed

**Cons:**
- Black box - no control over what edits happen
- Harder to guide/control output
- May just learn to copy input or produce average output

**Implementation:**
```python
class DirectEditor(nn.Module):
    def forward(self, raw_mel):
        latent = self.encoder(raw_mel)
        edited_mel = self.decoder(latent, raw_mel)  # with residual
        return edited_mel

# Train with L1 + perceptual loss
loss = l1_loss(pred_mel, target_mel) + perceptual_loss(pred_mel, target_mel)
```

---

## Recommendation: Option 3

**Why Option 3 has the best chance of success:**

1. **Proven pattern:** Hard overrides for KEEP/CUT work perfectly. Extending this to other labels follows the same reliable approach.

2. **Predictable behavior:** Each label does exactly one thing. No ambiguity, no learning failures.

3. **Phase 2 becomes tractable:** With deterministic effects, the RL agent has a clear action space. "Apply FADE_OUT here" always produces the same result, so the agent can learn what sounds good.

4. **Incremental improvement:** Start with simple effects (fades), verify they work, then add more complex ones (loop, effects).

5. **Fallback to model:** For labels without hard-coded effects, can still use the model's learned output.

**Risk with other options:**
- Option 1: Too limited for a real editor
- Option 2: Still depends on Phase 1 working for non-KEEP/CUT
- Option 4: Loses all interpretability and control

---

## Implementation Priority

1. Implement hard-coded FADE_IN and FADE_OUT (highest impact, easiest)
2. Test with manual labels to verify fades sound correct
3. Implement LOOP (repeat frames)
4. Train Phase 2 with the improved Phase 1
5. Evaluate and iterate

---

## Notes

- The rl_editor approach (behavioral cloning + PPO) works because it learns from actual edit patterns in data, not from a learned reconstruction model
- super_editor's two-phase approach is elegant in theory but requires Phase 1 to produce meaningful outputs for ALL labels
- Hard-coding effects is a pragmatic middle ground: deterministic where needed, learned where possible
