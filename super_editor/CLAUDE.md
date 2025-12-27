# CLAUDE.md - Super Editor Implementation Guide

Complete implementation guide for the two-phase supervised audio editing system.

## Overview

**Goal**: Learn to automatically edit raw audio into polished tracks by:
1. First learning HOW to execute edits (reconstruction)
2. Then learning WHAT edits to make (prediction)

## Phase 1: Supervised Reconstruction

### Objective
Train an autoencoder that takes (raw_mel, edit_labels) and produces edited_mel.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           EditEncoder                                     │
│                                                                           │
│  raw_mel (B, T, 128) ──→ Linear(128, 512) ──┐                            │
│                                              ├──→ Concat ──→ Linear(640, 512)
│  edit_labels (B, T) ──→ Embedding(8, 128) ──┘                            │
│                                                                           │
│  ──→ PositionalEncoding ──→ TransformerEncoder(6 layers) ──→ latent      │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           MelDecoder                                      │
│                                                                           │
│  latent (B, T, 512) ──→ TransformerEncoder(3 layers) ──→ Linear(512, 128)│
│                                                              │            │
│  raw_mel (B, T, 128) ────────────────────────→ * α ─────────+            │
│                                                              │            │
│                                               edited_mel ◄───┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Edit Label Vocabulary

| Label | Value | Meaning |
|-------|-------|---------|
| CUT | 0 | Remove this beat entirely |
| KEEP | 1 | Keep this beat unchanged |
| LOOP | 2 | Repeat this beat |
| FADE_IN | 3 | Apply fade in |
| FADE_OUT | 4 | Apply fade out |
| EFFECT | 5 | Apply audio effect |
| TRANSITION | 6 | Crossfade region |
| PAD | 7 | Padding token |

### Loss Functions

```python
total_loss = (
    l1_weight * L1Loss(pred_mel, target_mel) +
    mse_weight * MSELoss(pred_mel, target_mel) +
    stft_weight * MultiScaleSTFTLoss(pred_mel, target_mel) +
    consistency_weight * EditConsistencyLoss(pred_mel, target_mel, edit_labels)
)
```

#### 1. L1 Reconstruction Loss
```python
def l1_loss(pred, target, mask=None):
    """Pixel-wise L1 on mel spectrogram."""
    loss = torch.abs(pred - target)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
    return loss.mean()
```

#### 2. Multi-Scale STFT Loss
```python
class MultiScaleSTFTLoss:
    """Frequency-aware loss at multiple resolutions."""

    fft_sizes = [512, 1024, 2048]
    hop_sizes = [128, 256, 512]
    win_sizes = [512, 1024, 2048]

    def forward(self, pred, target):
        total = 0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Spectral convergence
            pred_stft = torch.stft(pred.flatten(1), fft, hop, win)
            tgt_stft = torch.stft(target.flatten(1), fft, hop, win)
            sc = torch.norm(tgt_stft - pred_stft) / torch.norm(tgt_stft)

            # Log magnitude loss
            mag = F.l1_loss(torch.log(pred_stft.abs() + 1e-8),
                          torch.log(tgt_stft.abs() + 1e-8))
            total += sc + mag
        return total / len(self.fft_sizes)
```

#### 3. Edit Consistency Loss
```python
def edit_consistency_loss(pred, target, edit_labels):
    """Penalize changes where edit_labels == KEEP."""
    keep_mask = (edit_labels == 1).float().unsqueeze(-1)  # (B, T, 1)
    diff = torch.abs(pred - target) * keep_mask
    return diff.mean()
```

### Training Configuration

```python
@dataclass
class Phase1Config:
    # Model
    encoder_dim: int = 512
    decoder_dim: int = 512
    n_encoder_layers: int = 6
    n_decoder_layers: int = 3
    n_heads: int = 8
    dropout: float = 0.1

    # Audio
    n_mels: int = 128
    sample_rate: int = 22050
    hop_length: int = 512

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 8
    max_seq_len: int = 2048  # ~47 seconds at 22050/512
    gradient_clip: float = 1.0

    # Loss weights
    l1_weight: float = 1.0
    mse_weight: float = 0.0
    stft_weight: float = 1.0
    consistency_weight: float = 0.5

    # LR schedule
    warmup_steps: int = 1000
    total_steps: int = 100000
```

### Data Pipeline

```python
class PairedMelDataset(Dataset):
    """
    Loads paired (raw, edited) mel spectrograms with edit labels.

    Expected cache structure:
        cache/features/{pair_id}_raw.npz  -> {'mel': (T, 128), 'beat_times': ...}
        cache/features/{pair_id}_edit.npz -> {'mel': (T, 128), 'beat_times': ...}
        cache/labels/{pair_id}_labels.npy -> (T,) int array of edit labels
    """

    def __getitem__(self, idx):
        raw_mel = load_mel(self.raw_paths[idx])      # (T, 128)
        edit_mel = load_mel(self.edit_paths[idx])    # (T', 128)
        labels = load_labels(self.label_paths[idx])  # (T,)

        # Align lengths (edited may be shorter due to cuts)
        raw_mel, edit_mel, labels = self.align(raw_mel, edit_mel, labels)

        # Random crop to max_seq_len
        if len(raw_mel) > self.max_seq_len:
            start = random.randint(0, len(raw_mel) - self.max_seq_len)
            raw_mel = raw_mel[start:start + self.max_seq_len]
            edit_mel = edit_mel[start:start + self.max_seq_len]
            labels = labels[start:start + self.max_seq_len]

        return {
            'raw_mel': torch.from_numpy(raw_mel).float(),
            'edit_mel': torch.from_numpy(edit_mel).float(),
            'edit_labels': torch.from_numpy(labels).long(),
            'length': len(raw_mel),
        }
```

### Training Loop

```python
def train_phase1(config, data_dir, save_dir):
    # Setup
    model = ReconstructionModel(config).cuda()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    scaler = GradScaler()

    dataset = PairedMelDataset(data_dir, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        for batch in loader:
            optimizer.zero_grad()

            with autocast():
                pred_mel = model(batch['raw_mel'], batch['edit_labels'])
                loss = compute_losses(pred_mel, batch['edit_mel'], batch['edit_labels'], config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validation & checkpointing
        if epoch % 10 == 0:
            val_loss = validate(model, val_loader)
            save_checkpoint(model, optimizer, epoch, val_loss, save_dir)
```

### Success Criteria for Phase 1

| Metric | Target | How to Measure |
|--------|--------|----------------|
| L1 Loss | < 0.1 | Training metric |
| STFT Loss | < 1.0 | Training metric |
| Visual Quality | Good | Plot mel spectrograms |
| Audio Quality | Recognizable | Listen to reconstructions |
| KEEP Preservation | > 95% | Measure diff on KEEP regions |

---

## Phase 2: Edit Label Prediction with RL

### Objective
Train an RL agent to predict edit_labels that maximize output quality when passed through the frozen Phase 1 model.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         EditPredictor (Policy Network)                    │
│                                                                           │
│  raw_mel (B, T, 128) ──→ ConvEncoder ──→ TransformerEncoder ──→          │
│                                                              │            │
│                         ┌────────────────────────────────────┘            │
│                         ▼                                                 │
│                    Linear(512, 8) ──→ Softmax ──→ edit_probs (B, T, 8)   │
│                                                                           │
│  Sampling: edit_labels = Categorical(edit_probs).sample()                 │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Reward Computation                                │
│                                                                           │
│  1. Pass (raw_mel, predicted_labels) through FROZEN Phase 1 model        │
│  2. Compare output to target edited mel                                   │
│  3. Reward = -L1(pred_edit_mel, target_edit_mel) + bonuses               │
│                                                                           │
│  Bonuses:                                                                 │
│    - Duration match: reward if output length ≈ target length             │
│    - Label accuracy: reward if labels match ground truth                 │
│    - Smoothness: penalize rapid label changes                            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Reward Function

```python
def compute_reward(
    raw_mel: Tensor,           # (B, T, 128)
    pred_labels: Tensor,       # (B, T)
    target_mel: Tensor,        # (B, T', 128)
    target_labels: Tensor,     # (B, T)
    recon_model: nn.Module,    # Frozen Phase 1 model
) -> Tensor:
    """Compute reward for predicted edit labels."""

    # 1. Reconstruction quality (main reward)
    with torch.no_grad():
        pred_edit_mel = recon_model(raw_mel, pred_labels)

    # Align lengths for comparison
    min_len = min(pred_edit_mel.size(1), target_mel.size(1))
    recon_loss = F.l1_loss(
        pred_edit_mel[:, :min_len],
        target_mel[:, :min_len]
    )
    recon_reward = -recon_loss * 10  # Scale to reasonable range

    # 2. Label accuracy bonus
    label_match = (pred_labels == target_labels).float().mean(dim=1)
    label_reward = label_match * 5  # Up to +5 for perfect match

    # 3. Duration match bonus
    pred_keep_ratio = (pred_labels == 1).float().mean(dim=1)
    target_keep_ratio = (target_labels == 1).float().mean(dim=1)
    duration_diff = torch.abs(pred_keep_ratio - target_keep_ratio)
    duration_reward = (1 - duration_diff) * 3  # Up to +3 for perfect duration

    # 4. Smoothness penalty (penalize rapid changes)
    label_changes = (pred_labels[:, 1:] != pred_labels[:, :-1]).float().mean(dim=1)
    smoothness_penalty = -label_changes * 2  # Penalize excessive changes

    total_reward = recon_reward + label_reward + duration_reward + smoothness_penalty
    return total_reward
```

### PPO Training

```python
@dataclass
class Phase2Config:
    # Model
    predictor_dim: int = 256
    n_layers: int = 4
    n_heads: int = 4

    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5

    # Training
    batch_size: int = 32
    n_epochs_per_update: int = 4
    max_grad_norm: float = 0.5
```

```python
class EditPredictorAgent:
    """PPO agent for edit label prediction."""

    def __init__(self, config: Phase2Config, recon_model: nn.Module):
        self.policy = EditPredictor(config).cuda()
        self.value_net = ValueNetwork(config).cuda()
        self.recon_model = recon_model.eval()  # Frozen

        for p in self.recon_model.parameters():
            p.requires_grad = False

    def select_action(self, raw_mel: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample edit labels from policy."""
        logits = self.policy(raw_mel)  # (B, T, 8)
        dist = Categorical(logits=logits)
        actions = dist.sample()  # (B, T)
        log_probs = dist.log_prob(actions).sum(dim=1)  # (B,)
        entropy = dist.entropy().mean()
        return actions, log_probs, entropy

    def update(self, batch):
        """PPO update step."""
        raw_mel = batch['raw_mel']
        old_actions = batch['actions']
        old_log_probs = batch['log_probs']
        rewards = batch['rewards']
        advantages = batch['advantages']
        returns = batch['returns']

        for _ in range(self.config.n_epochs_per_update):
            # Policy loss
            logits = self.policy(raw_mel)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(old_actions).sum(dim=1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.value_net(raw_mel).squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = policy_loss + self.config.value_coeff * value_loss - self.config.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
```

### Training Loop for Phase 2

```python
def train_phase2(config, data_dir, recon_model_path, save_dir):
    # Load frozen reconstruction model
    recon_model = load_reconstruction_model(recon_model_path)

    # Setup agent
    agent = EditPredictorAgent(config, recon_model)

    # Dataset (same as Phase 1, but we only use raw_mel and target for reward)
    dataset = PairedMelDataset(data_dir, config)

    for epoch in range(config.epochs):
        # Collect rollouts
        rollout_buffer = []

        for batch in dataset.sample_batch(config.batch_size):
            raw_mel = batch['raw_mel'].cuda()
            target_mel = batch['edit_mel'].cuda()
            target_labels = batch['edit_labels'].cuda()

            # Sample actions from policy
            with torch.no_grad():
                actions, log_probs, _ = agent.select_action(raw_mel)
                values = agent.value_net(raw_mel).squeeze(-1)
                rewards = compute_reward(raw_mel, actions, target_mel, target_labels, recon_model)

            rollout_buffer.append({
                'raw_mel': raw_mel,
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
                'rewards': rewards,
            })

        # Compute advantages and returns
        batch = process_rollouts(rollout_buffer, config.gamma, config.gae_lambda)

        # PPO update
        agent.update(batch)

        # Logging
        if epoch % 10 == 0:
            avg_reward = batch['rewards'].mean().item()
            logger.info(f"Epoch {epoch} | Reward: {avg_reward:.2f}")
```

### Success Criteria for Phase 2

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Average Reward | > 0 | Training metric |
| Label Accuracy | > 70% | Compare to ground truth |
| Duration Match | < 10% error | Compare keep ratios |
| Audio Quality | Good | Listen to outputs |

---

## Data Preprocessing

### Creating Edit Labels from Paired Data

```python
def create_edit_labels(raw_audio_path: str, edit_audio_path: str) -> np.ndarray:
    """
    Infer edit labels by comparing raw and edited audio.

    Uses DTW alignment to find which beats were kept/cut.
    """
    # Load and extract features
    raw_mel = extract_mel(raw_audio_path)
    edit_mel = extract_mel(edit_audio_path)

    # Beat detection
    raw_beats = detect_beats(raw_audio_path)

    # DTW alignment
    alignment = dtw_align(raw_mel, edit_mel)

    # Determine label for each raw beat
    labels = np.ones(len(raw_beats), dtype=np.int64)  # Default: KEEP

    for i, beat_time in enumerate(raw_beats):
        beat_frame = int(beat_time * sr / hop_length)

        # Check if this beat was cut (no alignment match)
        if not has_alignment_match(alignment, beat_frame):
            labels[i] = 0  # CUT

        # Check for other edits (loops, fades, etc.)
        # ... additional detection logic

    return labels
```

### Mel Spectrogram Extraction

```python
def extract_mel(audio_path: str, config: AudioConfig) -> np.ndarray:
    """Extract mel spectrogram from audio file."""
    y, sr = librosa.load(audio_path, sr=config.sample_rate)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )

    # Log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_norm.T  # (T, n_mels)
```

---

## Inference Pipeline

```python
def edit_audio(
    input_path: str,
    edit_predictor: nn.Module,
    recon_model: nn.Module,
    vocoder: nn.Module,  # Optional: HiFi-GAN for waveform
    output_path: str,
):
    """Full inference pipeline."""

    # 1. Extract mel from input
    raw_mel = extract_mel(input_path)
    raw_mel_tensor = torch.from_numpy(raw_mel).float().unsqueeze(0).cuda()

    # 2. Predict edit labels
    with torch.no_grad():
        edit_logits = edit_predictor(raw_mel_tensor)
        edit_labels = edit_logits.argmax(dim=-1)  # Greedy decoding

    # 3. Reconstruct edited mel
    with torch.no_grad():
        edited_mel = recon_model(raw_mel_tensor, edit_labels)

    # 4. Convert to audio
    if vocoder is not None:
        # Neural vocoder
        audio = vocoder(edited_mel)
    else:
        # Griffin-Lim fallback
        audio = griffin_lim(edited_mel.squeeze().cpu().numpy())

    # 5. Save
    sf.write(output_path, audio, samplerate=22050)

    return {
        'edit_labels': edit_labels.squeeze().cpu().numpy(),
        'edited_mel': edited_mel.squeeze().cpu().numpy(),
    }
```

---

## Implementation Checklist

### Phase 1 Implementation
- [ ] `super_editor/config.py` - Configuration dataclasses
- [ ] `super_editor/models/encoder.py` - EditEncoder
- [ ] `super_editor/models/decoder.py` - MelDecoder
- [ ] `super_editor/losses/reconstruction.py` - L1, STFT losses
- [ ] `super_editor/losses/consistency.py` - Edit consistency loss
- [ ] `super_editor/data/dataset.py` - PairedMelDataset
- [ ] `super_editor/data/preprocessing.py` - Mel extraction, label creation
- [ ] `super_editor/trainers/phase1_trainer.py` - Training loop
- [ ] Validation and checkpointing
- [ ] TensorBoard logging

### Phase 2 Implementation
- [ ] `super_editor/models/edit_predictor.py` - Policy network
- [ ] `super_editor/models/value_network.py` - Value network
- [ ] `super_editor/trainers/phase2_trainer.py` - PPO training
- [ ] Reward function with frozen reconstruction model
- [ ] Rollout collection and GAE computation

### Inference
- [ ] `super_editor/inference/reconstruct.py` - Reconstruction only
- [ ] `super_editor/inference/full_pipeline.py` - End-to-end
- [ ] Optional: Integrate HiFi-GAN vocoder

---

## Training Schedule

### Phase 1
1. Start with small model (encoder_dim=256, n_layers=4)
2. Train for 50 epochs, check reconstructions visually
3. Scale up model if needed
4. Target: < 0.1 L1 loss on validation set

### Phase 2
1. Load best Phase 1 checkpoint, freeze it
2. Start with high entropy (exploration)
3. Decay entropy as policy converges
4. Target: > 0 average reward, > 70% label accuracy

---

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Reconstruction is blurry | Increase model capacity, add perceptual loss |
| KEEP regions change | Increase consistency_weight |
| Phase 2 reward stuck | Check reconstruction model is frozen, increase exploration |
| Gradient explosion | Reduce learning rate, increase gradient clipping |
| OOM errors | Reduce batch size, use gradient checkpointing |
