"""Train learned reward model from human preference pairs.

This implements RLHF-style reward model training:
1. Generate edit pairs from rollouts or saved edits
2. Collect human preferences (or simulate with heuristics for bootstrap)
3. Train Bradley-Terry reward model: P(a > b) = sigmoid(r_a - r_b)
4. Use learned reward for RL fine-tuning

Usage:
    # Bootstrap with automatic preferences (heuristic)
    python -m rl_editor.train_reward_model --data_dir ./training_data --mode bootstrap
    
    # Train from human preference file
    python -m rl_editor.train_reward_model --data_dir ./training_data --mode train --preferences ./preferences.json
    
    # Generate edit pairs for labeling
    python -m rl_editor.train_reward_model --data_dir ./training_data --mode generate --output ./pairs_to_label.json
"""

import argparse
import json
import logging
import pickle
import random
import time
try:
    import orjson
except ImportError:
    orjson = None
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from .config import Config, get_default_config
from .data import PairedAudioDataset
from .agent import Agent
from .state import AudioState, EditHistory, StateRepresentation
from .environment import AudioEditingEnv

logger = logging.getLogger(__name__)


@dataclass
class EditPair:
    """A pair of edits for preference comparison."""
    track_id: str
    edit_a: Dict[str, Any]  # {kept_beats, cut_beats, actions}
    edit_b: Dict[str, Any]
    preference: Optional[float] = None  # 1.0 = A better, 0.0 = B better, 0.5 = tie
    confidence: float = 1.0  # How confident the labeler is
    features_a: Optional[np.ndarray] = None
    features_b: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "edit_a": self.edit_a,
            "edit_b": self.edit_b,
            "preference": self.preference,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "EditPair":
        return cls(
            track_id=d["track_id"],
            edit_a=d["edit_a"],
            edit_b=d["edit_b"],
            preference=d.get("preference"),
            confidence=d.get("confidence", 1.0),
        )


class LearnedRewardModel(nn.Module):
    """Transformer-based reward model trained on preference pairs.
    
    Takes edit sequence features and predicts a scalar reward.
    Trained with Bradley-Terry loss on preference pairs.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_beats: int = 500,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection with LayerNorm for stability
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding (smaller initialization to avoid explosions)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_beats, hidden_dim) * 0.01)
        
        # Transformer encoder for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Edit action embedding (maps action taken to embedding)
        self.action_embed = nn.Embedding(10, hidden_dim // 4)  # 10 action types max
        
        # Aggregation and output with normalization
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        beat_features: torch.Tensor,
        action_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward for edit sequence.
        
        Args:
            beat_features: (batch, n_beats, input_dim) - per-beat audio features
            action_ids: (batch, n_beats) - action taken at each beat (0=KEEP, 1=CUT, etc.)
            mask: (batch, n_beats) - True for valid beats, False for padding
            
        Returns:
            (batch,) - scalar reward for each edit
        """
        batch_size, n_beats, _ = beat_features.shape
        
        # Project features and normalize
        x = self.input_proj(beat_features)  # (batch, n_beats, hidden)
        x = self.input_norm(x)  # Normalize for stability
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :n_beats, :]
        
        # Add action embeddings
        action_emb = self.action_embed(action_ids)  # (batch, n_beats, hidden//4)
        x = x + F.pad(action_emb, (0, self.hidden_dim - self.hidden_dim // 4))
        
        # Apply transformer (mask for padding)
        if mask is not None:
            # PyTorch transformer uses inverted mask (True = ignore)
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
            
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Aggregate: mean pool over valid positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)
        
        # Final reward prediction
        x = self.aggregator(x)
        reward = self.reward_head(x).squeeze(-1)
        
        # Clip reward to prevent explosion
        reward = torch.clamp(reward, min=-10.0, max=10.0)
        
        return reward


def compute_edit_aware_features(
    beat_features: np.ndarray,
    edit: Dict,
    n_beats: int,
) -> np.ndarray:
    """Compute features that encode both audio content AND edit decisions.
    
    Instead of just passing raw audio features (which are identical for both edits),
    we create edit-aware representations:
    - For KEPT beats: original features + keep indicator
    - For CUT beats: zeroed/masked features + cut indicator
    - Additional edit statistics as context
    
    This forces the model to learn from the actual edit pattern, not just audio.
    """
    feat_dim = beat_features.shape[1] if beat_features.ndim > 1 else 1
    kept_set = set(edit.get("kept_beats", []))
    
    # Output: original features + edit indicators (3 extra dims: keep, cut, position_in_output)
    output_dim = feat_dim + 4
    output = np.zeros((n_beats, output_dim), dtype=np.float32)
    
    kept_count = 0
    for i in range(min(n_beats, len(beat_features))):
        if i in kept_set:
            # Kept beat: include original features
            output[i, :feat_dim] = beat_features[i]
            output[i, feat_dim] = 1.0  # Keep indicator
            output[i, feat_dim + 1] = 0.0  # Cut indicator
            output[i, feat_dim + 2] = kept_count / max(len(kept_set), 1)  # Position in output
            kept_count += 1
        else:
            # Cut beat: mask features (scale down significantly)
            output[i, :feat_dim] = beat_features[i] * 0.1  # Attenuated, not zeroed
            output[i, feat_dim] = 0.0  # Keep indicator
            output[i, feat_dim + 1] = 1.0  # Cut indicator
            output[i, feat_dim + 2] = -1.0  # Not in output
        
        # Edit density in local window (8 beats)
        window_start = max(0, i - 4)
        window_end = min(n_beats, i + 4)
        window_kept = sum(1 for j in range(window_start, window_end) if j in kept_set)
        output[i, feat_dim + 3] = window_kept / max(window_end - window_start, 1)
    
    return output


class PreferenceDataset(Dataset):
    """Dataset of edit preference pairs for reward model training."""
    
    def __init__(
        self,
        pairs: List[EditPair],
        beat_feature_dim: int,
        max_beats: int = 500,
        use_edit_aware_features: bool = True,
    ):
        self.pairs = [p for p in pairs if p.preference is not None]
        self.beat_feature_dim = beat_feature_dim
        self.max_beats = max_beats
        self.use_edit_aware_features = use_edit_aware_features
        # Edit-aware features add 4 extra dimensions
        self.output_feature_dim = beat_feature_dim + 4 if use_edit_aware_features else beat_feature_dim
        
        logger.info(f"Loaded {len(self.pairs)} labeled preference pairs")
        logger.info(f"Using edit-aware features: {use_edit_aware_features} (dim: {self.output_feature_dim})")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        
        # Get raw features (already stored or compute)
        raw_feat_a = pair.features_a
        raw_feat_b = pair.features_b
        
        if raw_feat_a is None or raw_feat_b is None:
            # Fallback: use zeros if features not computed
            raw_feat_a = np.zeros((self.max_beats, self.beat_feature_dim))
            raw_feat_b = np.zeros((self.max_beats, self.beat_feature_dim))
        
        n_beats_a = min(len(raw_feat_a), self.max_beats)
        n_beats_b = min(len(raw_feat_b), self.max_beats)
        
        # Compute edit-aware features if enabled
        if self.use_edit_aware_features:
            feat_a = compute_edit_aware_features(raw_feat_a, pair.edit_a, n_beats_a)
            feat_b = compute_edit_aware_features(raw_feat_b, pair.edit_b, n_beats_b)
            out_dim = self.output_feature_dim
        else:
            feat_a = raw_feat_a[:n_beats_a]
            feat_b = raw_feat_b[:n_beats_b]
            out_dim = self.beat_feature_dim
        
        # Pad/truncate to max_beats
        padded_a = np.zeros((self.max_beats, out_dim), dtype=np.float32)
        padded_b = np.zeros((self.max_beats, out_dim), dtype=np.float32)
        padded_a[:n_beats_a] = feat_a[:n_beats_a]
        padded_b[:n_beats_b] = feat_b[:n_beats_b]
        
        # Create masks
        mask_a = np.zeros(self.max_beats, dtype=bool)
        mask_b = np.zeros(self.max_beats, dtype=bool)
        mask_a[:n_beats_a] = True
        mask_b[:n_beats_b] = True
        
        # Get action sequences (0=KEEP, 1=CUT, 2+=other)
        actions_a = self._get_action_sequence(pair.edit_a, n_beats_a)
        actions_b = self._get_action_sequence(pair.edit_b, n_beats_b)
        
        padded_actions_a = np.zeros(self.max_beats, dtype=np.int64)
        padded_actions_b = np.zeros(self.max_beats, dtype=np.int64)
        padded_actions_a[:n_beats_a] = actions_a
        padded_actions_b[:n_beats_b] = actions_b
        
        return {
            "features_a": torch.from_numpy(padded_a).float(),
            "features_b": torch.from_numpy(padded_b).float(),
            "actions_a": torch.from_numpy(padded_actions_a).long(),
            "actions_b": torch.from_numpy(padded_actions_b).long(),
            "mask_a": torch.from_numpy(mask_a),
            "mask_b": torch.from_numpy(mask_b),
            "preference": torch.tensor(pair.preference, dtype=torch.float32),
            "confidence": torch.tensor(pair.confidence, dtype=torch.float32),
        }
    
    def _get_action_sequence(self, edit: Dict, n_beats: int) -> np.ndarray:
        """Convert edit dict to action sequence."""
        actions = np.ones(n_beats, dtype=np.int64)  # Default: CUT (1)
        
        kept_beats = set(edit.get("kept_beats", []))
        for beat_idx in kept_beats:
            if beat_idx < n_beats:
                actions[beat_idx] = 0  # KEEP
        
        # Mark loops as action 2
        for loop in edit.get("loops", []):
            beat_idx = loop.get("beat_index", -1)
            if 0 <= beat_idx < n_beats:
                actions[beat_idx] = 2
        
        return actions


def compute_heuristic_preference(
    edit_a: Dict,
    edit_b: Dict,
    beat_features: np.ndarray,
    target_labels: np.ndarray,
    target_keep_ratio: float = 0.35,
) -> float:
    """Compute heuristic preference using ground truth labels.
    
    Returns 1.0 if A is better, 0.0 if B is better, 0.5 if tie.
    
    Heuristics:
    1. Agreement with target labels (main signal)
    2. Keep ratio closeness to target
    3. Beat quality of kept beats
    """
    def score_edit(edit: Dict) -> float:
        kept_beats = set(edit.get("kept_beats", []))
        n_beats = len(target_labels)
        
        # 1. Label agreement score (most important)
        agreement = 0.0
        for i in range(n_beats):
            target_keep = target_labels[i] >= 0.5
            actual_keep = i in kept_beats
            if target_keep == actual_keep:
                agreement += 1.0
        agreement /= max(n_beats, 1)
        
        # 2. Keep ratio score
        actual_ratio = len(kept_beats) / max(n_beats, 1)
        ratio_diff = abs(actual_ratio - target_keep_ratio)
        ratio_score = max(0, 1.0 - ratio_diff * 2)  # Penalty for deviation
        
        # 3. Quality of kept beats (prefer keeping high-label beats)
        quality_score = 0.0
        if kept_beats:
            for beat_idx in kept_beats:
                if beat_idx < len(target_labels):
                    quality_score += target_labels[beat_idx]
            quality_score /= len(kept_beats)
        
        # Combine: agreement is main signal
        total = 0.6 * agreement + 0.2 * ratio_score + 0.2 * quality_score
        return total
    
    score_a = score_edit(edit_a)
    score_b = score_edit(edit_b)
    
    # Convert to preference with SMALL margin (more decisive labels)
    diff = score_a - score_b
    if diff > 0.01:  # Much smaller margin for more decisive labels
        return 1.0
    elif diff < -0.01:
        return 0.0
    else:
        return 0.5


def generate_gt_comparison_pairs(
    config: Config,
    agent: Agent,
    dataset: PairedAudioDataset,
    n_rollouts_per_track: int = 50,
    temperature_range: Tuple[float, float] = (0.1, 5.0),
) -> List[EditPair]:
    """Generate MANY pairs by comparing agent rollouts to GROUND TRUTH.
    
    This creates clear preference signals by generating many diverse edits
    and creating ALL meaningful pairwise comparisons.
    
    Target: ~100+ pairs per track from 13 tracks = 1300+ pairs
    """
    pairs = []
    device = agent.device
    state_rep = StateRepresentation(config)
    
    for track_idx in range(len(dataset)):
        item = dataset[track_idx]
        raw_data = item["raw"]
        
        beat_times = raw_data["beat_times"].numpy()
        beat_features = raw_data["beat_features"].numpy()
        tempo = raw_data["tempo"].item()
        
        if beat_features.ndim > 1:
            state_rep.set_beat_feature_dim(beat_features.shape[1])
        
        target_labels = item.get("edit_labels")
        if target_labels is None:
            continue
        target_labels = target_labels.numpy()
        
        track_id = f"track_{track_idx}"
        n_beats = min(len(beat_times), 500)
        
        # Create ground truth edit
        gt_kept = [i for i in range(n_beats) if target_labels[i] >= 0.5]
        gt_cut = [i for i in range(n_beats) if target_labels[i] < 0.5]
        gt_edit = {"kept_beats": gt_kept, "cut_beats": gt_cut, "loops": [], "actions": [], "gt_agreement": 1.0, "source": "gt"}
        
        # Generate MANY diverse rollouts
        rollouts = []
        
        # 1. Policy rollouts with many temperatures (main diversity source)
        temperatures = list(np.linspace(temperature_range[0], temperature_range[1], n_rollouts_per_track))
        for temp in temperatures:
            edit = rollout_policy(
                agent, config, state_rep,
                beat_times[:n_beats],
                beat_features[:n_beats],
                tempo,
                temperature=temp,
            )
            edit["temperature"] = temp
            edit["source"] = "policy"
            rollouts.append(edit)
        
        # 2. High-temp rollouts (near random from policy)
        for temp in [10.0, 20.0, 50.0, 100.0]:
            for _ in range(3):  # Multiple samples at each high temp
                edit = rollout_policy(
                    agent, config, state_rep,
                    beat_times[:n_beats],
                    beat_features[:n_beats],
                    tempo,
                    temperature=temp,
                )
                edit["temperature"] = temp
                edit["source"] = "high_temp"
                rollouts.append(edit)
        
        # 3. Random edits with different keep ratios
        for keep_ratio in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
            for _ in range(3):  # Multiple samples per ratio
                random_kept = [i for i in range(n_beats) if random.random() < keep_ratio]
                random_edit = {
                    "kept_beats": random_kept,
                    "cut_beats": [i for i in range(n_beats) if i not in random_kept],
                    "loops": [], "actions": [], "temperature": 1000.0,
                    "source": f"random_{keep_ratio:.2f}",
                }
                rollouts.append(random_edit)
        
        # 4. Contiguous block edits (keep first N%, last N%, middle N%)
        for keep_frac in [0.2, 0.35, 0.5]:
            n_keep = int(n_beats * keep_frac)
            # First N%
            first_kept = list(range(n_keep))
            rollouts.append({
                "kept_beats": first_kept,
                "cut_beats": [i for i in range(n_beats) if i not in first_kept],
                "loops": [], "actions": [], "temperature": 1000.0,
                "source": f"first_{keep_frac:.0%}",
            })
            # Last N%
            last_kept = list(range(n_beats - n_keep, n_beats))
            rollouts.append({
                "kept_beats": last_kept,
                "cut_beats": [i for i in range(n_beats) if i not in last_kept],
                "loops": [], "actions": [], "temperature": 1000.0,
                "source": f"last_{keep_frac:.0%}",
            })
            # Middle N%
            start = (n_beats - n_keep) // 2
            middle_kept = list(range(start, start + n_keep))
            rollouts.append({
                "kept_beats": middle_kept,
                "cut_beats": [i for i in range(n_beats) if i not in middle_kept],
                "loops": [], "actions": [], "temperature": 1000.0,
                "source": f"middle_{keep_frac:.0%}",
            })
        
        # 5. Alternating patterns (every Nth beat)
        for skip in [2, 3, 4, 5]:
            alt_kept = [i for i in range(n_beats) if i % skip == 0]
            rollouts.append({
                "kept_beats": alt_kept,
                "cut_beats": [i for i in range(n_beats) if i not in alt_kept],
                "loops": [], "actions": [], "temperature": 1000.0,
                "source": f"every_{skip}th",
            })
        
        # 6. Inverted GT (opposite of ground truth - definitely bad)
        inverted_kept = [i for i in range(n_beats) if target_labels[i] < 0.5]
        rollouts.append({
            "kept_beats": inverted_kept,
            "cut_beats": [i for i in range(n_beats) if i not in inverted_kept],
            "loops": [], "actions": [], "temperature": 1000.0,
            "source": "inverted_gt",
        })
        
        # 7. Noisy GT (GT with random noise)
        for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for _ in range(2):
                noisy_kept = []
                for i in range(n_beats):
                    gt_keep = target_labels[i] >= 0.5
                    # Flip with probability noise_level
                    if random.random() < noise_level:
                        gt_keep = not gt_keep
                    if gt_keep:
                        noisy_kept.append(i)
                rollouts.append({
                    "kept_beats": noisy_kept,
                    "cut_beats": [i for i in range(n_beats) if i not in noisy_kept],
                    "loops": [], "actions": [], "temperature": 1000.0,
                    "source": f"noisy_gt_{noise_level:.0%}",
                })
        
        # Score ALL rollouts against GT
        for edit in rollouts:
            kept_set = set(edit["kept_beats"])
            score = sum(1 for i in range(n_beats) if (target_labels[i] >= 0.5) == (i in kept_set)) / n_beats
            edit["gt_agreement"] = score
        
        # Sort by GT agreement
        rollouts.sort(key=lambda x: x["gt_agreement"], reverse=True)
        
        # Count pairs before this track
        pairs_before = len(pairs)
        
        # === GENERATE LOTS OF PAIRS - MIX OF EASY, MEDIUM, AND HARD ===
        
        # Strategy 1: GT vs ALL other rollouts (GT is always best)
        for rollout in rollouts:
            if rollout["gt_agreement"] < 0.995:  # Exclude near-identical to GT
                score_diff = 1.0 - rollout["gt_agreement"]
                pairs.append(EditPair(
                    track_id=track_id,
                    edit_a=gt_edit,
                    edit_b=rollout,
                    preference=1.0,
                    confidence=min(1.0, score_diff * 2 + 0.3),  # Base confidence + scaled diff
                    features_a=beat_features[:n_beats].copy(),
                    features_b=beat_features[:n_beats].copy(),
                ))
        
        # Strategy 2: All pairwise comparisons with sufficient difference
        # This generates O(n^2) pairs but we filter by score difference
        for i in range(len(rollouts)):
            for j in range(i + 1, len(rollouts)):
                r1, r2 = rollouts[i], rollouts[j]
                score_diff = r1["gt_agreement"] - r2["gt_agreement"]
                
                # Include pairs with ANY meaningful difference (>0.02)
                if score_diff > 0.02:
                    # Confidence scales with difference - harder pairs get lower confidence
                    confidence = min(1.0, score_diff * 3 + 0.2)
                    pairs.append(EditPair(
                        track_id=track_id,
                        edit_a=r1, edit_b=r2,
                        preference=1.0,
                        confidence=confidence,
                        features_a=beat_features[:n_beats].copy(),
                        features_b=beat_features[:n_beats].copy(),
                    ))
                    # Also add reverse for balance
                    pairs.append(EditPair(
                        track_id=track_id,
                        edit_a=r2, edit_b=r1,
                        preference=0.0,
                        confidence=confidence,
                        features_a=beat_features[:n_beats].copy(),
                        features_b=beat_features[:n_beats].copy(),
                    ))
        
        pairs_this_track = len(pairs) - pairs_before
        logger.info(f"Track {track_idx + 1}/{len(dataset)}: Generated {pairs_this_track} pairs "
                   f"(total: {len(pairs)}) from {len(rollouts)} rollouts")
    
    return pairs


def generate_reference_pairs(
    config: Config,
    agent: Agent,
    dataset: PairedAudioDataset,
    n_rollouts_per_track: int = 30,
) -> List[EditPair]:
    """Generate pairs from REFERENCE tracks using audio quality heuristics.
    
    Reference tracks are professionally edited - no raw input available.
    We score edits by audio quality metrics instead of GT label agreement.
    
    Scoring heuristics:
    - Keep ratio closeness to target (~35%)
    - Phrase boundary alignment
    - Energy flow smoothness
    - Structural consistency
    """
    pairs = []
    state_rep = StateRepresentation(config)
    
    # Only process reference tracks
    reference_indices = [i for i in range(len(dataset)) if dataset.is_reference_track(i)]
    logger.info(f"Found {len(reference_indices)} reference tracks")
    
    for ref_idx, track_idx in enumerate(reference_indices):
        item = dataset[track_idx]
        raw_data = item["raw"]  # For reference, this IS the reference track
        
        beat_times = raw_data["beat_times"].numpy()
        beat_features = raw_data["beat_features"].numpy()
        tempo = raw_data["tempo"].item()
        
        if beat_features.ndim > 1:
            state_rep.set_beat_feature_dim(beat_features.shape[1])
        
        track_id = f"ref_track_{ref_idx}"
        n_beats = min(len(beat_times), 500)
        
        # Generate diverse edits using policy + random
        rollouts = []
        
        # Policy rollouts with various temperatures
        for temp in np.linspace(0.1, 5.0, n_rollouts_per_track):
            edit = rollout_policy(
                agent, config, state_rep,
                beat_times[:n_beats],
                beat_features[:n_beats],
                tempo,
                temperature=temp,
            )
            edit["temperature"] = temp
            edit["source"] = "policy"
            rollouts.append(edit)
        
        # Random edits with different keep ratios
        for keep_ratio in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
            for _ in range(2):
                random_kept = [i for i in range(n_beats) if random.random() < keep_ratio]
                rollouts.append({
                    "kept_beats": random_kept,
                    "cut_beats": [i for i in range(n_beats) if i not in random_kept],
                    "loops": [], "actions": [],
                    "temperature": 1000.0,
                    "source": f"random_{keep_ratio:.2f}",
                })
        
        # Contiguous block edits (bad examples)
        for keep_frac in [0.2, 0.35, 0.5]:
            n_keep = int(n_beats * keep_frac)
            rollouts.append({
                "kept_beats": list(range(n_keep)),
                "cut_beats": list(range(n_keep, n_beats)),
                "loops": [], "actions": [],
                "temperature": 1000.0,
                "source": f"first_{keep_frac:.0%}",
            })
        
        # Alternating patterns (bad)
        for skip in [2, 3, 4]:
            rollouts.append({
                "kept_beats": [i for i in range(n_beats) if i % skip == 0],
                "cut_beats": [i for i in range(n_beats) if i % skip != 0],
                "loops": [], "actions": [],
                "temperature": 1000.0,
                "source": f"every_{skip}th",
            })
        
        # === IMPROVEMENT 2: Harder negatives - subtle variations of good edits ===
        # These are much harder to distinguish from good edits
        
        # First, generate some baseline "good" policy edits at low temperature
        baseline_good_edits = []
        for temp in [0.3, 0.5, 0.7]:
            edit = rollout_policy(
                agent, config, state_rep,
                beat_times[:n_beats],
                beat_features[:n_beats],
                tempo,
                temperature=temp,
            )
            edit["source"] = "baseline_good"
            baseline_good_edits.append(edit)
        
        # Create subtle corruptions of good edits
        for base_edit in baseline_good_edits:
            kept_set = set(base_edit["kept_beats"])
            kept_list = list(kept_set)
            
            if len(kept_list) < 5:
                continue
            
            # Variant 1: Shift all beats by 1-2 positions (breaks phrase alignment)
            for shift in [1, 2, -1, -2]:
                shifted_kept = [(b + shift) % n_beats for b in kept_list]
                rollouts.append({
                    "kept_beats": shifted_kept,
                    "cut_beats": [i for i in range(n_beats) if i not in shifted_kept],
                    "loops": [], "actions": [],
                    "temperature": 1000.0,
                    "source": f"shifted_{shift}",
                })
            
            # Variant 2: Remove a few random beats (slightly wrong ratio)
            for n_remove in [2, 4, 6]:
                if len(kept_list) > n_remove + 5:
                    remove_indices = random.sample(range(len(kept_list)), n_remove)
                    partial_kept = [k for i, k in enumerate(kept_list) if i not in remove_indices]
                    rollouts.append({
                        "kept_beats": partial_kept,
                        "cut_beats": [i for i in range(n_beats) if i not in partial_kept],
                        "loops": [], "actions": [],
                        "temperature": 1000.0,
                        "source": f"partial_-{n_remove}",
                    })
            
            # Variant 3: Add a few extra random beats (slightly wrong ratio)
            cut_list = [i for i in range(n_beats) if i not in kept_set]
            for n_add in [2, 4, 6]:
                if len(cut_list) >= n_add:
                    add_beats = random.sample(cut_list, n_add)
                    expanded_kept = kept_list + add_beats
                    rollouts.append({
                        "kept_beats": expanded_kept,
                        "cut_beats": [i for i in range(n_beats) if i not in expanded_kept],
                        "loops": [], "actions": [],
                        "temperature": 1000.0,
                        "source": f"expanded_+{n_add}",
                    })
            
            # Variant 4: Break contiguity - swap some kept/cut beats
            for n_swap in [2, 4]:
                if len(kept_list) >= n_swap and len(cut_list) >= n_swap:
                    kept_to_cut = random.sample(kept_list, n_swap)
                    cut_to_keep = random.sample(cut_list, n_swap)
                    swapped_kept = [b for b in kept_list if b not in kept_to_cut] + cut_to_keep
                    rollouts.append({
                        "kept_beats": swapped_kept,
                        "cut_beats": [i for i in range(n_beats) if i not in swapped_kept],
                        "loops": [], "actions": [],
                        "temperature": 1000.0,
                        "source": f"swapped_{n_swap}",
                    })
            
            # Variant 5: Duplicate a section (mimics loop but breaks flow)
            if len(kept_list) >= 8:
                # Find a contiguous section and "double" it
                sorted_kept = sorted(kept_list)
                section_start = random.randint(0, len(sorted_kept) - 4)
                section = sorted_kept[section_start:section_start + 4]
                # Create duplicated pattern by keeping same beats twice conceptually
                # (since we can't actually duplicate, we extend the edit window)
                dup_kept = kept_list.copy()
                # Add nearby beats that would simulate extending the section
                for b in section:
                    for offset in [-8, 8]:
                        new_b = b + offset
                        if 0 <= new_b < n_beats and new_b not in dup_kept:
                            dup_kept.append(new_b)
                            break
                rollouts.append({
                    "kept_beats": dup_kept,
                    "cut_beats": [i for i in range(n_beats) if i not in dup_kept],
                    "loops": [], "actions": [],
                    "temperature": 1000.0,
                    "source": "section_dup",
                })
        
        # Score all edits using audio quality heuristics
        target_ratio = config.reward.target_keep_ratio
        for edit in rollouts:
            kept = set(edit["kept_beats"])
            n_kept = len(kept)
            
            # 1. Keep ratio score (most important for reference tracks)
            actual_ratio = n_kept / max(n_beats, 1)
            ratio_diff = abs(actual_ratio - target_ratio)
            ratio_score = max(0, 1.0 - ratio_diff * 3)  # Steep penalty for wrong ratio
            
            # 2. Phrase boundary score (prefer cuts at phrase boundaries)
            # Phrases are typically 8 or 16 beats
            phrase_score = 0.0
            if n_kept > 0:
                cut_points = []
                sorted_kept = sorted(kept)
                for i in range(1, len(sorted_kept)):
                    if sorted_kept[i] - sorted_kept[i-1] > 1:
                        cut_points.append(sorted_kept[i-1])
                        cut_points.append(sorted_kept[i])
                
                for cp in cut_points:
                    # Bonus for cuts at phrase boundaries
                    if cp % 8 == 0 or cp % 8 == 7:
                        phrase_score += 0.1
                    if cp % 16 == 0 or cp % 16 == 15:
                        phrase_score += 0.15
                phrase_score = min(1.0, phrase_score / max(len(cut_points), 1) * 5)
            
            # 3. Contiguity score (prefer keeping contiguous sections, not scattered beats)
            contiguity_score = 0.0
            if n_kept > 1:
                sorted_kept = sorted(kept)
                n_gaps = sum(1 for i in range(1, len(sorted_kept)) if sorted_kept[i] - sorted_kept[i-1] > 1)
                # Fewer gaps = better
                contiguity_score = max(0, 1.0 - n_gaps / max(n_kept / 4, 1))
            
            # 4. Energy consistency score (using beat features)
            energy_score = 0.5  # Default
            if n_kept > 2:
                kept_energies = [beat_features[i, 0] if i < len(beat_features) else 0 for i in sorted(kept)]
                if len(kept_energies) > 1:
                    # Lower variance in kept beats = more consistent
                    energy_var = np.var(kept_energies)
                    energy_score = max(0, 1.0 - energy_var * 2)
            
            # Combined score
            edit["quality_score"] = (
                0.4 * ratio_score + 
                0.25 * phrase_score + 
                0.2 * contiguity_score + 
                0.15 * energy_score
            )
        
        # Sort by quality score
        rollouts.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Create pairs
        pairs_before = len(pairs)
        
        # === GENERATE LOTS OF PAIRS - MIX OF EASY, MEDIUM, AND HARD ===
        
        # Strategy 1: All pairwise comparisons with sufficient difference
        for i in range(len(rollouts)):
            for j in range(i + 1, len(rollouts)):
                r1, r2 = rollouts[i], rollouts[j]
                score_diff = r1["quality_score"] - r2["quality_score"]
                
                # Include pairs with ANY meaningful difference (>0.02)
                if score_diff > 0.02:
                    # Confidence scales with difference
                    confidence = min(1.0, score_diff * 3 + 0.2)
                    pairs.append(EditPair(
                        track_id=track_id,
                        edit_a=r1, edit_b=r2,
                        preference=1.0,
                        confidence=confidence,
                        features_a=beat_features[:n_beats].copy(),
                        features_b=beat_features[:n_beats].copy(),
                    ))
                    # Also add reverse for balance
                    pairs.append(EditPair(
                        track_id=track_id,
                        edit_a=r2, edit_b=r1,
                        preference=0.0,
                        confidence=confidence,
                        features_a=beat_features[:n_beats].copy(),
                        features_b=beat_features[:n_beats].copy(),
                    ))
        
        pairs_this_track = len(pairs) - pairs_before
        logger.info(f"Reference {ref_idx + 1}/{len(reference_indices)}: Generated {pairs_this_track} pairs "
                   f"(total: {len(pairs)})")
    
    return pairs


def generate_edit_pairs_from_rollouts(
    config: Config,
    agent: Agent,
    dataset: PairedAudioDataset,
    n_pairs_per_track: int = 5,
    temperature_range: Tuple[float, float] = (0.5, 2.0),
) -> List[EditPair]:
    """Generate edit pairs by rolling out policy with different temperatures.
    
    Uses temperature sampling to create diverse edits from same track.
    """
    pairs = []
    device = agent.device
    
    state_rep = StateRepresentation(config)
    
    for track_idx in range(len(dataset)):
        item = dataset[track_idx]
        raw_data = item["raw"]
        
        beat_times = raw_data["beat_times"].numpy()
        beat_features = raw_data["beat_features"].numpy()
        tempo = raw_data["tempo"].item()
        
        # Get actual feature dim
        if beat_features.ndim > 1:
            state_rep.set_beat_feature_dim(beat_features.shape[1])
        
        target_labels = item.get("edit_labels")
        if target_labels is not None:
            target_labels = target_labels.numpy()
        
        track_id = f"track_{track_idx}"
        n_beats = min(len(beat_times), 500)
        
        # Generate multiple edits with different temperatures
        edits = []
        for temp in np.linspace(temperature_range[0], temperature_range[1], n_pairs_per_track * 2):
            edit = rollout_policy(
                agent, config, state_rep,
                beat_times[:n_beats],
                beat_features[:n_beats],
                tempo,
                temperature=temp,
            )
            edit["temperature"] = temp
            edits.append(edit)
        
        # Create pairs
        random.shuffle(edits)
        for i in range(0, len(edits) - 1, 2):
            edit_a = edits[i]
            edit_b = edits[i + 1]
            
            # Compute heuristic preference if labels available
            preference = None
            if target_labels is not None:
                preference = compute_heuristic_preference(
                    edit_a, edit_b,
                    beat_features[:n_beats],
                    target_labels[:n_beats],
                    config.reward.target_keep_ratio,
                )
            
            pair = EditPair(
                track_id=track_id,
                edit_a=edit_a,
                edit_b=edit_b,
                preference=preference,
                features_a=beat_features[:n_beats].copy(),
                features_b=beat_features[:n_beats].copy(),  # Same track, same features
            )
            pairs.append(pair)
        
        logger.info(f"Generated {len(pairs)} pairs from track {track_idx + 1}/{len(dataset)}")
    
    return pairs


def rollout_policy(
    agent: Agent,
    config: Config,
    state_rep: StateRepresentation,
    beat_times: np.ndarray,
    beat_features: np.ndarray,
    tempo: float,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Roll out policy on a track with given temperature.
    
    Returns edit dict with kept_beats, cut_beats, etc.
    """
    device = agent.device
    n_beats = len(beat_times)
    total_duration = beat_times[-1] if len(beat_times) > 0 else 1.0
    
    kept_beats = []
    cut_beats = []
    loops = []
    actions = []
    
    edit_history = EditHistory()
    
    for beat_idx in range(n_beats):
        # Construct state
        audio_state = AudioState(
            beat_index=beat_idx,
            beat_times=beat_times,
            beat_features=beat_features,
            tempo=tempo,
        )
        
        # Compute remaining duration
        kept_duration = sum(
            beat_times[i+1] - beat_times[i] if i+1 < len(beat_times) else 0
            for i in edit_history.kept_beats
        )
        target_duration = total_duration * config.reward.target_keep_ratio
        remaining = max(0.0, target_duration - kept_duration)
        
        obs = state_rep.construct_observation(
            audio_state, edit_history,
            remaining_duration=remaining,
            total_duration=total_duration,
        )
        
        # Get action with temperature
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = agent.policy_net(obs_tensor, temperature=temperature)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        actions.append(action)
        
        # Apply action (simplified: 0=KEEP, 1=CUT, 2-4=LOOP variants, 5-8=REORDER)
        if action == 0:
            kept_beats.append(beat_idx)
            edit_history.add_keep(beat_idx)
        elif action == 1:
            cut_beats.append(beat_idx)
            edit_history.add_cut(beat_idx)
        elif 2 <= action <= 4:
            kept_beats.append(beat_idx)
            loops.append({"beat_index": beat_idx, "n_times": action - 1})
            edit_history.add_keep(beat_idx)
        else:
            # REORDER - treat as KEEP for now
            kept_beats.append(beat_idx)
            edit_history.add_keep(beat_idx)
    
    return {
        "kept_beats": kept_beats,
        "cut_beats": cut_beats,
        "loops": loops,
        "actions": actions,
    }


def train_reward_model(
    model: LearnedRewardModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 200,
    lr: float = 1e-4,
    device: torch.device = None,
    save_path: Optional[Path] = None,
    target_accuracy: float = 0.90,  # Stop early if we hit this
    patience: int = 100,  # Early stopping patience (increased to allow longer training)
) -> Dict[str, float]:
    """Train reward model on preference pairs.
    
    Uses Bradley-Terry loss: P(A > B) = sigmoid(r_A - r_B)
    Trains until target_accuracy is reached or n_epochs completed.
    """
    from tqdm import tqdm
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Starting training on {device}, {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0
    
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            feat_a = batch["features_a"].to(device)
            feat_b = batch["features_b"].to(device)
            actions_a = batch["actions_a"].to(device)
            actions_b = batch["actions_b"].to(device)
            mask_a = batch["mask_a"].to(device)
            mask_b = batch["mask_b"].to(device)
            preference = batch["preference"].to(device)
            confidence = batch["confidence"].to(device)
            
            # Forward pass
            reward_a = model(feat_a, actions_a, mask_a)
            reward_b = model(feat_b, actions_b, mask_b)
            
            # Bradley-Terry loss: P(A > B) = sigmoid(r_A - r_B)
            logits = reward_a - reward_b
            loss = F.binary_cross_entropy_with_logits(
                logits, preference,
                weight=confidence,
            )
            
            optimizer.zero_grad()
            loss.backward()
            # Stricter gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Check for NaN/Inf and stop early if detected
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at epoch {epoch}, batch {train_total//len(preference)}")
                logger.error(f"Loss value: {loss.item()}")
                raise RuntimeError(f"Training diverged at epoch {epoch}")
            
            train_loss += loss.item() * len(preference)
            predictions = (logits > 0).float()
            targets = (preference > 0.5).float()
            train_correct += (predictions == targets).sum().item()
            train_total += len(preference)
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), acc=train_correct/max(train_total,1))
        
        scheduler.step()
        
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for batch in val_pbar:
                feat_a = batch["features_a"].to(device)
                feat_b = batch["features_b"].to(device)
                actions_a = batch["actions_a"].to(device)
                actions_b = batch["actions_b"].to(device)
                mask_a = batch["mask_a"].to(device)
                mask_b = batch["mask_b"].to(device)
                preference = batch["preference"].to(device)
                
                reward_a = model(feat_a, actions_a, mask_a)
                reward_b = model(feat_b, actions_b, mask_b)
                
                logits = reward_a - reward_b
                predictions = (logits > 0).float()
                targets = (preference > 0.5).float()
                val_correct += (predictions == targets).sum().item()
                val_total += len(preference)
                
                val_pbar.set_postfix(acc=val_correct/max(val_total,1))
        
        val_acc = val_correct / max(val_total, 1)
        
        # Log every epoch (not just every 10)
        logger.info(
            f"Epoch {epoch:3d}: Loss={train_loss:.4f}, "
            f"Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%"
        )
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            epochs_without_improvement = 0
            
            if save_path:
                torch.save({
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_accuracy": best_val_acc,
                }, save_path / "reward_model_best.pt")
        else:
            epochs_without_improvement += 1
        
        # Early stopping: target reached or no improvement
        # Skip accuracy check if target_accuracy > 1.0 (train until convergence)
        if target_accuracy <= 1.0 and val_acc >= target_accuracy:
            logger.info(f"Target accuracy {target_accuracy*100:.0f}% reached at epoch {epoch}!")
            break
        
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping: no improvement for {patience} epochs")
            break
    
    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    logger.info(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    
    return {
        "best_val_accuracy": best_val_acc,
        "final_train_loss": train_loss,
    }


def bootstrap_reward_model(
    config: Config,
    data_dir: str,
    save_dir: str,
    n_pairs_per_track: int = 50,  # Policy rollouts per track
    n_epochs: int = 100,  # More epochs for larger dataset
    agent_checkpoint: Optional[str] = None,
    include_reference: bool = True,  # Also generate pairs from reference tracks
) -> LearnedRewardModel:
    """Bootstrap reward model using heuristic preferences.
    
    1. Load agent (or use random policy)
    2. Generate MANY edit pairs with diverse strategies
       - From paired tracks: use GT label agreement for scoring
       - From reference tracks: use audio quality heuristics for scoring
    3. Train reward model on combined preference data
    
    With 13 paired tracks + 33 reference tracks, expect ~5000+ pairs.
    """
    logger.info("=" * 60)
    logger.info("Bootstrapping Reward Model (Comprehensive)")
    logger.info("=" * 60)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    
    # Load dataset WITH reference tracks
    dataset = PairedAudioDataset(data_dir, config, include_reference=include_reference)
    n_paired = dataset.get_num_paired()
    n_reference = dataset.get_num_reference()
    logger.info(f"Loaded {n_paired} paired tracks + {n_reference} reference tracks = {len(dataset)} total")
    
    # Get feature dimension from first sample
    first_item = dataset[0]
    beat_features = first_item["raw"]["beat_features"].numpy()
    beat_feature_dim = beat_features.shape[1] if beat_features.ndim > 1 else 109
    logger.info(f"Beat feature dim: {beat_feature_dim}")
    
    # Initialize agent
    state_rep = StateRepresentation(config)
    state_rep.set_beat_feature_dim(beat_feature_dim)
    input_dim = state_rep.feature_dim
    
    agent = Agent(config, input_dim=input_dim, n_actions=9)
    
    if agent_checkpoint and Path(agent_checkpoint).exists():
        logger.info(f"Loading agent from {agent_checkpoint}")
        ckpt = torch.load(agent_checkpoint, map_location=device, weights_only=False)
        agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
    else:
        logger.info("Using randomly initialized agent")
    
    all_pairs = []
    
    # 1. Generate pairs from PAIRED tracks (with GT labels)
    if n_paired > 0:
        # Create a dataset view with only paired tracks
        paired_dataset = PairedAudioDataset(data_dir, config, include_reference=False)
        logger.info(f"\n--- Generating pairs from {n_paired} PAIRED tracks (GT labels) ---")
        paired_pairs = generate_gt_comparison_pairs(
            config, agent, paired_dataset,
            n_rollouts_per_track=n_pairs_per_track,
            temperature_range=(0.1, 5.0),
        )
        logger.info(f"Generated {len(paired_pairs)} pairs from paired tracks")
        all_pairs.extend(paired_pairs)
    
    # 2. Generate pairs from REFERENCE tracks (quality heuristics)
    if include_reference and n_reference > 0:
        logger.info(f"\n--- Generating pairs from {n_reference} REFERENCE tracks (quality heuristics) ---")
        reference_pairs = generate_reference_pairs(
            config, agent, dataset,
            n_rollouts_per_track=30,  # Fewer rollouts since no GT
        )
        logger.info(f"Generated {len(reference_pairs)} pairs from reference tracks")
        all_pairs.extend(reference_pairs)
    
    logger.info(f"\n=== TOTAL: {len(all_pairs)} edit pairs ===")
    
    # Filter pairs with preferences
    labeled_pairs = [p for p in all_pairs if p.preference is not None]
    logger.info(f"Pairs with labels: {len(labeled_pairs)}")
    
    # Cap pairs at max_pairs for reasonable training time
    max_pairs = 40000
    if len(labeled_pairs) > max_pairs:
        logger.info(f"Sampling {max_pairs} pairs from {len(labeled_pairs)} (stratified by confidence)")
        # Stratified sampling: keep mix of easy (high conf) and hard (low conf) pairs
        labeled_pairs.sort(key=lambda p: p.confidence)
        # Take equal parts from each quartile
        quarter = len(labeled_pairs) // 4
        samples_per_quarter = max_pairs // 4
        sampled = []
        for q in range(4):
            start = q * quarter
            end = start + quarter if q < 3 else len(labeled_pairs)
            quartile_pairs = labeled_pairs[start:end]
            sampled.extend(random.sample(quartile_pairs, min(samples_per_quarter, len(quartile_pairs))))
        labeled_pairs = sampled
        random.shuffle(labeled_pairs)
        logger.info(f"Sampled to {len(labeled_pairs)} pairs")
    
    if len(labeled_pairs) < 10:
        logger.error("Not enough labeled pairs for training!")
        return None
    
    # Save pairs for reference
    pairs_file = save_path / "bootstrap_pairs.json"
    with open(pairs_file, "w") as f:
        json.dump([p.to_dict() for p in labeled_pairs], f, indent=2)
    logger.info(f"Saved pairs to {pairs_file}")
    
    # Create dataset and loaders
    # Edit-aware features add 4 extra dimensions
    use_edit_aware = True
    pref_dataset = PreferenceDataset(
        labeled_pairs, beat_feature_dim, 
        use_edit_aware_features=use_edit_aware
    )
    effective_feature_dim = pref_dataset.output_feature_dim
    logger.info(f"Feature dim: {beat_feature_dim} base + 4 edit = {effective_feature_dim} total")
    
    n_val = max(int(0.15 * len(pref_dataset)), 1)
    n_train = len(pref_dataset) - n_val
    
    train_dataset, val_dataset = random_split(pref_dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Initialize reward model with edit-aware feature dimension
    reward_model = LearnedRewardModel(
        input_dim=effective_feature_dim,
        hidden_dim=256,
        n_layers=3,
        n_heads=4,
        dropout=0.1,
    )
    
    # Train
    metrics = train_reward_model(
        reward_model, train_loader, val_loader,
        n_epochs=n_epochs,
        lr=1e-4,
        device=device,
        save_path=save_path,
    )
    
    # Save final model
    final_path = save_path / "reward_model_final.pt"
    torch.save({
        "model_state_dict": reward_model.state_dict(),
        "config": {
            "input_dim": effective_feature_dim,
            "base_feature_dim": beat_feature_dim,
            "use_edit_aware_features": use_edit_aware,
            "hidden_dim": 256,
            "n_layers": 3,
            "n_heads": 4,
        },
        "metrics": metrics,
    }, final_path)
    logger.info(f"Saved final reward model to {final_path}")
    
    return reward_model


def train_from_cached_pairs(
    config: Config,
    cached_pairs_path: str,
    save_dir: str,
    n_epochs: int = 200,
    max_pairs: int = 40000,
    resume_from: Optional[str] = None,
    lr: float = 1e-5,
    target_accuracy: float = 0.90,
) -> LearnedRewardModel:
    """Train reward model from cached pairs JSON (skip pair generation!).
    
    This is MUCH faster since we don't need to:
    - Load audio files
    - Initialize agent
    - Run policy rollouts
    - Generate pairs
    
    Just load JSON -> create dataset -> train!
    
    Args:
        config: Config object
        cached_pairs_path: Path to bootstrap_pairs.json
        save_dir: Directory to save checkpoints
        n_epochs: Max epochs to train
        max_pairs: Max pairs to sample (None = use all)
        resume_from: Path to checkpoint to resume from (uses lower LR by default)
        lr: Learning rate (default 1e-5 for resume, 1e-4 for fresh start)
    """
    logger.info("=" * 60)
    logger.info("Training from Cached Pairs (Fast Mode)")
    logger.info("=" * 60)
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
    logger.info("=" * 60)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    
    # Load cached pairs
    logger.info(f"Loading pairs from {cached_pairs_path}")
    start_load = time.time()
    
    # Try to use pickle format first (much faster than JSON for large objects)
    pickle_path = cached_pairs_path.replace(".json", ".pkl")
    if Path(pickle_path).exists():
        logger.info(f"Loading from pickle cache: {pickle_path}")
        with open(pickle_path, "rb") as f:
            labeled_pairs = pickle.load(f)
    elif cached_pairs_path.endswith(".json"):
        # Load JSON with orjson (much faster than json.load)
        logger.info("Using fast JSON loading...")
        if orjson:
            with open(cached_pairs_path, "rb") as f:
                pairs_data = orjson.loads(f.read())
        else:
            with open(cached_pairs_path, "r") as f:
                pairs_data = json.load(f)
        
        logger.info(f"Loaded JSON, converting {len(pairs_data)} pairs...")
        labeled_pairs = [EditPair.from_dict(d) for d in pairs_data]
        
        # Save as pickle for next time
        logger.info(f"Saving pickle cache: {pickle_path}")
        with open(pickle_path, "wb") as f:
            pickle.dump(labeled_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unknown cache format: {cached_pairs_path}")
    
    load_time = time.time() - start_load
    logger.info(f"Loaded {len(labeled_pairs)} pairs from cache in {load_time:.1f}s")
    
    # Sample if too many
    if max_pairs and len(labeled_pairs) > max_pairs:
        logger.info(f"Sampling {max_pairs} pairs from {len(labeled_pairs)}")
        labeled_pairs.sort(key=lambda p: p.confidence)
        quarter = len(labeled_pairs) // 4
        samples_per_quarter = max_pairs // 4
        sampled = []
        for q in range(4):
            start = q * quarter
            end = start + quarter if q < 3 else len(labeled_pairs)
            quartile_pairs = labeled_pairs[start:end]
            sampled.extend(random.sample(quartile_pairs, min(samples_per_quarter, len(quartile_pairs))))
        labeled_pairs = sampled
        random.shuffle(labeled_pairs)
        logger.info(f"Sampled to {len(labeled_pairs)} pairs")
    
    # Infer beat feature dim from first pair
    first_edit = labeled_pairs[0].edit_a
    # For cached pairs, features_a/b are None, so we need a default
    # The edit dict has kept_beats which tells us track length
    n_beats = max(
        max(first_edit.get("kept_beats", [0])) if first_edit.get("kept_beats") else 0,
        max(first_edit.get("cut_beats", [0])) if first_edit.get("cut_beats") else 0,
    ) + 1
    
    # Use default beat_feature_dim (we don't have actual features in cache)
    # The edit-aware features will be computed from kept_beats/cut_beats masks
    beat_feature_dim = 121  # Default from full features mode
    
    # Create dataset
    use_edit_aware = True
    pref_dataset = PreferenceDataset(
        labeled_pairs, beat_feature_dim,
        use_edit_aware_features=use_edit_aware
    )
    effective_feature_dim = pref_dataset.output_feature_dim
    logger.info(f"Using edit-aware features: True (dim: {effective_feature_dim})")
    logger.info(f"Feature dim: {beat_feature_dim} base + 4 edit = {effective_feature_dim} total")
    
    n_val = max(int(0.15 * len(pref_dataset)), 1)
    n_train = len(pref_dataset) - n_val
    
    train_dataset, val_dataset = random_split(pref_dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Initialize reward model
    reward_model = LearnedRewardModel(
        input_dim=effective_feature_dim,
        hidden_dim=256,
        n_layers=3,
        n_heads=4,
        dropout=0.1,
    )
    
    # Load checkpoint if resuming
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        if "model_state_dict" in checkpoint:
            reward_model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        else:
            reward_model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint")
    
    # Train with given learning rate
    metrics = train_reward_model(
        reward_model, train_loader, val_loader,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        save_path=save_path,
        target_accuracy=target_accuracy,
    )
    
    # Save final model
    final_path = save_path / "reward_model_final.pt"
    torch.save({
        "model_state_dict": reward_model.state_dict(),
        "config": {
            "input_dim": effective_feature_dim,
            "base_feature_dim": beat_feature_dim,
            "use_edit_aware_features": use_edit_aware,
            "hidden_dim": 256,
            "n_layers": 3,
            "n_heads": 4,
        },
        "metrics": metrics,
    }, final_path)
    logger.info(f"Saved final reward model to {final_path}")
    
    return reward_model


def main():
    parser = argparse.ArgumentParser(description="Train Learned Reward Model")
    parser.add_argument("--data_dir", type=str, default="./training_data")
    parser.add_argument("--save_dir", type=str, default="./models/reward_model")
    parser.add_argument("--mode", type=str, default="bootstrap",
                        choices=["bootstrap", "train", "generate", "train_cached"])
    parser.add_argument("--preferences", type=str, default=None,
                        help="Path to preference JSON file (for mode=train)")
    parser.add_argument("--cached_pairs", type=str, default=None,
                        help="Path to cached bootstrap_pairs.json to skip pair generation")
    parser.add_argument("--agent_checkpoint", type=str, default=None,
                        help="Agent checkpoint for generating edits")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from (uses lower LR)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1e-5 if resuming, 1e-4 if fresh)")
    parser.add_argument("--max_pairs", type=int, default=40000,
                        help="Max pairs to load (default 40000, use less for faster startup)")
    parser.add_argument("--n_pairs", type=int, default=10,
                        help="Pairs per track for bootstrap")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="Max epochs (will stop early at 90% accuracy)")
    parser.add_argument("--target_accuracy", type=float, default=0.90,
                        help="Target validation accuracy to stop training")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = get_default_config()
    
    if args.mode == "bootstrap":
        bootstrap_reward_model(
            config,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            n_pairs_per_track=args.n_pairs,
            n_epochs=args.n_epochs,
            agent_checkpoint=args.agent_checkpoint,
        )
    elif args.mode == "train_cached":
        # Train from cached pairs JSON (skip pair generation!)
        if not args.cached_pairs:
            # Look for bootstrap_pairs.json in save_dir
            default_cache = Path(args.save_dir) / "bootstrap_pairs.json"
            if default_cache.exists():
                args.cached_pairs = str(default_cache)
            else:
                logger.error("--cached_pairs required for train_cached mode, or bootstrap_pairs.json must exist in save_dir")
                return
        
        # Determine learning rate
        if args.lr:
            lr = args.lr
        elif args.resume_from:
            lr = 1e-5  # Lower LR for fine-tuning
        else:
            lr = 1e-4  # Default for fresh start
        
        train_from_cached_pairs(
            config,
            cached_pairs_path=args.cached_pairs,
            save_dir=args.save_dir,
            n_epochs=args.n_epochs,
            max_pairs=args.max_pairs,
            resume_from=args.resume_from,
            lr=lr,
            target_accuracy=args.target_accuracy,
        )
    elif args.mode == "generate":
        logger.info("Generate mode: Creating unlabeled pairs for human annotation")
        # TODO: Implement pair generation without labels
        logger.warning("Generate mode not fully implemented yet")
    elif args.mode == "train":
        if not args.preferences:
            logger.error("--preferences required for train mode")
            return
        # TODO: Implement training from human preference file
        logger.warning("Train from file mode not fully implemented yet")


if __name__ == "__main__":
    main()
