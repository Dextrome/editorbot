"""
Edit Policy V10 - Sequence-Level Reinforcement Learning

Key innovations:
1. RL Policy Network: Decisions consider editing STATE (what's been kept, trajectory)
2. Dense Self-Supervised Rewards:
   - Transition smoothness (cross-correlation at cuts)
   - Energy arc (builds tension, has peaks/resolutions)
   - No immediate repetition (embedding distance from recent kept segments)
   - Reference similarity (style matching)
   - Duration target (penalize too short/long)
3. PPO-style policy gradient training
4. No new human labeling required - learns from self-supervised signals

Architecture:
- State: (segment_features, context_embedding, trajectory_stats)
- Action: keep (1) or cut (0)
- Policy: Neural network with LSTM for trajectory memory
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from pathlib import Path
import librosa
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
import random
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, correlate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUM_WORKERS = max(1, os.cpu_count() - 2)


# =============================================================================
# FEATURE EXTRACTOR (reused from V9)
# =============================================================================

class SegmentFeatureExtractor:
    """Extract audio features from a segment."""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from audio segment."""
        features = []
        
        # 1. MFCCs (13 coefficients, mean + std = 26)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        features.extend(mfcc.mean(axis=1))
        features.extend(mfcc.std(axis=1))
        
        # 2. Spectral features (5)
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        
        features.append(spec_cent.mean())
        features.append(spec_bw.mean())
        features.append(spec_flat.mean())
        features.append(spec_rolloff.mean())
        
        # 3. RMS energy (2)
        rms = librosa.feature.rms(y=audio)
        features.append(rms.mean())
        features.append(rms.std())
        
        # 4. Zero crossing rate (1)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(zcr.mean())
        
        # 5. Chroma (12 mean)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features.extend(chroma.mean(axis=1))
        
        # 6. Tempo (1)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features.append(float(tempo) / 200.0)
        
        # 7. Spectral contrast (7)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        features.extend(contrast.mean(axis=1))
        
        # 8. Onset strength (2)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        features.append(onset_env.mean())
        features.append(onset_env.std())
        
        # 9. Spectral flux (2)
        spec = np.abs(librosa.stft(audio))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        features.append(flux.mean())
        features.append(flux.std())
        
        return np.array(features, dtype=np.float32)


def create_context_windows(features: np.ndarray) -> np.ndarray:
    """Create context windows from sequential features."""
    n_segments = len(features)
    if n_segments == 0:
        return np.array([])
    
    feature_dim = features.shape[1]
    windowed = []
    
    for i in range(n_segments):
        prev_idx = max(0, i - 1)
        next_idx = min(n_segments - 1, i + 1)
        
        context = np.concatenate([
            features[prev_idx],
            features[i],
            features[next_idx]
        ])
        windowed.append(context)
    
    return np.array(windowed)


# =============================================================================
# RL POLICY NETWORK - Sequence-Aware Decision Making
# =============================================================================

class TrajectoryEncoder(nn.Module):
    """
    Encodes the history of editing decisions and kept segment features.
    Uses LSTM to capture sequential patterns.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Compress segment features
        self.feature_compress = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU()
        )
        
        # LSTM for trajectory encoding
        # Input: compressed features + action taken
        self.lstm = nn.LSTM(
            input_size=64 + 1,  # features + action
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        segment_features: torch.Tensor,  # (batch, seq_len, feature_dim)
        actions: torch.Tensor,  # (batch, seq_len)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode trajectory of past decisions.
        
        Returns: (trajectory_embedding, new_hidden_state)
        """
        batch_size, seq_len, _ = segment_features.shape
        
        # Compress features
        compressed = self.feature_compress(segment_features)  # (batch, seq, 64)
        
        # Concatenate with actions
        actions_expanded = actions.unsqueeze(-1)  # (batch, seq, 1)
        lstm_input = torch.cat([compressed, actions_expanded], dim=-1)  # (batch, seq, 65)
        
        # LSTM encoding
        if hidden is None:
            output, hidden = self.lstm(lstm_input)
        else:
            output, hidden = self.lstm(lstm_input, hidden)
        
        return output, hidden


class RLEditPolicy(nn.Module):
    """
    RL Policy for sequential editing decisions.
    
    State representation:
    - Current segment features (context window)
    - Trajectory encoding (what's been kept so far)
    - Position info (how far through track)
    - Running statistics (energy, kept ratio so far)
    
    Action: keep (1) or cut (0) - Bernoulli policy
    """
    
    def __init__(
        self,
        base_feature_dim: int,
        trajectory_hidden_dim: int = 128,
        style_embedding_dim: int = 64
    ):
        super().__init__()
        
        context_dim = base_feature_dim * 3  # context window
        
        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            feature_dim=context_dim,
            hidden_dim=trajectory_hidden_dim
        )
        
        # Style embedding head (like V9, for reference similarity)
        self.style_encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, style_embedding_dim)
        )
        
        # State dimension:
        # - Current segment: context_dim
        # - Trajectory: trajectory_hidden_dim
        # - Position: 3 (relative position, segments remaining, current kept ratio)
        # - Stats: 4 (running energy mean/var, last kept energy delta, monotonicity)
        state_dim = context_dim + trajectory_hidden_dim + 3 + 4
        
        # Policy head (outputs logit for keep probability)
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Value head (for PPO advantage estimation)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.base_feature_dim = base_feature_dim
        self.trajectory_hidden_dim = trajectory_hidden_dim
        self.style_embedding_dim = style_embedding_dim
        
        # Initialize policy head bias towards keeping ~35% (logit ~ -0.6)
        # This prevents early collapse to "cut everything"
        with torch.no_grad():
            self.policy_head[-1].bias.fill_(-0.5)  # Start near 35% keep rate
    
    def get_style_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized style embedding."""
        emb = self.style_encoder(x)
        return F.normalize(emb, p=2, dim=-1)
    
    def forward(
        self,
        current_segment: torch.Tensor,  # (batch, context_dim)
        trajectory_state: torch.Tensor,  # (batch, trajectory_hidden_dim)
        position_info: torch.Tensor,  # (batch, 3)
        running_stats: torch.Tensor,  # (batch, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_logits: (batch, 1)
            state_value: (batch, 1)
        """
        # Combine all state information
        state = torch.cat([
            current_segment,
            trajectory_state,
            position_info,
            running_stats
        ], dim=-1)
        
        action_logits = self.policy_head(state)
        state_value = self.value_head(state)
        
        return action_logits, state_value
    
    def get_action(
        self,
        current_segment: torch.Tensor,
        trajectory_state: torch.Tensor,
        position_info: torch.Tensor,
        running_stats: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: (batch,) 0 or 1
            log_prob: (batch,) log probability of action
            value: (batch,) state value estimate
        """
        action_logits, value = self.forward(
            current_segment, trajectory_state, position_info, running_stats
        )
        
        probs = torch.sigmoid(action_logits.squeeze(-1))
        
        if deterministic:
            action = (probs > 0.5).float()
            # For deterministic, log_prob is log of chosen probability
            log_prob = torch.where(
                action == 1,
                torch.log(probs + 1e-8),
                torch.log(1 - probs + 1e-8)
            )
        else:
            dist = Bernoulli(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)


# =============================================================================
# SELF-SUPERVISED REWARD FUNCTIONS
# =============================================================================

class RewardComputer:
    """
    Computes dense self-supervised rewards for editing decisions.
    
    Rewards:
    1. Transition smoothness: Audio continuity at cut points
    2. Energy arc: Builds/releases tension naturally  
    3. No repetition: Penalize keeping similar adjacent segments
    4. Reference similarity: Match target style
    5. Duration target: Penalize too short/long edits
    
    Also provides per-step rewards for dense feedback.
    """
    
    def __init__(
        self,
        reference_centroid: np.ndarray = None,
        target_keep_ratio: float = 0.35,
        sr: int = 22050
    ):
        self.reference_centroid = reference_centroid
        self.target_keep_ratio = target_keep_ratio
        self.sr = sr
        
        # Reward weights (must sum to 1.0 for interpretable total)
        self.w_transition = 0.15
        self.w_energy_arc = 0.10
        self.w_no_repeat = 0.15
        self.w_reference = 0.25
        self.w_duration = 0.35  # Highest weight - hitting target duration is critical
    
    def compute_transition_reward(
        self,
        audio: np.ndarray,
        kept_regions: List[Tuple[float, float]],
    ) -> float:
        """
        Reward for smooth transitions between kept regions.
        Uses normalized cross-correlation and RMS similarity at cut points.
        """
        if len(kept_regions) < 2:
            return 0.5  # Neutral if only one region
        
        transition_scores = []
        
        for i in range(len(kept_regions) - 1):
            end_of_first = int(kept_regions[i][1] * self.sr)
            start_of_second = int(kept_regions[i + 1][0] * self.sr)
            
            # Get 0.5s of audio at boundaries
            window = int(0.5 * self.sr)
            
            end_audio = audio[max(0, end_of_first - window):end_of_first]
            start_audio = audio[start_of_second:min(len(audio), start_of_second + window)]
            
            if len(end_audio) < window // 2 or len(start_audio) < window // 2:
                continue
            
            # Normalize audio segments
            end_audio = end_audio / (np.abs(end_audio).max() + 1e-8)
            start_audio = start_audio / (np.abs(start_audio).max() + 1e-8)
            
            # Normalized cross-correlation (returns value in [-1, 1])
            end_chunk = end_audio[-1000:] if len(end_audio) >= 1000 else end_audio
            start_chunk = start_audio[:1000] if len(start_audio) >= 1000 else start_audio
            
            # Compute normalized correlation coefficient
            end_norm = end_chunk - end_chunk.mean()
            start_norm = start_chunk - start_chunk.mean()
            
            # Pearson correlation-style normalization
            corr = np.correlate(end_norm, start_norm, mode='valid')
            norm_factor = np.sqrt(np.sum(end_norm**2) * np.sum(start_norm**2))
            max_corr = (corr.max() / (norm_factor + 1e-8)) if len(corr) > 0 else 0
            max_corr = np.clip(max_corr, -1, 1)  # Ensure in valid range
            
            # RMS similarity (closer = better, scale to [0, 1])
            rms_end = np.sqrt(np.mean(end_audio**2))
            rms_start = np.sqrt(np.mean(start_audio**2))
            rms_similarity = 1 - min(abs(rms_end - rms_start), 1)
            
            # Combined score: weighted average of correlation and RMS match
            # Both are now in [0, 1] range
            score = 0.5 * ((max_corr + 1) / 2) + 0.5 * rms_similarity
            transition_scores.append(score)
        
        if len(transition_scores) == 0:
            return 0.5
        
        return np.mean(transition_scores)
    
    def compute_energy_arc_reward(
        self,
        segment_energies: np.ndarray,
        kept_mask: np.ndarray
    ) -> float:
        """
        Reward for maintaining good energy arc.
        Good edits should have:
        - Variety in energy (not monotonous)
        - Build-ups and releases (not random jumps)
        - Strong ending (not fizzle out)
        """
        kept_energies = segment_energies[kept_mask]
        
        if len(kept_energies) < 3:
            return 0.5
        
        # 1. Energy variance (want some variation, but not too extreme)
        energy_var = np.var(kept_energies)
        variance_score = min(energy_var / 0.1, 1.0)  # Cap at 1.0
        
        # 2. Smoothness of energy changes (penalize big jumps)
        energy_diffs = np.diff(kept_energies)
        jump_penalty = np.mean(np.abs(energy_diffs) > 0.3)  # % of big jumps
        smoothness_score = 1 - jump_penalty
        
        # 3. Has at least one peak (not flat)
        max_idx = np.argmax(kept_energies)
        has_peak = 0.2 < max_idx / len(kept_energies) < 0.9  # Peak not at very start/end
        peak_score = 1.0 if has_peak else 0.5
        
        # 4. Ending strength (don't end on weakest)
        ending_percentile = np.sum(kept_energies <= kept_energies[-1]) / len(kept_energies)
        ending_score = 0.5 + 0.5 * (ending_percentile > 0.25)  # Ending above 25th percentile
        
        return 0.25 * (variance_score + smoothness_score + peak_score + ending_score)
    
    def compute_no_repeat_reward(
        self,
        segment_embeddings: np.ndarray,  # (n_segments, embed_dim)
        kept_mask: np.ndarray
    ) -> float:
        """
        Penalize keeping very similar adjacent segments.
        """
        kept_indices = np.where(kept_mask)[0]
        
        if len(kept_indices) < 2:
            return 1.0  # No repetition possible
        
        kept_embeddings = segment_embeddings[kept_indices]
        
        # Compute similarity between consecutive kept segments
        consecutive_sims = []
        for i in range(len(kept_embeddings) - 1):
            sim = np.dot(kept_embeddings[i], kept_embeddings[i + 1])
            consecutive_sims.append(sim)
        
        # Penalize high similarity
        mean_sim = np.mean(consecutive_sims)
        
        # Want similarity around 0.3-0.7 (related but not identical)
        if mean_sim > 0.9:
            return 0.2  # Very repetitive
        elif mean_sim > 0.7:
            return 0.6
        elif mean_sim > 0.5:
            return 1.0  # Good variety
        elif mean_sim > 0.3:
            return 0.8  # Maybe too random
        else:
            return 0.5  # Very disconnected
    
    def compute_reference_reward(
        self,
        segment_embeddings: np.ndarray,
        kept_mask: np.ndarray
    ) -> float:
        """
        Reward for keeping segments similar to reference style.
        """
        if self.reference_centroid is None:
            return 0.5  # Neutral if no reference
        
        kept_embeddings = segment_embeddings[kept_mask]
        
        if len(kept_embeddings) == 0:
            return 0.0
        
        # Cosine similarity to reference centroid
        similarities = kept_embeddings @ self.reference_centroid
        
        # Average similarity (shifted from [-1,1] to [0,1])
        return (np.mean(similarities) + 1) / 2
    
    def compute_duration_reward(
        self,
        total_segments: int,
        kept_segments: int
    ) -> float:
        """
        Reward for hitting target duration.
        Strong penalty for being too far from target.
        """
        actual_ratio = kept_segments / total_segments
        
        # Calculate deviation from target
        diff = abs(actual_ratio - self.target_keep_ratio)
        
        # Smooth reward curve centered on target
        # Use Gaussian-like shape: exp(-diff^2 / scale)
        if diff < 0.05:
            return 1.0
        elif diff < 0.10:
            return 0.9
        elif diff < 0.15:
            return 0.7
        elif diff < 0.20:
            return 0.4
        elif diff < 0.25:
            return 0.2
        else:
            return 0.0  # Strong penalty for being way off
    
    def compute_total_reward(
        self,
        audio: np.ndarray,
        segment_times: List[Tuple[float, float]],
        segment_energies: np.ndarray,
        segment_embeddings: np.ndarray,
        kept_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute total reward as weighted sum of components.
        
        Returns dict with breakdown and total.
        """
        # Get kept regions for transition reward
        kept_indices = np.where(kept_mask)[0]
        kept_regions = [segment_times[i] for i in kept_indices]
        
        # Merge overlapping regions
        if len(kept_regions) > 0:
            merged = [kept_regions[0]]
            for s, e in kept_regions[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            kept_regions = merged
        
        r_transition = self.compute_transition_reward(audio, kept_regions)
        r_energy = self.compute_energy_arc_reward(segment_energies, kept_mask)
        r_no_repeat = self.compute_no_repeat_reward(segment_embeddings, kept_mask)
        r_reference = self.compute_reference_reward(segment_embeddings, kept_mask)
        r_duration = self.compute_duration_reward(len(kept_mask), kept_mask.sum())
        
        total = (
            self.w_transition * r_transition +
            self.w_energy_arc * r_energy +
            self.w_no_repeat * r_no_repeat +
            self.w_reference * r_reference +
            self.w_duration * r_duration
        )
        
        return {
            'total': total,
            'transition': r_transition,
            'energy_arc': r_energy,
            'no_repeat': r_no_repeat,
            'reference': r_reference,
            'duration': r_duration
        }


# =============================================================================
# EXPERIENCE BUFFER FOR RL
# =============================================================================

class Experience:
    """Single step of experience."""
    def __init__(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float = 0.0,
        done: bool = False
    ):
        self.state = state
        self.action = action
        self.log_prob = log_prob
        self.value = value
        self.reward = reward
        self.done = done


class EpisodeBuffer:
    """Buffer for one episode (one track editing)."""
    def __init__(self):
        self.experiences = []
    
    def add(self, exp: Experience):
        self.experiences.append(exp)
    
    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE returns."""
        n = len(self.experiences)
        if n == 0:
            return
        
        returns = []
        advantages = []
        
        last_gae = 0
        last_return = 0
        
        for t in reversed(range(n)):
            exp = self.experiences[t]
            
            if t == n - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = self.experiences[t + 1].value.item()
                next_non_terminal = 1
            
            delta = exp.reward + gamma * next_value * next_non_terminal - exp.value.item()
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            last_return = exp.reward + gamma * last_return * next_non_terminal
            
            advantages.insert(0, last_gae)
            returns.insert(0, last_return)
        
        # Store
        for i, exp in enumerate(self.experiences):
            exp.advantage = advantages[i]
            exp.return_target = returns[i]
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Convert to batch for training."""
        states = {
            'current_segment': torch.stack([e.state['current_segment'] for e in self.experiences]),
            'trajectory_state': torch.stack([e.state['trajectory_state'] for e in self.experiences]),
            'position_info': torch.stack([e.state['position_info'] for e in self.experiences]),
            'running_stats': torch.stack([e.state['running_stats'] for e in self.experiences]),
        }
        
        actions = torch.stack([e.action for e in self.experiences])
        old_log_probs = torch.stack([e.log_prob for e in self.experiences])
        returns = torch.tensor([e.return_target for e in self.experiences], dtype=torch.float32)
        advantages = torch.tensor([e.advantage for e in self.experiences], dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'returns': returns,
            'advantages': advantages
        }


# =============================================================================
# RL TRAINER
# =============================================================================

class EditPolicyTrainerV10:
    """
    V10: Reinforcement Learning for Sequential Edit Decisions.
    
    Strategy:
    1. Use V9's quality+reference scores as per-segment baseline
    2. RL learns sequential patterns: transitions, energy arcs, pacing
    3. Reward = V9_baseline_score + transition_bonus + energy_arc_bonus
    
    This avoids the sparse reward problem by giving dense per-step rewards.
    """
    
    def __init__(
        self,
        training_audio_dir: str,  # Directory with audio files to practice on
        reference_dir: str = None,
        model_dir: str = "./models",
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        target_keep_ratio: float = 0.35,
        style_embedding_dim: int = 64,
    ):
        self.training_audio_dir = Path(training_audio_dir)
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.target_keep_ratio = target_keep_ratio
        self.style_embedding_dim = style_embedding_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.extractor = SegmentFeatureExtractor()
        self.base_feature_dim = 57  # Known from feature extractor
        
        # Initialize policy
        self.policy = RLEditPolicy(
            base_feature_dim=self.base_feature_dim,
            trajectory_hidden_dim=128,
            style_embedding_dim=style_embedding_dim
        ).to(self.device)
        
        # Load V9 model as baseline scorer
        self.v9_model = None
        self._load_v9_baseline()
        
        # Try to load reference centroid from V9 if available
        self.reference_centroid = None
        centroid_path = self.model_dir / "reference_centroid_v9.npy"
        if centroid_path.exists():
            self.reference_centroid = np.load(centroid_path)
            self.ref_centroid_t = torch.FloatTensor(self.reference_centroid).to(self.device)
            logger.info(f"Loaded reference centroid from V9")
        
        # Reward computer
        self.reward_computer = RewardComputer(
            reference_centroid=self.reference_centroid,
            target_keep_ratio=target_keep_ratio
        )
        
        # Training audio cache
        self.training_tracks = []
    
    def _load_v9_baseline(self):
        """Load V9 model to provide per-segment quality baseline."""
        try:
            from train_edit_policy_v9 import DualHeadModel
            
            feature_dim = int(np.load(self.model_dir / "feature_dim_v9.npy"))
            self.v9_model = DualHeadModel(
                base_feature_dim=feature_dim,
                embedding_dim=64
            ).to(self.device)
            
            self.v9_model.load_state_dict(
                torch.load(self.model_dir / "classifier_v9_best.pt", weights_only=True)
            )
            self.v9_model.eval()
            logger.info("Loaded V9 model as baseline scorer")
        except Exception as e:
            logger.warning(f"Could not load V9 baseline: {e}")
            self.v9_model = None
    
    def find_training_files(self) -> List[Path]:
        """Find audio files to train on."""
        files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            files.extend(self.training_audio_dir.glob(f"**/{ext}"))
        
        logger.info(f"Found {len(files)} training audio files")
        return files
    
    def preprocess_track(self, audio_path: Path) -> Optional[Dict]:
        """
        Preprocess a track: extract features, compute energies, embeddings.
        Also compute V9 baseline scores.
        """
        try:
            audio, sr = librosa.load(str(audio_path), sr=22050, mono=True)
            duration = len(audio) / sr
            
            if duration < self.segment_duration * 3:
                return None  # Too short
            
            # Extract segments
            features = []
            energies = []
            times = []
            
            start = 0.0
            while start + self.segment_duration <= duration:
                end = start + self.segment_duration
                seg_audio = audio[int(start * sr):int(end * sr)]
                
                feat = self.extractor.extract(seg_audio)
                features.append(feat)
                
                # Energy (RMS)
                rms = np.sqrt(np.mean(seg_audio ** 2))
                energies.append(rms)
                
                times.append((start, end))
                start += self.hop_duration
            
            if len(features) < 5:
                return None
            
            features = np.array(features)
            windowed = create_context_windows(features)
            energies = np.array(energies)
            
            # Normalize energies
            energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
            
            # Compute V9 baseline scores
            v9_scores = np.zeros(len(windowed))
            if self.v9_model is not None:
                with torch.no_grad():
                    windowed_t = torch.FloatTensor(windowed).to(self.device)
                    quality_logits, style_emb = self.v9_model(windowed_t)
                    quality = torch.sigmoid(quality_logits).squeeze().cpu().numpy()
                    
                    # Reference similarity
                    if hasattr(self, 'ref_centroid_t'):
                        ref_sim = torch.mm(style_emb, self.ref_centroid_t.unsqueeze(1)).squeeze()
                        ref_sim = ((ref_sim + 1) / 2).cpu().numpy()  # Normalize to [0, 1]
                    else:
                        ref_sim = np.zeros_like(quality)
                    
                    # Combined V9 score
                    v9_scores = quality + 0.3 * ref_sim
                    v9_scores = v9_scores / 1.3  # Normalize to [0, 1]
            
            return {
                'path': audio_path,
                'audio': audio,
                'sr': sr,
                'features': features,
                'windowed': windowed,
                'energies': energies,
                'times': times,
                'duration': duration,
                'v9_scores': v9_scores
            }
        
        except Exception as e:
            logger.warning(f"Failed to preprocess {audio_path}: {e}")
            return None
    
    def rollout_episode(
        self,
        track: Dict,
        deterministic: bool = False,
        exploration_epsilon: float = 0.0
    ) -> Tuple[EpisodeBuffer, Dict]:
        """
        Run one episode: make editing decisions for entire track.
        
        Args:
            track: Preprocessed track data
            deterministic: Use deterministic policy (greedy)
            exploration_epsilon: Random action probability for exploration
        
        Returns:
            buffer: Episode buffer with experiences
            reward_info: Dict with reward breakdown
        """
        self.policy.eval()
        
        buffer = EpisodeBuffer()
        
        windowed = torch.FloatTensor(track['windowed']).to(self.device)
        n_segments = len(windowed)
        
        # Initialize trajectory state (zero hidden state)
        h = torch.zeros(2, 1, 128).to(self.device)
        c = torch.zeros(2, 1, 128).to(self.device)
        trajectory_state = torch.zeros(1, 128).to(self.device)
        
        # Running statistics
        kept_count = 0
        energy_sum = 0
        energy_sq_sum = 0
        last_kept_energy = 0
        
        kept_mask = np.zeros(n_segments, dtype=bool)
        
        with torch.no_grad():
            for i in range(n_segments):
                # Current segment features
                current_segment = windowed[i:i+1]  # (1, context_dim)
                
                # Position info
                rel_position = i / n_segments
                segments_remaining = (n_segments - i) / n_segments
                current_kept_ratio = kept_count / (i + 1) if i > 0 else 0
                
                position_info = torch.tensor(
                    [[rel_position, segments_remaining, current_kept_ratio]],
                    dtype=torch.float32
                ).to(self.device)
                
                # Running stats
                if kept_count > 0:
                    energy_mean = energy_sum / kept_count
                    energy_var = max(0, energy_sq_sum / kept_count - energy_mean ** 2)
                    energy_delta = track['energies'][i] - last_kept_energy
                    # Monotonicity: are we building or releasing?
                    monotonicity = np.sign(energy_delta)
                else:
                    energy_mean = 0
                    energy_var = 0
                    energy_delta = 0
                    monotonicity = 0
                
                running_stats = torch.tensor(
                    [[energy_mean, energy_var, energy_delta, monotonicity]],
                    dtype=torch.float32
                ).to(self.device)
                
                # Get action
                action, log_prob, value = self.policy.get_action(
                    current_segment,
                    trajectory_state,
                    position_info,
                    running_stats,
                    deterministic=deterministic
                )
                
                # Epsilon-greedy exploration
                if exploration_epsilon > 0 and random.random() < exploration_epsilon:
                    # Random action with bias towards target keep ratio
                    random_keep_prob = 0.35 + 0.2 * (random.random() - 0.5)  # ~25-45%
                    random_action_val = 1.0 if random.random() < random_keep_prob else 0.0
                    action = torch.tensor([random_action_val], device=self.device)
                    log_prob_val = np.log(random_keep_prob if random_action_val > 0.5 else (1 - random_keep_prob))
                    log_prob = torch.tensor([log_prob_val], device=self.device)
                
                # Store experience
                exp = Experience(
                    state={
                        'current_segment': current_segment.squeeze(0),
                        'trajectory_state': trajectory_state.squeeze(0),
                        'position_info': position_info.squeeze(0),
                        'running_stats': running_stats.squeeze(0)
                    },
                    action=action,
                    log_prob=log_prob,
                    value=value
                )
                buffer.add(exp)
                
                # Update kept mask
                kept = action.item() > 0.5
                kept_mask[i] = kept
                
                # Update running stats
                if kept:
                    kept_count += 1
                    energy_sum += track['energies'][i]
                    energy_sq_sum += track['energies'][i] ** 2
                    last_kept_energy = track['energies'][i]
                
                # Update trajectory state (feed through LSTM)
                # Use features of current segment and action
                seg_feat = current_segment.unsqueeze(1)  # (1, 1, context_dim)
                action_t = action.view(1, 1).to(self.device)  # (1, 1)
                
                _, (h, c) = self.policy.trajectory_encoder(
                    seg_feat, action_t, (h, c)
                )
                trajectory_state = h[-1]  # Last layer's hidden state
        
        # Compute reward for entire episode
        segment_embeddings = self.policy.get_style_embedding(windowed).detach().cpu().numpy()
        
        reward_info = self.reward_computer.compute_total_reward(
            audio=track['audio'],
            segment_times=track['times'],
            segment_energies=track['energies'],
            segment_embeddings=segment_embeddings,
            kept_mask=kept_mask
        )
        
        # Add keep_ratio to reward_info for monitoring
        reward_info['keep_ratio'] = kept_mask.sum() / len(kept_mask) if len(kept_mask) > 0 else 0
        
        # Dense per-step rewards using V9 baseline
        # This gives immediate feedback while RL learns sequential patterns
        n_experiences = len(buffer.experiences)
        
        if n_experiences > 0:
            v9_scores = track.get('v9_scores', np.zeros(n_experiences))
            
            for i, exp in enumerate(buffer.experiences):
                step_reward = 0.0
                
                if kept_mask[i]:
                    # Reward = V9 score for this segment (quality + ref similarity)
                    step_reward += 0.2 * v9_scores[i]
                    
                    # Bonus for keeping high-energy segments
                    step_reward += 0.05 * track['energies'][i]
                else:
                    # Small penalty for cutting high V9-score segments
                    if v9_scores[i] > 0.7:
                        step_reward -= 0.1
                
                # Progressive keep ratio tracking
                current_ratio = kept_mask[:i+1].sum() / (i + 1) if i > 0 else 0
                target = self.reward_computer.target_keep_ratio
                
                # Penalty for being too far from target
                if abs(current_ratio - target) > 0.15:
                    if current_ratio < target - 0.15 and not kept_mask[i]:
                        step_reward -= 0.05  # Should be keeping more
                    elif current_ratio > target + 0.15 and kept_mask[i]:
                        step_reward -= 0.05  # Should be keeping less
                
                exp.reward = step_reward
            
            # Terminal bonus based on overall quality
            buffer.experiences[-1].reward += reward_info['total'] * 0.3
            buffer.experiences[-1].done = True
        
        # Compute returns
        buffer.compute_returns(gamma=0.99, lam=0.95)
        
        return buffer, reward_info
    
    def ppo_update(
        self,
        batch: Dict[str, torch.Tensor],
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05  # Increased entropy for exploration
    ) -> Dict[str, float]:
        """PPO policy update."""
        self.policy.train()
        
        states = {k: v.to(self.device) for k, v in batch['states'].items()}
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        returns = batch['returns'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        
        # Forward pass
        action_logits, values = self.policy(
            states['current_segment'],
            states['trajectory_state'],
            states['position_info'],
            states['running_stats']
        )
        
        probs = torch.sigmoid(action_logits.squeeze(-1))
        dist = Bernoulli(probs)
        
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Total loss
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train(
        self,
        epochs: int = 100,
        episodes_per_epoch: int = 10,
        ppo_epochs: int = 4,
        lr: float = 3e-4,
        max_tracks: int = None
    ):
        """
        Train RL policy.
        """
        logger.info("=" * 70)
        logger.info("TRAINING V10 - Reinforcement Learning Policy")
        logger.info("=" * 70)
        
        # Find and preprocess training tracks
        training_files = self.find_training_files()
        if max_tracks:
            training_files = training_files[:max_tracks]
        
        logger.info(f"Preprocessing {len(training_files)} tracks...")
        for f in tqdm(training_files):
            track = self.preprocess_track(f)
            if track is not None:
                self.training_tracks.append(track)
        
        logger.info(f"Successfully preprocessed {len(self.training_tracks)} tracks")
        
        if len(self.training_tracks) == 0:
            raise ValueError("No valid training tracks found!")
        
        # Optimizer
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        
        # Training loop
        best_reward = -float('inf')
        
        for epoch in range(epochs):
            epoch_rewards = []
            epoch_keep_ratios = []
            epoch_reward_breakdown = {
                'transition': [], 'energy_arc': [], 'no_repeat': [],
                'reference': [], 'duration': []
            }
            
            # Exploration rate (decreases over time)
            epsilon = max(0.1, 0.5 * (1 - epoch / epochs))
            
            # Collect episodes
            all_buffers = []
            
            for _ in range(episodes_per_epoch):
                # Sample random track
                track = random.choice(self.training_tracks)
                
                # Rollout with exploration
                buffer, reward_info = self.rollout_episode(
                    track, 
                    deterministic=False,
                    exploration_epsilon=epsilon
                )
                
                epoch_rewards.append(reward_info['total'])
                epoch_keep_ratios.append(reward_info.get('keep_ratio', 0))
                for key in epoch_reward_breakdown:
                    epoch_reward_breakdown[key].append(reward_info[key])
                
                all_buffers.append(buffer)
            
            # Combine all episodes into one batch
            all_experiences = []
            for buf in all_buffers:
                all_experiences.extend(buf.experiences)
            
            if len(all_experiences) == 0:
                continue
            
            # Create combined batch
            combined_buffer = EpisodeBuffer()
            combined_buffer.experiences = all_experiences
            batch = combined_buffer.get_batch()
            
            # PPO updates
            total_loss = 0
            for _ in range(ppo_epochs):
                losses = self.ppo_update(batch)
                
                optimizer.zero_grad()
                
                # Recompute loss for backward (can't reuse)
                self.policy.train()
                states = {k: v.to(self.device) for k, v in batch['states'].items()}
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                returns = batch['returns'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                
                action_logits, values = self.policy(
                    states['current_segment'],
                    states['trajectory_state'],
                    states['position_info'],
                    states['running_stats']
                )
                
                probs = torch.sigmoid(action_logits.squeeze(-1))
                dist = Bernoulli(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns)
                
                loss = policy_loss + 0.5 * value_loss - 0.05 * entropy  # Increased entropy bonus
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            avg_reward = np.mean(epoch_rewards)
            avg_keep_ratio = np.mean(epoch_keep_ratios)
            
            is_best = avg_reward > best_reward
            if is_best:
                best_reward = avg_reward
                torch.save(self.policy.state_dict(), self.model_dir / "policy_v10_best.pt")
            
            if (epoch + 1) % 5 == 0 or is_best:
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"reward={avg_reward:.3f} keep={avg_keep_ratio*100:.1f}% "
                    f"(trans={np.mean(epoch_reward_breakdown['transition']):.2f} "
                    f"energy={np.mean(epoch_reward_breakdown['energy_arc']):.2f} "
                    f"norepeat={np.mean(epoch_reward_breakdown['no_repeat']):.2f} "
                    f"ref={np.mean(epoch_reward_breakdown['reference']):.2f} "
                    f"dur={np.mean(epoch_reward_breakdown['duration']):.2f}) "
                    f"eps={epsilon:.2f} loss={total_loss/ppo_epochs:.4f}"
                    f"{' *BEST*' if is_best else ''}"
                )
        
        # Save final model
        torch.save(self.policy.state_dict(), self.model_dir / "policy_v10_final.pt")
        np.save(self.model_dir / "feature_dim_v10.npy", self.base_feature_dim)
        
        logger.info(f"Training complete. Best reward: {best_reward:.3f}")
        
        return self.policy


# =============================================================================
# V10 EDITOR
# =============================================================================

def detect_beats(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Detect beat times in audio."""
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times


def detect_phrase_boundaries(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Detect likely phrase boundaries."""
    phrase_candidates = []
    
    rms = librosa.feature.rms(y=audio)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    rms_smooth = uniform_filter1d(rms, size=10)
    
    inverted = -rms_smooth
    peaks, _ = find_peaks(inverted, distance=50, prominence=0.01)
    
    for idx in peaks:
        if idx < len(rms_times):
            phrase_candidates.append(rms_times[idx])
    
    spec = np.abs(librosa.stft(audio))
    flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
    flux_times = librosa.frames_to_time(np.arange(len(flux)), sr=sr)
    flux_smooth = uniform_filter1d(flux, size=5)
    
    flux_peaks, _ = find_peaks(flux_smooth, distance=50, prominence=np.std(flux_smooth) * 0.5)
    
    for idx in flux_peaks:
        if idx < len(flux_times):
            phrase_candidates.append(flux_times[idx])
    
    if len(phrase_candidates) == 0:
        return np.array([])
    
    phrase_candidates = np.sort(phrase_candidates)
    
    merged = [phrase_candidates[0]]
    for t in phrase_candidates[1:]:
        if t - merged[-1] > 0.5:
            merged.append(t)
    
    return np.array(merged)


def find_best_cut_point(
    target_time: float,
    beat_times: np.ndarray,
    phrase_times: np.ndarray,
    search_window: float = 1.0
) -> float:
    """Find best cut point near target_time."""
    if len(phrase_times) > 0:
        phrase_in_window = phrase_times[
            (phrase_times >= target_time - search_window) &
            (phrase_times <= target_time + search_window)
        ]
        if len(phrase_in_window) > 0:
            closest_phrase = phrase_in_window[np.argmin(np.abs(phrase_in_window - target_time))]
            if len(beat_times) > 0:
                distances = np.abs(beat_times - closest_phrase)
                if distances.min() <= 0.25:
                    return beat_times[np.argmin(distances)]
            return closest_phrase
    
    if len(beat_times) > 0:
        distances = np.abs(beat_times - target_time)
        if distances.min() <= 0.5:
            return beat_times[np.argmin(distances)]
    
    return target_time


class V10Editor:
    """V10 editor with RL policy for sequential decisions."""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load feature dim
        self.base_feature_dim = int(np.load(self.model_dir / "feature_dim_v10.npy"))
        
        # Try to load reference centroid
        self.reference_centroid = None
        centroid_path = self.model_dir / "reference_centroid_v9.npy"
        if centroid_path.exists():
            self.reference_centroid = np.load(centroid_path)
        
        # Load policy
        self.policy = RLEditPolicy(
            base_feature_dim=self.base_feature_dim,
            trajectory_hidden_dim=128,
            style_embedding_dim=64
        ).to(self.device)
        
        # Try best model first, fall back to final
        model_path = self.model_dir / "policy_v10_best.pt"
        if not model_path.exists():
            model_path = self.model_dir / "policy_v10_final.pt"
        
        self.policy.load_state_dict(
            torch.load(model_path, weights_only=True)
        )
        self.policy.eval()
        
        self.extractor = SegmentFeatureExtractor()
        
        # Reward computer for stats
        self.reward_computer = RewardComputer(
            reference_centroid=self.reference_centroid,
            target_keep_ratio=0.35
        )
        
        logger.info(f"Loaded V10 RL policy from {model_path}")
    
    def process_track(
        self,
        audio_path: str,
        output_path: str,
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        crossfade_duration: float = 0.5,
        beat_align: bool = True,
        phrase_detect: bool = True,
        deterministic: bool = True,  # Use deterministic policy at inference
    ) -> Dict:
        """
        Process track using RL policy.
        
        Unlike V9, this makes sequential decisions considering trajectory.
        """
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(audio) / sr
        
        print("Detecting beats...")
        beat_times = detect_beats(audio, sr) if beat_align else np.array([])
        print(f"  Found {len(beat_times)} beats")
        
        print("Detecting phrase boundaries...")
        phrase_times = detect_phrase_boundaries(audio, sr) if phrase_detect else np.array([])
        print(f"  Found {len(phrase_times)} phrase boundaries")
        
        # Extract features
        features = []
        energies = []
        times = []
        start = 0.0
        
        while start + segment_duration <= duration:
            end = start + segment_duration
            seg_audio = audio[int(start * sr):int(end * sr)]
            
            feat = self.extractor.extract(seg_audio)
            features.append(feat)
            
            rms = np.sqrt(np.mean(seg_audio ** 2))
            energies.append(rms)
            
            times.append((start, end))
            start += hop_duration
        
        if len(features) == 0:
            return {'success': False, 'error': 'File too short'}
        
        features = np.array(features)
        windowed = create_context_windows(features)
        energies = np.array(energies)
        energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
        
        n_segments = len(windowed)
        windowed_t = torch.FloatTensor(windowed).to(self.device)
        
        # Run RL policy sequentially
        print("Running RL policy...")
        
        h = torch.zeros(2, 1, 128).to(self.device)
        c = torch.zeros(2, 1, 128).to(self.device)
        trajectory_state = torch.zeros(1, 128).to(self.device)
        
        kept_count = 0
        energy_sum = 0
        energy_sq_sum = 0
        last_kept_energy = 0
        
        kept_mask = np.zeros(n_segments, dtype=bool)
        decisions = []
        
        with torch.no_grad():
            for i in range(n_segments):
                current_segment = windowed_t[i:i+1]
                
                rel_position = i / n_segments
                segments_remaining = (n_segments - i) / n_segments
                current_kept_ratio = kept_count / (i + 1) if i > 0 else 0
                
                position_info = torch.tensor(
                    [[rel_position, segments_remaining, current_kept_ratio]],
                    dtype=torch.float32
                ).to(self.device)
                
                if kept_count > 0:
                    energy_mean = energy_sum / kept_count
                    energy_var = max(0, energy_sq_sum / kept_count - energy_mean ** 2)
                    energy_delta = energies[i] - last_kept_energy
                    monotonicity = np.sign(energy_delta)
                else:
                    energy_mean = 0
                    energy_var = 0
                    energy_delta = 0
                    monotonicity = 0
                
                running_stats = torch.tensor(
                    [[energy_mean, energy_var, energy_delta, monotonicity]],
                    dtype=torch.float32
                ).to(self.device)
                
                action, log_prob, value = self.policy.get_action(
                    current_segment,
                    trajectory_state,
                    position_info,
                    running_stats,
                    deterministic=deterministic
                )
                
                kept = action.item() > 0.5
                kept_mask[i] = kept
                decisions.append({
                    'time': times[i],
                    'kept': kept,
                    'energy': energies[i],
                    'value': value.item()
                })
                
                if kept:
                    kept_count += 1
                    energy_sum += energies[i]
                    energy_sq_sum += energies[i] ** 2
                    last_kept_energy = energies[i]
                
                # Update trajectory state
                seg_feat = current_segment.unsqueeze(1)
                action_t = action.view(1, 1).to(self.device)
                
                _, (h, c) = self.policy.trajectory_encoder(
                    seg_feat, action_t, (h, c)
                )
                trajectory_state = h[-1]
        
        n_kept = kept_mask.sum()
        print(f"Kept {n_kept}/{n_segments} segments ({100*n_kept/n_segments:.1f}%)")
        
        # Get kept regions
        kept_indices = np.where(kept_mask)[0]
        if len(kept_indices) == 0:
            return {'success': False, 'error': 'No segments kept'}
        
        kept_regions = [(times[i][0], times[i][1]) for i in kept_indices]
        
        # Merge adjacent
        merged = [kept_regions[0]]
        for s, e in kept_regions[1:]:
            if s <= merged[-1][1] + hop_duration:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        
        # Beat-align cuts
        print(f"Adjusting {len(merged)} regions to beat/phrase boundaries...")
        aligned_regions = []
        for i, (start, end) in enumerate(merged):
            if i == 0:
                new_start = max(0, find_best_cut_point(start, beat_times, phrase_times, 0.5))
            else:
                new_start = find_best_cut_point(start, beat_times, phrase_times, 1.0)
            
            if i == len(merged) - 1:
                new_end = min(duration, find_best_cut_point(end, beat_times, phrase_times, 0.5))
            else:
                new_end = find_best_cut_point(end, beat_times, phrase_times, 1.0)
            
            if new_end > new_start + 0.5:
                aligned_regions.append((new_start, new_end))
        
        merged = aligned_regions if aligned_regions else merged
        
        # Build output with crossfades
        crossfade_samples = int(crossfade_duration * sr)
        output = []
        
        for i, (s, e) in enumerate(merged):
            seg = audio[int(s * sr):int(e * sr)]
            
            if i > 0 and len(output) >= crossfade_samples and len(seg) > crossfade_samples:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                crossfaded = np.array(output[-crossfade_samples:]) * fade_out + seg[:crossfade_samples] * fade_in
                output = output[:-crossfade_samples]
                output.extend(crossfaded.tolist())
                output.extend(seg[crossfade_samples:].tolist())
            else:
                output.extend(seg.tolist())
        
        output = np.array(output)
        
        sf.write(output_path, output, sr)
        
        # Compute reward info
        segment_embeddings = self.policy.get_style_embedding(windowed_t).cpu().numpy()
        
        reward_info = self.reward_computer.compute_total_reward(
            audio=audio,
            segment_times=times,
            segment_energies=energies,
            segment_embeddings=segment_embeddings,
            kept_mask=kept_mask
        )
        
        return {
            'success': True,
            'input_duration': duration,
            'output_duration': len(output) / sr,
            'n_segments': n_segments,
            'n_kept': int(n_kept),
            'n_regions': len(merged),
            'keep_ratio_actual': n_kept / n_segments,
            'n_beats': len(beat_times),
            'n_phrases': len(phrase_times),
            'reward': reward_info,
            'decisions': decisions[:10]  # First 10 for debugging
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Train on reference songs + any available audio
    trainer = EditPolicyTrainerV10(
        training_audio_dir="./training_data/reference",  # Use reference songs to learn style
        reference_dir="./training_data/reference",
        model_dir="./models",
        target_keep_ratio=0.35,
    )
    
    trainer.train(
        epochs=100,
        episodes_per_epoch=10,
        ppo_epochs=4,
        lr=3e-4,
        max_tracks=50  # Limit tracks for faster iteration
    )
