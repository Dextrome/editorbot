"""
Edit Policy V10 - RL-Enhanced V9

Instead of training RL from scratch, this version:
1. Uses V9's quality + reference scores as the BASE
2. Learns a small trajectory-aware ADJUSTMENT via RL
3. Final score = V9_score + trajectory_adjustment

The RL component learns:
- Transition bonuses (boost if previous kept segment flows well into this one)
- Energy arc bonuses (boost if this helps build good tension/release)
- Pacing bonuses (avoid keeping too many consecutive or too few)

This is much more stable than pure RL because V9 provides strong guidance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import librosa
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Optional
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        features.extend(mfcc.mean(axis=1))
        features.extend(mfcc.std(axis=1))
        
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        
        features.append(spec_cent.mean())
        features.append(spec_bw.mean())
        features.append(spec_flat.mean())
        features.append(spec_rolloff.mean())
        
        rms = librosa.feature.rms(y=audio)
        features.append(rms.mean())
        features.append(rms.std())
        
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(zcr.mean())
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features.extend(chroma.mean(axis=1))
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        features.append(float(tempo) / 200.0)
        
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        features.extend(contrast.mean(axis=1))
        
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        features.append(onset_env.mean())
        features.append(onset_env.std())
        
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
# TRAJECTORY ADJUSTMENT NETWORK
# =============================================================================

class TrajectoryAdjustment(nn.Module):
    """
    Small network that learns trajectory-aware score adjustments.
    
    Input: 
    - Current segment features (context window)
    - Recent history (last N kept segments' features, averaged)
    - Position info (relative position, current keep ratio)
    - Energy trajectory (recent energy trend)
    
    Output:
    - Score adjustment in [-0.3, 0.3] range
    """
    
    def __init__(self, base_feature_dim: int):
        super().__init__()
        
        context_dim = base_feature_dim * 3  # context window
        
        # History embedding (compressed features of recent kept segments)
        self.history_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU()
        )
        
        # Input: current features + history + position + energy trend
        # context_dim + 64 + 3 + 3 = context_dim + 70
        input_dim = context_dim + 64 + 3 + 3
        
        self.adjustment_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Scale adjustment to [-0.3, 0.3]
        self.scale = 0.3
        
        self.base_feature_dim = base_feature_dim
    
    def forward(
        self,
        current_features: torch.Tensor,  # (batch, context_dim)
        history_features: torch.Tensor,  # (batch, context_dim) - avg of recent kept
        position_info: torch.Tensor,  # (batch, 3) - rel_pos, kept_ratio, gap_since_last_keep
        energy_trend: torch.Tensor,  # (batch, 3) - recent energy mean, var, delta
    ) -> torch.Tensor:
        """
        Compute trajectory-aware score adjustment.
        """
        history_emb = self.history_encoder(history_features)
        
        combined = torch.cat([
            current_features,
            history_emb,
            position_info,
            energy_trend
        ], dim=-1)
        
        adjustment = self.adjustment_net(combined) * self.scale
        
        return adjustment.squeeze(-1)


# =============================================================================
# V10 EDITOR - V9 + Trajectory Adjustments
# =============================================================================

class V10Editor:
    """
    V10 editor that combines V9 scoring with learned trajectory adjustments.
    
    Final score = V9_score + trajectory_adjustment
    
    The trajectory adjustment helps with:
    - Better transitions (boost segments that flow well from previous kept)
    - Energy arc (encourage build-up and release patterns)
    - Pacing (avoid too many or too few consecutive keeps)
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load V9 model
        self._load_v9_model()
        
        # Load trajectory adjustment network if available
        self.trajectory_net = None
        self._load_trajectory_net()
        
        self.extractor = SegmentFeatureExtractor()
        
        logger.info(f"V10 Editor loaded (V9 base + trajectory adjustment)")
    
    def _load_v9_model(self):
        """Load V9 model."""
        from train_edit_policy_v9 import DualHeadModel
        
        self.base_feature_dim = int(np.load(self.model_dir / "feature_dim_v9.npy"))
        self.reference_centroid = np.load(self.model_dir / "reference_centroid_v9.npy")
        self.similarity_weight = float(np.load(self.model_dir / "similarity_weight_v9.npy"))
        
        self.v9_model = DualHeadModel(
            base_feature_dim=self.base_feature_dim,
            embedding_dim=len(self.reference_centroid)
        ).to(self.device)
        
        self.v9_model.load_state_dict(
            torch.load(self.model_dir / "classifier_v9_best.pt", weights_only=True)
        )
        self.v9_model.eval()
        
        self.ref_centroid_t = torch.FloatTensor(self.reference_centroid).to(self.device)
        
        logger.info("Loaded V9 model")
    
    def _load_trajectory_net(self):
        """Load trajectory adjustment network."""
        try:
            self.trajectory_net = TrajectoryAdjustment(
                base_feature_dim=self.base_feature_dim
            ).to(self.device)
            
            model_path = self.model_dir / "trajectory_v10.pt"
            if model_path.exists():
                self.trajectory_net.load_state_dict(
                    torch.load(model_path, weights_only=True)
                )
                self.trajectory_net.eval()
                logger.info("Loaded trajectory adjustment network")
            else:
                logger.info("No trajectory network found, using V9 only")
                self.trajectory_net = None
        except Exception as e:
            logger.warning(f"Could not load trajectory network: {e}")
            self.trajectory_net = None
    
    def compute_v9_scores(self, windowed: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Compute V9 quality and reference similarity scores."""
        with torch.no_grad():
            quality_logits, style_emb = self.v9_model(windowed)
            quality = torch.sigmoid(quality_logits).squeeze().cpu().numpy()
            
            ref_sim = torch.mm(style_emb, self.ref_centroid_t.unsqueeze(1)).squeeze()
            ref_sim = ((ref_sim + 1) / 2).cpu().numpy()
            
        return quality, ref_sim
    
    def compute_trajectory_adjustments(
        self,
        windowed: torch.Tensor,
        energies: np.ndarray,
        v9_scores: np.ndarray
    ) -> np.ndarray:
        """
        Compute heuristic trajectory-aware score adjustments.
        
        These encode musical editing principles:
        1. Transition smoothness - prefer segments that flow well from previous kept
        2. Energy arc - encourage build-up and release patterns
        3. Pacing - penalize too many consecutive keeps or cuts
        4. Variety - penalize very similar consecutive segments
        """
        n_segments = len(windowed)
        adjustments = np.zeros(n_segments)
        
        # Get style embeddings for similarity computation
        with torch.no_grad():
            _, style_emb = self.v9_model(windowed)
            embeddings = style_emb.cpu().numpy()
        
        # Preliminary selection based on V9 scores
        threshold = np.percentile(v9_scores, 100 * (1 - 0.35))
        preliminary_kept = v9_scores >= threshold
        
        # Track state for trajectory adjustments
        kept_indices = []
        last_kept_energy = None
        consecutive_keeps = 0
        consecutive_cuts = 0
        
        for i in range(n_segments):
            adj = 0.0
            
            # 1. Pacing adjustment
            if preliminary_kept[i]:
                consecutive_keeps += 1
                consecutive_cuts = 0
                
                # Slight penalty for very long continuous sections (>8 segments ~ 40s)
                if consecutive_keeps > 8:
                    adj -= 0.05
                
                # Bonus for keeping after a gap (creates variety)
                if len(kept_indices) > 0 and i - kept_indices[-1] > 3:
                    adj += 0.05
                    
            else:
                consecutive_cuts += 1
                consecutive_keeps = 0
                
                # Penalty for cutting high-quality segments after long gap
                if consecutive_cuts > 5 and v9_scores[i] > 0.8:
                    adj += 0.08  # Boost to encourage keeping
            
            # 2. Energy arc adjustment
            if preliminary_kept[i]:
                current_energy = energies[i]
                
                if last_kept_energy is not None:
                    energy_delta = current_energy - last_kept_energy
                    
                    # Bonus for energy variety (not monotonous)
                    if abs(energy_delta) > 0.1:
                        adj += 0.03
                    
                    # Bonus for gradual builds (rising energy)
                    if 0.05 < energy_delta < 0.3:
                        adj += 0.02
                
                last_kept_energy = current_energy
                kept_indices.append(i)
            
            # 3. Similarity adjustment (avoid repetition)
            if preliminary_kept[i] and len(kept_indices) >= 2:
                # Check similarity to recently kept segments
                recent_kept = kept_indices[-3:] if len(kept_indices) >= 3 else kept_indices
                recent_embeddings = embeddings[recent_kept[:-1]]  # Exclude current
                current_emb = embeddings[i]
                
                similarities = recent_embeddings @ current_emb
                max_sim = similarities.max() if len(similarities) > 0 else 0
                
                # Penalty for very similar to recent (repetitive)
                if max_sim > 0.95:
                    adj -= 0.1
                elif max_sim > 0.9:
                    adj -= 0.05
                # Bonus for moderate similarity (cohesive but not repetitive)
                elif 0.6 < max_sim < 0.85:
                    adj += 0.02
            
            # 4. Position-based adjustment
            rel_pos = i / n_segments
            
            # Slight boost for strong openings and endings
            if rel_pos < 0.1 or rel_pos > 0.9:
                if v9_scores[i] > np.percentile(v9_scores, 70):
                    adj += 0.03
            
            # Ensure we have some content in the middle
            if 0.3 < rel_pos < 0.7:
                if len([k for k in kept_indices if 0.3 < k/n_segments < 0.7]) < 5:
                    if v9_scores[i] > np.percentile(v9_scores, 50):
                        adj += 0.02
            
            adjustments[i] = adj
        
        return adjustments
    
    def process_track(
        self,
        audio_path: str,
        output_path: str,
        keep_ratio: float = 0.35,
        segment_duration: float = 5.0,
        hop_duration: float = 2.5,
        crossfade_duration: float = 0.5,
        beat_align: bool = True,
        phrase_detect: bool = True,
        use_trajectory: bool = True,  # Whether to use trajectory adjustments
    ) -> Dict:
        """
        Process track with V9 + trajectory adjustments.
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
        energies_norm = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
        
        windowed_t = torch.FloatTensor(windowed).to(self.device)
        
        # Get V9 scores
        print("Computing V9 scores...")
        quality, ref_sim = self.compute_v9_scores(windowed_t)
        
        v9_combined = quality + self.similarity_weight * ref_sim
        v9_combined = v9_combined / (1 + self.similarity_weight)
        
        # Get trajectory adjustments (heuristic-based, no neural network needed)
        trajectory_adj = np.zeros(len(windowed))
        if use_trajectory:
            print("Computing trajectory adjustments...")
            trajectory_adj = self.compute_trajectory_adjustments(
                windowed_t, energies_norm, v9_combined
            )
        
        # Final scores
        final_scores = v9_combined + trajectory_adj
        
        # Select segments
        n_keep = max(1, int(len(final_scores) * keep_ratio))
        threshold_idx = np.argsort(final_scores)[-n_keep]
        adaptive_threshold = final_scores[threshold_idx]
        
        results = [(s, e, float(f), float(v), float(t)) 
                   for (s, e), f, v, t in zip(times, final_scores, v9_combined, trajectory_adj)]
        kept = [(s, e, f, v, t) for s, e, f, v, t in results if f >= adaptive_threshold]
        
        if len(kept) == 0:
            return {'success': False, 'error': 'No segments kept'}
        
        # Merge adjacent segments
        merged = []
        cs, ce, _, _, _ = kept[0]
        for s, e, f, v, t in kept[1:]:
            if s <= ce + hop_duration:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
        
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
        
        return {
            'success': True,
            'input_duration': duration,
            'output_duration': len(output) / sr,
            'n_segments': len(features),
            'n_kept': len(kept),
            'n_regions': len(merged),
            'keep_ratio_actual': len(kept) / len(features),
            'score_stats': {
                'final': {'min': float(final_scores.min()), 'max': float(final_scores.max()), 'mean': float(final_scores.mean())},
                'v9': {'min': float(v9_combined.min()), 'max': float(v9_combined.max()), 'mean': float(v9_combined.mean())},
                'trajectory': {'min': float(trajectory_adj.min()), 'max': float(trajectory_adj.max()), 'mean': float(trajectory_adj.mean())},
                'quality': {'min': float(quality.min()), 'max': float(quality.max()), 'mean': float(quality.mean())},
                'ref_sim': {'min': float(ref_sim.min()), 'max': float(ref_sim.max()), 'mean': float(ref_sim.mean())},
            },
            'n_beats': len(beat_times),
            'n_phrases': len(phrase_times),
        }


# =============================================================================
# HELPER FUNCTIONS
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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # V10 Editor - uses V9 as base, ready for trajectory adjustment training
    editor = V10Editor("./models")
    
    print("\nTesting V10 on walkerjam...")
    result = editor.process_track(
        './training_data/test_input/walkerjam.wav',
        './training_data/test_input/walkerjam_v10_output.wav',
        keep_ratio=0.35,
        use_trajectory=False  # No trajectory net trained yet
    )
    
    if result['success']:
        print(f"Input: {result['input_duration']/60:.1f} min")
        print(f"Output: {result['output_duration']/60:.1f} min")
        print(f"Kept: {result['keep_ratio_actual']*100:.1f}%")
        print(f"Regions: {result['n_regions']}")
        print(f"Score stats:")
        print(f"  V9: min={result['score_stats']['v9']['min']:.3f} max={result['score_stats']['v9']['max']:.3f} mean={result['score_stats']['v9']['mean']:.3f}")
        print(f"  Quality: min={result['score_stats']['quality']['min']:.3f} max={result['score_stats']['quality']['max']:.3f} mean={result['score_stats']['quality']['mean']:.3f}")
        print(f"  Ref Sim: min={result['score_stats']['ref_sim']['min']:.3f} max={result['score_stats']['ref_sim']['max']:.3f} mean={result['score_stats']['ref_sim']['mean']:.3f}")
