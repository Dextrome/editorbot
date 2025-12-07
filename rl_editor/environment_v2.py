"""V2 Environment with expanded action space and longer-horizon rewards.

Key differences from V1:
1. Section-level actions (KEEP_PHRASE, CUT_PHRASE, etc.)
2. Minimal step rewards - main reward at episode end
3. Harder policy problem (must learn musical structure)
4. Episode-level audio quality assessment
"""

import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional, Tuple, Dict, List
import numpy as np
import logging

from .config import Config
from .actions_v2 import ActionSpaceV2, ActionV2, ActionTypeV2, EditHistoryV2
from .state import AudioState, StateRepresentation

logger = logging.getLogger(__name__)


class AudioEditingEnvV2(gym.Env):
    """V2 RL environment with section-level actions and episode rewards.
    
    Key design principles:
    1. Minimal step rewards (just progress tracking)
    2. Full reward computed at episode end from audio quality
    3. Section-level actions make each decision more impactful
    4. Policy must learn phrase structure, not just beat-level patterns
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        config: Config,
        audio_state: Optional[AudioState] = None,
    ) -> None:
        """Initialize V2 environment.
        
        Args:
            config: Configuration object
            audio_state: Optional audio state
        """
        super().__init__()
        self.config = config
        self.audio_state = audio_state
        
        # Components
        self.action_space_v2: Optional[ActionSpaceV2] = None
        self.observation_space: Optional[spaces.Box] = None
        self.state_rep: Optional[StateRepresentation] = None
        self.edit_history = EditHistoryV2()
        
        # Tracking
        self.current_step = 0
        self.current_beat = 0
        self.max_steps = 500  # Fewer steps since actions affect multiple beats
        self.episode_actions: List[ActionV2] = []
        self.step_rewards: List[float] = []  # Minimal step rewards
        
        # V2 specific: track section-level decisions
        self.n_section_decisions = 0
        self.n_beat_decisions = 0
        
        # Episode reward components (computed at end)
        self.episode_reward_breakdown: Dict[str, float] = {}
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        if options and "audio_state" in options:
            self.audio_state = options["audio_state"]
        
        if self.audio_state is None:
            raise ValueError("audio_state must be provided")
        
        # Initialize components
        if self.state_rep is None:
            self.state_rep = StateRepresentation(self.config)
        
        # Update state representation with actual beat feature dimension
        if self.audio_state.beat_features is not None and self.audio_state.beat_features.ndim > 1:
            actual_dim = self.audio_state.beat_features.shape[1]
            if actual_dim != self.state_rep.beat_feature_dim:
                self.state_rep.set_beat_feature_dim(actual_dim)
                self.observation_space = None
        
        # Initialize action space
        n_beats = len(self.audio_state.beat_times)
        self.action_space_v2 = ActionSpaceV2(n_beats=n_beats)
        self.action_space_v2.reset()
        
        # Gymnasium action space (discrete)
        self.action_space = spaces.Discrete(ActionSpaceV2.N_ACTIONS)
        
        # Observation space
        if self.observation_space is None:
            obs_dim = self.state_rep.feature_dim
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        
        # Reset tracking
        self.edit_history = EditHistoryV2()
        self.current_step = 0
        self.current_beat = 0
        self.episode_actions = []
        self.step_rewards = []
        self.n_section_decisions = 0
        self.n_beat_decisions = 0
        self.episode_reward_breakdown = {}
        
        # Reset audio state beat index
        self.audio_state.beat_index = 0
        
        obs = self._get_observation()
        
        info = {
            "step": 0,
            "beat": 0,
            "n_beats": n_beats,
            "action_mask": self.action_space_v2.get_action_mask(0, self.edit_history).tolist(),
        }
        
        return obs, info
    
    def set_audio_state(self, audio_state: AudioState) -> None:
        """Set audio state for the environment.
        
        This allows reusing the environment instance with different audio.
        Call reset() after setting the audio state.
        """
        self.audio_state = audio_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step.
        
        Key difference from V1: minimal step reward, main reward at episode end.
        """
        if self.audio_state is None or self.action_space_v2 is None:
            raise RuntimeError("Environment not reset")
        
        # Decode action
        try:
            action_obj = self.action_space_v2.decode_action(action, self.current_beat)
        except ValueError as e:
            logger.error(f"Invalid action: {e}")
            return self._get_observation(), -1.0, False, True, {}
        
        # Apply action
        n_beats_advanced = self._apply_action(action_obj)
        self.episode_actions.append(action_obj)
        
        # Track decision type
        if action_obj.n_beats_affected > 1:
            self.n_section_decisions += 1
        else:
            self.n_beat_decisions += 1
        
        # Advance position
        self.current_beat += n_beats_advanced
        self.audio_state.beat_index = min(self.current_beat, len(self.audio_state.beat_times) - 1)
        
        # Minimal step reward (just progress tracking)
        step_reward = self._compute_minimal_step_reward(action_obj)
        self.step_rewards.append(step_reward)
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # If episode ended, compute the big reward
        episode_reward = 0.0
        if terminated or truncated:
            episode_reward = self._compute_episode_reward()
        
        # Total reward = minimal step reward + episode reward
        total_reward = step_reward + episode_reward
        
        obs = self._get_observation()
        
        info = {
            "step": self.current_step,
            "beat": self.current_beat,
            "action_type": action_obj.action_type.name,
            "n_beats_affected": action_obj.n_beats_affected,
            "n_kept": len(self.edit_history.kept_beats),
            "n_cut": len(self.edit_history.cut_beats),
            "keep_ratio": self.edit_history.get_keep_ratio(),
            "n_section_decisions": self.n_section_decisions,
            "n_beat_decisions": self.n_beat_decisions,
            "episode_reward": episode_reward if (terminated or truncated) else 0.0,
            "action_mask": self.action_space_v2.get_action_mask(self.current_beat, self.edit_history).tolist(),
            # Note: beat_times/beat_features not included by default (too large for IPC)
            # Access directly via env.audio_state for threading mode
        }
        
        if terminated or truncated:
            info["reward_breakdown"] = self.episode_reward_breakdown
            info["total_episode_reward"] = sum(self.step_rewards) + episode_reward
        
        self.current_step += 1
        
        return obs, total_reward, terminated, truncated, info
    
    def _apply_action(self, action: ActionV2) -> int:
        """Apply action to edit history. Returns number of beats to advance."""
        
        action_type = action.action_type
        beat_idx = action.beat_index
        n_beats = action.n_beats_affected
        
        if action_type == ActionTypeV2.KEEP_BEAT:
            self.edit_history.add_keep(beat_idx)
            return 1
        
        elif action_type == ActionTypeV2.CUT_BEAT:
            self.edit_history.add_cut(beat_idx)
            return 1
        
        elif action_type == ActionTypeV2.KEEP_BAR:
            self.edit_history.add_keep_section(beat_idx, n_beats)
            return n_beats
        
        elif action_type == ActionTypeV2.CUT_BAR:
            self.edit_history.add_cut_section(beat_idx, n_beats)
            return n_beats
        
        elif action_type == ActionTypeV2.KEEP_PHRASE:
            self.edit_history.add_keep_section(beat_idx, n_beats)
            return n_beats
        
        elif action_type == ActionTypeV2.CUT_PHRASE:
            self.edit_history.add_cut_section(beat_idx, n_beats)
            return n_beats
        
        elif action_type == ActionTypeV2.LOOP_2X:
            self.edit_history.add_loop(beat_idx, 2)
            return 1
        
        elif action_type == ActionTypeV2.LOOP_4X:
            self.edit_history.add_loop(beat_idx, 4)
            return 1
        
        elif action_type == ActionTypeV2.LOOP_BAR_2X:
            self.edit_history.add_section_loop(beat_idx, n_beats, 2)
            return n_beats
        
        elif action_type == ActionTypeV2.JUMP_BACK_4:
            # Jump back but don't re-edit (just mark for non-linear playback)
            target = max(0, beat_idx - 4)
            self.edit_history.add_jump(beat_idx, target)
            return 1  # Still advance (jump is for rendering)
        
        elif action_type == ActionTypeV2.JUMP_BACK_8:
            target = max(0, beat_idx - 8)
            self.edit_history.add_jump(beat_idx, target)
            return 1
        
        elif action_type == ActionTypeV2.MARK_SOFT_TRANSITION:
            self.edit_history.set_transition_marker(beat_idx, 'soft')
            return 0  # No beat advance
        
        elif action_type == ActionTypeV2.MARK_HARD_CUT:
            self.edit_history.set_transition_marker(beat_idx, 'hard')
            return 0
        
        return 1  # Default advance
    
    def _compute_minimal_step_reward(self, action: ActionV2) -> float:
        """Compute step reward.
        
        MONTE CARLO MODE: Zero step rewards.
        All learning signal comes from episode end.
        This forces true multi-step credit assignment.
        """
        # Pure Monte Carlo = NO step rewards
        return 0.0
    
    def _compute_episode_reward(self) -> float:
        """Compute episode reward based ONLY on audio quality.
        
        PURE AUDIO QUALITY - NO GROUND TRUTH LABELS.
        
        The model must learn what SOUNDS good, not what MATCHES human edits.
        Components:
        1. Audio quality (clicks, smoothness)
        2. Energy consistency (no jarring jumps)
        3. Duration target (right length)
        4. Section coherence (kept beats are related)
        5. Phrase alignment (cuts at musical boundaries)
        
        NO: Ground truth F1, label matching
        """
        self.episode_reward_breakdown = {}
        
        n_beats = len(self.audio_state.beat_times)
        n_kept = len(self.edit_history.kept_beats)
        n_cut = len(self.edit_history.cut_beats)
        n_edited = n_kept + n_cut
        
        # Catastrophic failure checks
        if n_edited == 0:
            self.episode_reward_breakdown["failure"] = "no_edits"
            return -100.0
        
        keep_ratio = n_kept / n_edited if n_edited > 0 else 0.0
        
        if keep_ratio < 0.05:  # Kept almost nothing
            self.episode_reward_breakdown["failure"] = "too_little_kept"
            return -50.0
        
        if keep_ratio > 0.95:  # Kept almost everything
            self.episode_reward_breakdown["failure"] = "too_much_kept"
            return -50.0
        
        # === AUDIO QUALITY METRICS ===
        total_score = 0.0
        total_weight = 0.0
        
        # 1. Build and analyze actual edited audio (MOST IMPORTANT)
        audio_quality = self._compute_audio_quality_score()
        weight_audio = 3.0  # Highest weight - actual audio quality
        total_score += weight_audio * audio_quality
        total_weight += weight_audio
        self.episode_reward_breakdown["audio_quality"] = audio_quality
        
        # 2. Duration target (hit ~35% keep ratio)
        target = self.config.reward.target_keep_ratio
        deviation = abs(keep_ratio - target)
        duration_score = max(0.0, 1.0 - deviation * 2.0)  # Linear penalty
        weight_duration = 1.5
        total_score += weight_duration * duration_score
        total_weight += weight_duration
        self.episode_reward_breakdown["duration_score"] = duration_score
        self.episode_reward_breakdown["keep_ratio"] = keep_ratio
        
        # 3. Energy consistency in edited audio
        energy_score = self._compute_edited_energy_consistency()
        weight_energy = 2.0
        total_score += weight_energy * energy_score
        total_weight += weight_energy
        self.episode_reward_breakdown["energy_consistency"] = energy_score
        
        # 4. Section coherence (reward keeping related beats together)
        coherence_score = self._compute_section_coherence()
        weight_coherence = 1.5
        total_score += weight_coherence * coherence_score
        total_weight += weight_coherence
        self.episode_reward_breakdown["section_coherence"] = coherence_score
        
        # 5. Phrase alignment (cuts at musical boundaries)
        phrase_score = self._compute_phrase_alignment()
        weight_phrase = 1.0
        total_score += weight_phrase * phrase_score
        total_weight += weight_phrase
        self.episode_reward_breakdown["phrase_alignment"] = phrase_score
        
        # 6. Flow continuity (smooth feature transitions)
        flow_score = self._compute_flow_score()
        weight_flow = 1.0
        total_score += weight_flow * flow_score
        total_weight += weight_flow
        self.episode_reward_breakdown["flow_continuity"] = flow_score
        
        # Normalize and scale
        normalized = total_score / total_weight if total_weight > 0 else 0.0
        
        # Scale to meaningful range (0-100)
        episode_reward = normalized * 100.0
        
        self.episode_reward_breakdown["total_normalized"] = normalized
        self.episode_reward_breakdown["total_scaled"] = episode_reward
        
        logger.debug(f"Episode reward: {episode_reward:.2f} | breakdown: {self.episode_reward_breakdown}")
        
        return episode_reward
    
    def _compute_edited_energy_consistency(self) -> float:
        """Compute energy consistency of the edited audio output."""
        if self.audio_state.raw_audio is None:
            return 0.5
        
        try:
            edited_audio = self._build_edited_audio()
            if edited_audio is None or len(edited_audio) < 1000:
                return 0.0
            
            return self._compute_energy_consistency(edited_audio)
        except Exception as e:
            logger.debug(f"Energy consistency computation failed: {e}")
            return 0.5
    
    def _compute_ground_truth_f1(self) -> float:
        """Compute F1 score against ground truth labels.
        
        NOTE: This is kept for logging/debugging but NOT used in rewards anymore.
        """
        target = self.audio_state.target_labels
        if target is None:
            return 0.0
        n_beats = len(target)
        
        # Build prediction
        pred = np.zeros(n_beats)
        for beat_idx in self.edit_history.kept_beats:
            if 0 <= beat_idx < n_beats:
                pred[beat_idx] = 1.0
        
        # F1 calculation
        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        fn = np.sum((pred == 0) & (target == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    
    def _compute_section_coherence(self) -> float:
        """Reward for keeping consecutive beats together."""
        kept = sorted(self.edit_history.kept_beats)
        if len(kept) < 2:
            return 0.5
        
        n_consecutive = sum(1 for i in range(len(kept) - 1) if kept[i+1] == kept[i] + 1)
        max_consecutive = len(kept) - 1
        
        if max_consecutive == 0:
            return 1.0
        
        return 0.3 + 0.7 * (n_consecutive / max_consecutive)
    
    def _compute_phrase_alignment(self) -> float:
        """Reward for cutting at phrase boundaries (every 4 or 8 beats)."""
        kept = sorted(self.edit_history.kept_beats)
        if len(kept) < 2:
            return 0.5
        
        # Find edit boundaries (where kept beats are non-consecutive)
        n_edit_points = 0
        good_cuts = 0
        
        for i in range(len(kept) - 1):
            if kept[i+1] != kept[i] + 1:
                n_edit_points += 1
                # Check if cut is at phrase boundary
                cut_at = kept[i] + 1  # First cut beat
                if cut_at % 8 == 0:
                    good_cuts += 1.0
                elif cut_at % 4 == 0:
                    good_cuts += 0.7
                elif cut_at % 2 == 0:
                    good_cuts += 0.3
        
        if n_edit_points == 0:
            return 1.0
        
        return good_cuts / n_edit_points
    
    def _compute_decision_quality(self) -> float:
        """Reward for using appropriate decision granularity.
        
        Section-level actions should be used for homogeneous sections.
        Beat-level actions for fine control at boundaries.
        """
        total_decisions = self.n_section_decisions + self.n_beat_decisions
        if total_decisions == 0:
            return 0.0
        
        # Ideal: mix of both (not all section, not all beat)
        section_ratio = self.n_section_decisions / total_decisions
        
        # Reward moderate use of section decisions (30-70% is good)
        if 0.3 <= section_ratio <= 0.7:
            return 1.0
        elif 0.2 <= section_ratio <= 0.8:
            return 0.7
        elif 0.1 <= section_ratio <= 0.9:
            return 0.4
        else:
            return 0.2  # Too extreme
    
    def _compute_flow_score(self) -> float:
        """Compute flow continuity from beat features."""
        if self.audio_state.beat_features is None:
            return 0.5
        
        kept = sorted(self.edit_history.kept_beats)
        if len(kept) < 2:
            return 0.5
        
        features = self.audio_state.beat_features
        n_beats = len(features)
        
        # Compute distances between consecutive kept beats
        edit_distances = []
        for i in range(len(kept) - 1):
            if kept[i] < n_beats and kept[i+1] < n_beats:
                dist = np.linalg.norm(features[kept[i]] - features[kept[i+1]])
                edit_distances.append(dist)
        
        if not edit_distances:
            return 0.5
        
        # Compare to original consecutive distances
        orig_distances = []
        for i in range(min(n_beats - 1, 100)):
            dist = np.linalg.norm(features[i] - features[i+1])
            orig_distances.append(dist)
        
        if not orig_distances:
            return 0.5
        
        mean_edit = np.mean(edit_distances)
        mean_orig = np.mean(orig_distances)
        
        if mean_orig > 0:
            ratio = mean_edit / (mean_orig + 1e-6)
            return max(0.0, min(1.0, 2.0 - ratio))  # Score 1.0 if ratio <= 1, decreasing above
        
        return 0.5
    
    def _compute_audio_quality_score(self) -> float:
        """Compute audio quality by building edited audio and analyzing."""
        if self.audio_state.raw_audio is None:
            return 0.5
        
        try:
            edited_audio = self._build_edited_audio()
            if edited_audio is None or len(edited_audio) == 0:
                return 0.0
            
            # Check for clicks/pops (high-frequency transients)
            click_score = self._detect_clicks(edited_audio)
            
            # Check energy consistency
            energy_score = self._compute_energy_consistency(edited_audio)
            
            return 0.5 * click_score + 0.5 * energy_score
            
        except Exception as e:
            logger.debug(f"Audio quality computation failed: {e}")
            return 0.5
    
    def _build_edited_audio(self, crossfade_ms: float = 50.0) -> Optional[np.ndarray]:
        """Build edited audio from edit history."""
        audio = self.audio_state.raw_audio
        sr = self.audio_state.sample_rate
        beat_times = self.audio_state.beat_times
        
        beat_samples = (beat_times * sr).astype(int)
        beat_samples = np.append(beat_samples, len(audio))
        
        crossfade_samples = int(crossfade_ms * sr / 1000)
        
        segments = []
        kept_sorted = sorted(self.edit_history.kept_beats)
        
        for beat_idx in kept_sorted:
            if beat_idx >= len(beat_times):
                continue
            
            start = beat_samples[beat_idx]
            end = beat_samples[beat_idx + 1] if beat_idx + 1 < len(beat_samples) else len(audio)
            
            segment = audio[start:end].copy()
            
            # Handle loops
            if beat_idx in self.edit_history.looped_beats:
                n_times = self.edit_history.looped_beats[beat_idx]
                segment = np.tile(segment, n_times)
            
            segments.append(segment)
        
        if not segments:
            return None
        
        # Simple concatenation (could add crossfades)
        return np.concatenate(segments)
    
    def _detect_clicks(self, audio: np.ndarray) -> float:
        """Detect clicks/pops in audio. Returns 1.0 if clean, 0.0 if many clicks."""
        # Simple approach: count large sample-to-sample jumps
        diff = np.abs(np.diff(audio))
        threshold = np.percentile(diff, 99)
        n_clicks = np.sum(diff > threshold * 3)
        
        # Normalize by audio length
        click_density = n_clicks / len(audio)
        
        # Score: 1.0 if no clicks, decreasing
        return max(0.0, 1.0 - click_density * 10000)
    
    def _compute_energy_consistency(self, audio: np.ndarray) -> float:
        """Compute energy consistency (smooth dynamics)."""
        # Compute RMS energy in windows
        window_size = int(self.audio_state.sample_rate * 0.1)  # 100ms windows
        n_windows = len(audio) // window_size
        
        if n_windows < 2:
            return 0.5
        
        energies = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            rms = np.sqrt(np.mean(audio[start:end] ** 2))
            energies.append(rms)
        
        energies = np.array(energies)
        
        # Compute energy variance (lower = more consistent)
        if np.mean(energies) > 0:
            cv = np.std(energies) / np.mean(energies)  # Coefficient of variation
            return max(0.0, 1.0 - cv)
        
        return 0.5
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.audio_state is None or self.state_rep is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Update audio state beat index
        self.audio_state.beat_index = self.current_beat
        
        # Create temporary EditHistory for state rep (it expects the old format)
        from .state import EditHistory
        temp_history = EditHistory()
        # Convert sets to lists for compatibility with EditHistory
        temp_history.kept_beats = list(self.edit_history.kept_beats)
        temp_history.cut_beats = list(self.edit_history.cut_beats)
        temp_history.looped_beats = dict(self.edit_history.looped_beats)
        
        total_duration = self.audio_state.beat_times[-1] if len(self.audio_state.beat_times) > 0 else 1.0
        
        return self.state_rep.construct_observation(
            self.audio_state,
            temp_history,
            remaining_duration=self._get_remaining_duration(),
            total_duration=total_duration,
        )
    
    def _get_remaining_duration(self) -> float:
        """Get remaining duration budget."""
        if len(self.audio_state.beat_times) == 0:
            return 0.0
        
        total_duration = self.audio_state.beat_times[-1]
        target_duration = total_duration * self.config.reward.target_keep_ratio
        
        # Estimate kept duration
        kept_duration = 0.0
        beat_times = self.audio_state.beat_times
        for beat_idx in self.edit_history.kept_beats:
            if beat_idx < len(beat_times) - 1:
                kept_duration += beat_times[beat_idx + 1] - beat_times[beat_idx]
        
        return max(0.0, target_duration - kept_duration)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        if self.audio_state is None:
            return False
        
        # Terminate if we've processed all beats
        n_beats = len(self.audio_state.beat_times)
        return self.current_beat >= n_beats
    
    def get_action_mask(self) -> np.ndarray:
        """Get current action mask (for masked action selection)."""
        if self.action_space_v2 is None:
            return np.ones(ActionSpaceV2.N_ACTIONS, dtype=bool)
        return self.action_space_v2.get_action_mask(self.current_beat, self.edit_history)
    
    def render(self) -> None:
        """Render environment state."""
        if self.audio_state is None:
            return
        
        n_beats = len(self.audio_state.beat_times)
        logger.info(
            f"Step {self.current_step}: Beat {self.current_beat}/{n_beats}, "
            f"Kept: {len(self.edit_history.kept_beats)}, "
            f"Cut: {len(self.edit_history.cut_beats)}, "
            f"Section decisions: {self.n_section_decisions}, "
            f"Beat decisions: {self.n_beat_decisions}"
        )
    
    def close(self) -> None:
        """Close environment."""
        pass
