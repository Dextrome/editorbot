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
            # Transition markers now IMPLICITLY keep the beat and advance
            # This prevents spam: can't just mark transitions without processing beats
            self.edit_history.add_keep(beat_idx)  # Keep the beat
            self.edit_history.set_transition_marker(beat_idx, 'soft')
            return 1  # ADVANCE by 1 beat (no stalling!)
        
        elif action_type == ActionTypeV2.MARK_HARD_CUT:
            # Hard cut marker also keeps beat and advances
            self.edit_history.add_keep(beat_idx)
            self.edit_history.set_transition_marker(beat_idx, 'hard')
            return 1  # ADVANCE by 1 beat
        
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
        
        BALANCED REWARD DESIGN - Anti-Exploitation V2C (Soft Penalties)
        
        Key insight: Previous rewards allowed exploitation because:
        1. Model learned to loop instead of cut (looping = safe, cutting = risky)
        2. Keep ratio constraint wasn't strict enough (80% still allowed)
        3. No penalty for making output LONGER than input
        
        New design: SOFT PENALTIES with HARD FLOORS
        - Soft penalties give gradient signal toward valid region
        - Hard floors only for truly degenerate policies (< 5% cut)
        - Penalties scale with how far from target
        """
        self.episode_reward_breakdown = {}
        
        n_beats = len(self.audio_state.beat_times)
        n_kept = len(self.edit_history.kept_beats)
        n_cut = len(self.edit_history.cut_beats)
        n_edited = n_kept + n_cut
        
        # Count loop actions
        loop_action_types = {
            ActionTypeV2.LOOP_2X, ActionTypeV2.LOOP_4X, ActionTypeV2.LOOP_BAR_2X,
            ActionTypeV2.JUMP_BACK_4, ActionTypeV2.JUMP_BACK_8
        }
        n_loops = sum(1 for a in self.episode_actions if a.action_type in loop_action_types)
        
        # === HARD FLOOR (only for truly broken policies) ===
        if n_edited == 0:
            self.episode_reward_breakdown["failure"] = "no_edits"
            return -100.0
        
        keep_ratio = n_kept / n_edited if n_edited > 0 else 0.0
        cut_ratio = n_cut / n_edited if n_edited > 0 else 0.0
        self.episode_reward_breakdown["keep_ratio"] = keep_ratio
        self.episode_reward_breakdown["cut_ratio"] = cut_ratio
        self.episode_reward_breakdown["n_loops"] = n_loops
        
        # Hard floor: Must have SOME cuts and keeps (prevents degenerate all-one-action)
        if n_kept < 3 or n_cut < 3:
            self.episode_reward_breakdown["failure"] = "degenerate_policy"
            return -50.0
        
        # === SOFT PENALTIES (scale with violation severity) ===
        penalties = {}
        
        # Penalty 1: CUT DEFICIT - Must cut at least 30% target
        # Scales from 0 (at 30%+ cuts) to -40 (at 0% cuts)
        target_cut_ratio = 0.30
        if cut_ratio < target_cut_ratio:
            cut_deficit = target_cut_ratio - cut_ratio  # 0 to 0.30
            # Quadratic penalty: small violations = small penalty, large = big
            penalties["cut_deficit"] = -(cut_deficit / target_cut_ratio) ** 2 * 40.0
        else:
            penalties["cut_deficit"] = 0.0
        
        # Penalty 2: EXCESSIVE CUTS - Don't cut more than 80%
        if cut_ratio > 0.80:
            excess_cuts = cut_ratio - 0.80  # 0 to 0.20
            penalties["excess_cuts"] = -(excess_cuts / 0.20) ** 2 * 30.0
        else:
            penalties["excess_cuts"] = 0.0
        
        # Penalty 3: LOOP BUDGET - 1 loop allowed per 4 cuts
        # Excess loops get penalized, not zeroed
        max_loops = n_cut // 4
        if n_loops > max_loops:
            excess_loops = n_loops - max_loops
            # -5 points per excess loop, capped at -30
            penalties["excess_loops"] = -min(excess_loops * 5.0, 30.0)
        else:
            penalties["excess_loops"] = 0.0
        
        # Penalty 4: OUTPUT DURATION - Target 35% of input (for 10-30min → 3-9min edits)
        total_looped_beats = sum(self.edit_history.looped_beats.values()) if self.edit_history.looped_beats else 0
        estimated_output_beats = n_kept + total_looped_beats
        duration_ratio = estimated_output_beats / n_beats if n_beats > 0 else 1.0
        self.episode_reward_breakdown["duration_ratio"] = duration_ratio
        
        # Target: 35% duration, penalty starts at 50%
        # Strong penalty for anything over 50% - we want SHORT edits
        target_duration = 0.35
        if duration_ratio > 0.50:
            # Quadratic penalty: -60 points at 100%, -0 at 50%
            excess_duration = duration_ratio - 0.50  # 0 to 0.50
            penalties["excess_duration"] = -(excess_duration / 0.50) ** 2 * 60.0
        elif duration_ratio > target_duration:
            # Light penalty between 35-50%
            excess_duration = duration_ratio - target_duration  # 0 to 0.15
            penalties["excess_duration"] = -(excess_duration / 0.15) * 10.0
        else:
            # Bonus for hitting target (up to +15 at exactly 35%, less if too short)
            if duration_ratio >= 0.20:
                # Sweet spot: 20-35% gets bonus
                closeness = 1.0 - abs(duration_ratio - target_duration) / 0.15
                penalties["excess_duration"] = closeness * 15.0
            else:
                # Too short (<20%) - small penalty
                penalties["excess_duration"] = -((0.20 - duration_ratio) / 0.20) * 20.0
        
        # Target: 25-45% keep ratio (centered on 35%)
        target_ratio = self.config.reward.target_keep_ratio  # 0.35
        
        # === INDEPENDENT REWARD COMPONENTS ===
        # Each component is scored 0-1 and contributes additively
        # Total possible: 100 points, but practically ~60-80 is good
        
        components = {}
        
        # Component 1: Keep Ratio Score (25 points max)
        # Gaussian centered on target, with reasonable width
        ratio_deviation = abs(keep_ratio - target_ratio)
        # Score: 1.0 at target, 0.5 at ±0.15, ~0 at ±0.30
        ratio_score = np.exp(-(ratio_deviation ** 2) / (2 * 0.10 ** 2))
        components["keep_ratio"] = ratio_score * 25.0
        
        # Component 2: Audio Quality (20 points max)
        audio_quality = self._compute_audio_quality_score()
        # Diminishing returns: sqrt to prevent exploitation
        components["audio_quality"] = np.sqrt(audio_quality) * 20.0
        
        # Component 3: Energy Consistency (15 points max)
        energy_score = self._compute_edited_energy_consistency()
        components["energy_consistency"] = np.sqrt(energy_score) * 15.0
        
        # Component 4: Cut Quality (15 points max)
        # Reward for cutting LOW quality beats, keeping HIGH quality
        cut_quality = self._compute_cut_quality()
        components["cut_quality"] = np.sqrt(cut_quality) * 15.0
        
        # Component 5: Edit Structure (10 points max)
        # Reward for having multiple kept sections (not one big chunk or scattered)
        structure_score = self._compute_edit_structure_score()
        components["edit_structure"] = structure_score * 10.0
        
        # Component 6: Phrase Alignment (10 points max)
        phrase_score = self._compute_phrase_alignment()
        components["phrase_alignment"] = phrase_score * 10.0
        
        # Component 7: Action Diversity Bonus (5 points max)
        # Penalize using only one type of action (prevents section-only exploit)
        diversity_score = self._compute_action_diversity()
        components["action_diversity"] = diversity_score * 5.0
        
        # === COMPUTE FINAL REWARD ===
        base_reward = sum(components.values())
        total_penalty = sum(penalties.values())
        total_reward = base_reward + total_penalty
        
        # Store breakdown for logging
        self.episode_reward_breakdown.update(components)
        self.episode_reward_breakdown.update(penalties)
        self.episode_reward_breakdown["base_reward"] = base_reward
        self.episode_reward_breakdown["total_penalty"] = total_penalty
        self.episode_reward_breakdown["total"] = total_reward
        
        # Debug logging
        logger.debug(f"Episode reward: {total_reward:.2f} | cut_ratio: {cut_ratio:.2%} | penalties: {penalties}")
        
        return total_reward
    
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
        """Reward for keeping COHERENT sections (similar features within kept regions).
        
        Instead of rewarding consecutive beats, we reward:
        1. Low variance within kept sections (coherent)
        2. High variance between kept sections (distinct sections)
        3. Cutting at feature boundaries (not mid-section)
        """
        kept = sorted(self.edit_history.kept_beats)
        if len(kept) < 2:
            return 0.5
        
        features = self.audio_state.beat_features
        if features is None:
            return 0.5
        
        n_beats = len(features)
        
        # Find kept sections (groups of consecutive kept beats)
        sections = []
        current_section = [kept[0]]
        
        for i in range(1, len(kept)):
            if kept[i] == kept[i-1] + 1:
                current_section.append(kept[i])
            else:
                if len(current_section) >= 1:
                    sections.append(current_section)
                current_section = [kept[i]]
        if current_section:
            sections.append(current_section)
        
        if not sections:
            return 0.5
        
        # 1. Within-section coherence: low feature variance within each section
        within_variances = []
        for section in sections:
            if len(section) >= 2:
                section_features = [features[b] for b in section if b < n_beats]
                if len(section_features) >= 2:
                    variance = np.var(section_features, axis=0).mean()
                    within_variances.append(variance)
        
        # 2. Between-section distinctness: sections should be different from each other
        section_means = []
        for section in sections:
            section_features = [features[b] for b in section if b < n_beats]
            if section_features:
                section_means.append(np.mean(section_features, axis=0))
        
        between_variance = 0.0
        if len(section_means) >= 2:
            between_variance = np.var(section_means, axis=0).mean()
        
        # Score: reward low within-section variance, high between-section variance
        avg_within = np.mean(within_variances) if within_variances else 0.0
        
        # Normalize scores (these are relative metrics)
        if avg_within > 0 and between_variance > 0:
            # Ratio of between/within variance - higher is better
            coherence_ratio = between_variance / (avg_within + 1e-6)
            coherence_score = min(1.0, coherence_ratio / 2.0)  # Cap at 1.0
        elif avg_within == 0:
            coherence_score = 0.8  # Perfect within-section coherence
        else:
            coherence_score = 0.4  # No between-section variance
        
        # 3. Bonus for having multiple distinct sections (not just one big chunk)
        n_sections = len(sections)
        section_bonus = min(0.2, n_sections * 0.05)  # Up to 0.2 bonus for 4+ sections
        
        return min(1.0, coherence_score + section_bonus)
    
    def _compute_cut_quality(self) -> float:
        """Reward for making GOOD cuts - cutting low-quality or repetitive sections.
        
        This explicitly rewards the act of cutting, not just penalizes not cutting.
        Good cuts are:
        1. Cuts at low-energy sections (silence, noise)
        2. Cuts at repetitive sections (similar to other parts)
        3. Cuts that improve overall variance (remove boring parts)
        """
        cut_beats = sorted(self.edit_history.cut_beats)
        kept_beats = sorted(self.edit_history.kept_beats)
        
        if len(cut_beats) == 0:
            return 0.0  # No cuts = no cut quality
        
        if len(kept_beats) == 0:
            return 0.0  # Cut everything = bad
        
        features = self.audio_state.beat_features
        if features is None:
            return 0.5
        
        n_beats = len(features)
        
        # 1. Energy-based cut quality: did we cut low-energy beats?
        # Use first few features as energy proxy (usually related to amplitude)
        energy_scores = []
        for beat in cut_beats:
            if beat < n_beats:
                # Lower energy = better to cut
                energy = np.linalg.norm(features[beat][:10]) if len(features[beat]) >= 10 else np.linalg.norm(features[beat])
                energy_scores.append(energy)
        
        kept_energies = []
        for beat in kept_beats:
            if beat < n_beats:
                energy = np.linalg.norm(features[beat][:10]) if len(features[beat]) >= 10 else np.linalg.norm(features[beat])
                kept_energies.append(energy)
        
        # Good cuts: cut beats have lower energy than kept beats on average
        if energy_scores and kept_energies:
            avg_cut_energy = np.mean(energy_scores)
            avg_kept_energy = np.mean(kept_energies)
            if avg_kept_energy > 0:
                # If cut energy < kept energy, we're cutting the right stuff
                energy_ratio = avg_cut_energy / (avg_kept_energy + 1e-6)
                energy_cut_score = max(0.0, min(1.0, 1.5 - energy_ratio))  # Higher score if cutting low energy
            else:
                energy_cut_score = 0.5
        else:
            energy_cut_score = 0.5
        
        # 2. Repetition-based: did we cut repetitive sections?
        # Check if cut beats are similar to other beats (repetitive = good to cut)
        similarity_scores = []
        for cut_beat in cut_beats[:20]:  # Sample for efficiency
            if cut_beat < n_beats:
                # Find similarity to other beats
                cut_feat = features[cut_beat]
                similarities = []
                for other in range(0, n_beats, 4):  # Sample every 4th beat
                    if other != cut_beat:
                        sim = 1.0 / (1.0 + np.linalg.norm(cut_feat - features[other]))
                        similarities.append(sim)
                if similarities:
                    similarity_scores.append(np.mean(similarities))
        
        # Higher similarity = more repetitive = better to cut
        if similarity_scores:
            avg_similarity = np.mean(similarity_scores)
            repetition_cut_score = min(1.0, avg_similarity * 3.0)  # Scale up
        else:
            repetition_cut_score = 0.5
        
        # 3. Cut ratio bonus: reward for achieving target cut amount
        total_beats = len(cut_beats) + len(kept_beats)
        cut_ratio = len(cut_beats) / total_beats if total_beats > 0 else 0
        target_cut = 1.0 - self.config.reward.target_keep_ratio  # ~0.65
        
        # Reward being close to target cut ratio
        cut_deviation = abs(cut_ratio - target_cut)
        cut_ratio_score = max(0.0, 1.0 - cut_deviation * 2.0)
        
        # Combine scores
        return 0.4 * energy_cut_score + 0.3 * repetition_cut_score + 0.3 * cut_ratio_score
    
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
    
    def _compute_edit_structure_score(self) -> float:
        """Reward for having good edit structure - multiple coherent sections.
        
        Bad: One big kept chunk + one big cut chunk (trivial)
        Bad: Scattered single beats kept (noisy)
        Good: 3-10 kept sections of 4+ beats each
        """
        kept = sorted(self.edit_history.kept_beats)
        if len(kept) < 4:
            return 0.1
        
        # Find contiguous kept sections
        sections = []
        current_section = [kept[0]]
        
        for i in range(1, len(kept)):
            if kept[i] == kept[i-1] + 1:
                current_section.append(kept[i])
            else:
                sections.append(current_section)
                current_section = [kept[i]]
        sections.append(current_section)
        
        n_sections = len(sections)
        section_lengths = [len(s) for s in sections]
        avg_length = np.mean(section_lengths)
        
        # Score based on number of sections (3-8 is ideal)
        if n_sections == 1:
            section_count_score = 0.2  # One big chunk = trivial
        elif n_sections == 2:
            section_count_score = 0.5
        elif 3 <= n_sections <= 8:
            section_count_score = 1.0  # Ideal range
        elif n_sections <= 12:
            section_count_score = 0.7
        else:
            section_count_score = 0.3  # Too fragmented
        
        # Score based on average section length (4-16 beats is ideal)
        if avg_length < 2:
            length_score = 0.2  # Too short = choppy
        elif avg_length < 4:
            length_score = 0.5
        elif 4 <= avg_length <= 16:
            length_score = 1.0  # Ideal phrase length
        elif avg_length <= 32:
            length_score = 0.7
        else:
            length_score = 0.4  # Too long sections
        
        return 0.5 * section_count_score + 0.5 * length_score
    
    def _compute_action_diversity(self) -> float:
        """Reward for using diverse action types - prevent single-action exploits.
        
        If model only uses one type of action (e.g., always KEEP_SECTION),
        it's likely exploiting rather than learning.
        """
        if not self.episode_actions:
            return 0.5
        
        from collections import Counter
        # Convert actions to integers for hashing
        # ActionV2 is a dataclass with action_type attribute (an IntEnum)
        action_ints = []
        for a in self.episode_actions:
            if hasattr(a, 'action_type'):
                action_ints.append(int(a.action_type))
            elif hasattr(a, 'value'):
                action_ints.append(a.value)
            elif isinstance(a, (int, float)):
                action_ints.append(int(a))
            else:
                # Fallback - try to get any numeric attribute
                action_ints.append(0)  # Default
        
        action_counts = Counter(action_ints)
        
        n_unique = len(action_counts)
        n_total = len(action_ints)
        
        if n_total < 3:
            return 0.5
        
        # Method 1: Count unique action types used
        # At least 3 different actions should be used
        uniqueness_score = min(1.0, (n_unique - 1) / 4.0)  # Score 1.0 at 5+ unique actions
        
        # Method 2: Entropy of action distribution
        probs = np.array(list(action_counts.values())) / n_total
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(13)  # 13 possible actions
        entropy_score = entropy / max_entropy
        
        # Method 3: No single action dominates (>70%)
        max_action_ratio = max(action_counts.values()) / n_total
        dominance_score = 1.0 if max_action_ratio < 0.5 else (1.0 - max_action_ratio)
        
        return 0.3 * uniqueness_score + 0.4 * entropy_score + 0.3 * dominance_score
    
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
