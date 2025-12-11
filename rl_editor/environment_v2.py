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
        
        elif action_type == ActionTypeV2.LOOP_BEAT:
            self.edit_history.add_loop(beat_idx, 2)
            return 1
        
        elif action_type == ActionTypeV2.LOOP_BAR:
            self.edit_history.add_section_loop(beat_idx, n_beats, 2)
            return n_beats
        
        elif action_type == ActionTypeV2.LOOP_PHRASE:
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
        
        elif action_type == ActionTypeV2.REORDER_BEAT:
            # Move this beat to the end of the output (deferred placement)
            self.edit_history.add_reorder(beat_idx, 1)
            return 1
        
        elif action_type == ActionTypeV2.REORDER_BAR:
            # Move this 4-beat bar to the end of the output
            self.edit_history.add_reorder(beat_idx, n_beats)
            return n_beats
        
        elif action_type == ActionTypeV2.REORDER_PHRASE:
            # Move this 8-beat phrase to the end of the output
            self.edit_history.add_reorder(beat_idx, n_beats)
            return n_beats
        
        return 1  # Default advance
    
    def _compute_minimal_step_reward(self, action: ActionV2) -> float:
        """Compute step reward with duration warning signal.
        
        Mostly Monte Carlo (episode-end rewards), BUT with one exception:
        Give immediate negative signal when approaching duration limit.
        This helps the model learn the constraint faster.
        """
        n_beats = len(self.audio_state.beat_times)
        if n_beats == 0:
            return 0.0
        
        # Calculate current duration ratio - must match episode calculation!
        # Output = kept + reordered + loop_extra + jumped
        n_kept = len(self.edit_history.kept_beats)
        
        # Reordered beats
        total_reordered = sum(n for _, n in self.edit_history.reordered_sections) if self.edit_history.reordered_sections else 0
        
        # Jumped beats (JUMP_BACK does not replay content on its own)
        total_jumped = sum(max(0, from_b - to_b) for from_b, to_b in self.edit_history.jump_points) if self.edit_history.jump_points else 0
        
        # Loop extra from both looped_beats AND looped_sections
        loop_extra_beats = sum(max(0, times - 1) for times in self.edit_history.looped_beats.values()) if self.edit_history.looped_beats else 0
        loop_extra_sections = sum((end - start) * max(0, times - 1) for start, end, times in self.edit_history.looped_sections) if self.edit_history.looped_sections else 0
        loop_extra = loop_extra_beats + loop_extra_sections
        
        estimated_output = n_kept + total_reordered + loop_extra # + total_jumped
        duration_ratio = estimated_output / n_beats
        reward = 0.0
        
        # STEP-LEVEL PENALTIES
        # 1. Give negative signal when approaching/exceeding 90%
        if duration_ratio > 0.90:
            # Already over limit - strong negative penalty every step
            reward -= 2.0
        elif duration_ratio > 0.80:
            # Warning zone - mild negative penalty
            reward -= 0.8
        elif duration_ratio > 0.70:
            # Approaching warning - very mild penalty
            reward -= 0.2
    
        # 2. Penalty for repeating the same action type too many times in a row
        # If the last 3 actions (including current) are the same type, subtract 0.2
        if len(self.episode_actions) >= 2:
            last_two = self.episode_actions[-2:]
            if all(a.action_type == action.action_type for a in last_two):
                reward -= 0.2


        # === INTERMEDIATE POSITIVE REWARDS ===
        # 1. Bonus for cutting at phrase boundaries (every 8th beat)
        if action.action_type in [ActionTypeV2.CUT_PHRASE, ActionTypeV2.CUT_BAR, ActionTypeV2.CUT_BEAT]:
            if action.beat_index % 8 == 0:
                reward += 0.4  # Phrase boundary cut
            elif action.beat_index % 4 == 0:
                reward += 0.1  # Bar boundary cut
        # 2. Bonus for using diverse actions (not repeating last action)
        if len(self.episode_actions) > 1:
            last_action = self.episode_actions[-1]
            if action.action_type != last_action.action_type:
                reward += 0.1
        # 3. Bonus for explicit keep actions (encourage agent to keep good content)
        if action.action_type in [ActionTypeV2.KEEP_BEAT, ActionTypeV2.KEEP_BAR, ActionTypeV2.KEEP_PHRASE]:
            reward += 0.1
        # 4. Bonus for hitting keep ratio target early in episode
        keep_ratio = self.edit_history.get_keep_ratio()
        target_keep = self.config.reward.target_keep_ratio if hasattr(self.config.reward, 'target_keep_ratio') else 0.35
        if 0.25 < keep_ratio < target_keep:
            reward += 0.1
        # 5. Bonus for using transition markers sparingly
        if action.action_type in [ActionTypeV2.MARK_SOFT_TRANSITION, ActionTypeV2.MARK_HARD_CUT]:
            n_markers = sum(1 for a in self.episode_actions if a.action_type in [ActionTypeV2.MARK_SOFT_TRANSITION, ActionTypeV2.MARK_HARD_CUT])
            if n_markers < 3:
                reward += 0.1
        # 6. Bonus for cutting when duration ratio is high but NOT if it's a good bar/phrase
        if (
            duration_ratio > 0.75
            and action.action_type in [ActionTypeV2.CUT_BEAT, ActionTypeV2.CUT_BAR, ActionTypeV2.CUT_PHRASE]
            and (action.beat_index % 8 != 0 and action.beat_index % 4 != 0)
        ):
            reward += 0.1
        # 7. Reward for using a section action after several beat actions
        # If the last 3 actions were beat-level and now a section action, add 0.2
        if len(self.episode_actions) >= 3:
            last_three = self.episode_actions[-3:]
            beat_actions = [ActionTypeV2.KEEP_BEAT, ActionTypeV2.CUT_BEAT, ActionTypeV2.LOOP_BEAT, ActionTypeV2.REORDER_BEAT]
            section_actions = [ActionTypeV2.KEEP_BAR, ActionTypeV2.KEEP_PHRASE, ActionTypeV2.CUT_BAR, ActionTypeV2.CUT_PHRASE, ActionTypeV2.LOOP_BAR, ActionTypeV2.LOOP_PHRASE, ActionTypeV2.REORDER_BAR, ActionTypeV2.REORDER_PHRASE]
            if all(a.action_type in beat_actions for a in last_three) and action.action_type in section_actions:
                reward += 0.1

        # 8. Reward for KEEP_PHRASE at phrase boundary when duration ratio is low
        # If action is KEEP_PHRASE at beat % 8 == 0 and duration_ratio < 0.5, add 0.2
        if (
            action.action_type == ActionTypeV2.KEEP_PHRASE
            and action.beat_index % 8 == 0
            and duration_ratio < 0.5
        ):
            reward += 0.1

        return reward
    
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
        
        # EXPLOIT FIX: Count looped beats as implicit keeps
        # Looping a beat without keeping it is still keeping content
        looped_beat_indices = set(self.edit_history.looped_beats.keys()) if self.edit_history.looped_beats else set()
        n_looped_unique = len(looped_beat_indices - self.edit_history.kept_beats)  # Looped but not explicitly kept
        n_effective_kept = n_kept + n_looped_unique  # For ratio calculations
        
        # Count loop actions
        loop_action_types = {
            ActionTypeV2.LOOP_BEAT, ActionTypeV2.LOOP_BAR, ActionTypeV2.LOOP_PHRASE,
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
        
        # Penalty 3: LOOP BUDGET - More lenient: 1 loop per 3 cuts (was 4)
        # We WANT loops to be used, just not spammed
        max_loops = max(2, n_cut // 3)  # At least 2 loops allowed, then 1 per 3 cuts
        if n_loops > max_loops:
            excess_loops = n_loops - max_loops
            # -3 points per excess loop (was -5), capped at -20 (was -30)
            penalties["excess_loops"] = -min(excess_loops * 3.0, 20.0)
        else:
            penalties["excess_loops"] = 0.0
        
        # Penalty 5: EXCESSIVE JUMPS - Jump actions should be rare, not main strategy
        # Count only jump-back actions specifically
        jump_actions = {ActionTypeV2.JUMP_BACK_4, ActionTypeV2.JUMP_BACK_8}
        n_jumps = sum(1 for a in self.episode_actions if a.action_type in jump_actions)
        n_actions = len(self.episode_actions)
        jump_ratio = n_jumps / n_actions if n_actions > 0 else 0.0
        
        # Jumps should be < 25% of actions (more lenient - we WANT some jumps)
        # Only penalize if truly spamming (> 30%)
        if jump_ratio > 0.30:
            # Spam territory - moderate penalty
            excess_jump_ratio = jump_ratio - 0.30  # 0 to 0.70
            penalties["excess_jumps"] = -(excess_jump_ratio / 0.30) ** 2 * 20.0  # up to -20 (was -35)
        elif jump_ratio > 0.25:
            # Slightly over budget
            excess_jump_ratio = jump_ratio - 0.25  # 0 to 0.05
            penalties["excess_jumps"] = -(excess_jump_ratio / 0.05) * 5.0  # up to -5 (was -10)
        else:
            penalties["excess_jumps"] = 0.0
        
        # Penalty 6: LOW ACTION DIVERSITY - Must use more than 3 action types
        from collections import Counter
        action_types_used = Counter(a.action_type for a in self.episode_actions if hasattr(a, 'action_type'))
        n_unique_actions = len(action_types_used)
        
        # More lenient diversity requirements - we want SOME diversity, but
        # it is OK to have a dominant cutting strategy as long as other tools are used
        if n_unique_actions <= 2:
            penalties["low_diversity"] = -15.0  # Only 1-2 action types = bad (was -25)
        elif n_unique_actions == 3:
            penalties["low_diversity"] = -8.0   # 3 types = could be better (was -15)
        elif n_unique_actions == 4:
            penalties["low_diversity"] = -2.0   # 4 types = acceptable (was -5)
        else:
            penalties["low_diversity"] = 0.0    # 5+ types = good
        
        # Penalty 8: TRANSITION MARKER SPAM - Markers should be rare (< 5% of actions)
        marker_actions = {ActionTypeV2.MARK_SOFT_TRANSITION, ActionTypeV2.MARK_HARD_CUT}
        n_markers = sum(1 for a in self.episode_actions if a.action_type in marker_actions)
        marker_ratio = n_markers / n_actions if n_actions > 0 else 0.0
        if marker_ratio > 0.15:
            penalties["marker_spam"] = -20.0 - (marker_ratio - 0.15) * 50.0  # Heavy penalty
        elif marker_ratio > 0.05:
            penalties["marker_spam"] = -(marker_ratio - 0.05) * 100.0  # up to -10
        else:
            penalties["marker_spam"] = 0.0
        
        # Penalty 9: UNEVEN CUT DISTRIBUTION - Cuts should be spread throughout, not bunched
        # Prevents "cut everything at the end" exploit
        if n_cut > 5:
            cut_positions = sorted(self.edit_history.cut_beats)
            if len(cut_positions) > 1:
                # Check if cuts are spread evenly (std dev of gaps)
                gaps = [cut_positions[i+1] - cut_positions[i] for i in range(len(cut_positions)-1)]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                # Coefficient of variation - high = uneven distribution
                if avg_gap > 0:
                    gap_std = (sum((g - avg_gap)**2 for g in gaps) / len(gaps)) ** 0.5
                    cv = gap_std / avg_gap
                    # CV > 1.5 means very uneven (bunched cuts)
                    if cv > 2.0:
                        penalties["bunched_cuts"] = -15.0
                    elif cv > 1.5:
                        penalties["bunched_cuts"] = -8.0
                    elif cv > 1.0:
                        penalties["bunched_cuts"] = -3.0
                    else:
                        penalties["bunched_cuts"] = 0.0
                else:
                    penalties["bunched_cuts"] = 0.0
            else:
                penalties["bunched_cuts"] = 0.0
        else:
            penalties["bunched_cuts"] = 0.0
        
        # Penalty 10: LOW COVERAGE - Must process most of the song, not just a small portion
        # Prevents "do minimum then stop" exploit
        max_beat_processed = max(self.edit_history.kept_beats | self.edit_history.cut_beats) if (self.edit_history.kept_beats or self.edit_history.cut_beats) else 0
        coverage_ratio = max_beat_processed / n_beats if n_beats > 0 else 0.0
        if coverage_ratio < 0.50:
            penalties["low_coverage"] = -30.0 * (1.0 - coverage_ratio / 0.50)  # up to -30
        elif coverage_ratio < 0.80:
            penalties["low_coverage"] = -10.0 * (1.0 - coverage_ratio / 0.80)  # up to -10
        else:
            penalties["low_coverage"] = 0.0
        
        # Penalty 11: DOMINANT ACTION - No single action should be > 50% of all actions
        if action_types_used:
            most_common_action, most_common_count = action_types_used.most_common(1)[0]
            dominant_ratio = most_common_count / n_actions if n_actions > 0 else 0.0
            if dominant_ratio > 0.60:
                penalties["dominant_action"] = -25.0 * ((dominant_ratio - 0.60) / 0.40)  # up to -25
            elif dominant_ratio > 0.50:
                penalties["dominant_action"] = -10.0 * ((dominant_ratio - 0.50) / 0.10)  # up to -10
            else:
                penalties["dominant_action"] = 0.0
        else:
            penalties["dominant_action"] = 0.0
        
        # Penalty 12: SECTION ACTION IMBALANCE - Section actions vs beat actions should be balanced
        # Prevents gaming by using KEEP_PHRASE for large chunks while CUT_BEAT for small adjustments
        section_keep_actions = {ActionTypeV2.KEEP_BAR, ActionTypeV2.KEEP_PHRASE}
        section_cut_actions = {ActionTypeV2.CUT_BAR, ActionTypeV2.CUT_PHRASE}
        beat_keep_actions = {ActionTypeV2.KEEP_BEAT}
        beat_cut_actions = {ActionTypeV2.CUT_BEAT}
        
        n_section_keeps = sum(1 for a in self.episode_actions if a.action_type in section_keep_actions)
        n_section_cuts = sum(1 for a in self.episode_actions if a.action_type in section_cut_actions)
        n_beat_keeps = sum(1 for a in self.episode_actions if a.action_type in beat_keep_actions)
        n_beat_cuts = sum(1 for a in self.episode_actions if a.action_type in beat_cut_actions)
        
        # Ratio of section_keeps to section_cuts should be similar to beat_keeps to beat_cuts
        # Prevents: KEEP_PHRASE everything, CUT_BEAT selectively
        total_keeps = n_section_keeps + n_beat_keeps
        total_cuts = n_section_cuts + n_beat_cuts
        
        if total_keeps > 0 and total_cuts > 0:
            section_keep_ratio = n_section_keeps / total_keeps if total_keeps > 0 else 0.0
            section_cut_ratio = n_section_cuts / total_cuts if total_cuts > 0 else 0.0
            ratio_diff = abs(section_keep_ratio - section_cut_ratio)
            # If difference > 0.5, model is gaming (e.g., section keeps but beat cuts)
            if ratio_diff > 0.6:
                penalties["section_imbalance"] = -20.0
            elif ratio_diff > 0.4:
                penalties["section_imbalance"] = -10.0
            else:
                penalties["section_imbalance"] = 0.0
        else:
            penalties["section_imbalance"] = 0.0
        
        # Penalty 13: CUT POSITION GAMING - Cuts shouldn't cluster only at "easy" positions
        # Prevents: Only cutting at beat%8==0 to game phrase alignment score
        if n_cut > 10:
            cut_positions = sorted(self.edit_history.cut_beats)
            # Check what percentage of cuts are at beat%4==0 or beat%8==0
            easy_cuts = sum(1 for b in cut_positions if b % 4 == 0)
            easy_ratio = easy_cuts / len(cut_positions) if cut_positions else 0.0
            # In natural music, ~25% of beats are at %4==0
            # If > 60% of cuts are at easy positions, model is gaming
            if easy_ratio > 0.70:
                penalties["easy_cut_gaming"] = -15.0
            elif easy_ratio > 0.50:
                penalties["easy_cut_gaming"] = -5.0
            else:
                penalties["easy_cut_gaming"] = 0.0
        else:
            penalties["easy_cut_gaming"] = 0.0
        
        # Penalty 14: EFFECTIVE KEEP RATIO - Use effective keeps (including loops) for ratio calc
        # Prevents: Using loops to keep content without it counting as "keep"
        effective_keep_ratio = n_effective_kept / (n_effective_kept + n_cut) if (n_effective_kept + n_cut) > 0 else 0.0
        if effective_keep_ratio > 0.80:
            # Too much kept (including via loops)
            excess = effective_keep_ratio - 0.80
            penalties["effective_keep_excess"] = -(excess / 0.20) ** 2 * 30.0  # up to -30
        else:
            penalties["effective_keep_excess"] = 0.0
        
        # Penalty 15: MINIMUM CUT BEATS - Must cut at least 20% of total beats (not just actions)
        # Prevents: Processing whole song with mostly keeps
        actual_cut_ratio = n_cut / n_beats if n_beats > 0 else 0.0
        if actual_cut_ratio < 0.15:
            deficit = 0.15 - actual_cut_ratio
            penalties["min_cut_beats"] = -(deficit / 0.15) ** 2 * 40.0  # up to -40
        elif actual_cut_ratio < 0.20:
            deficit = 0.20 - actual_cut_ratio
            penalties["min_cut_beats"] = -(deficit / 0.05) * 10.0  # up to -10
        else:
            penalties["min_cut_beats"] = 0.0
        
        # Penalty 7: NO KEEP ACTIONS - Must actively decide to keep content, not just cut/jump
        keep_actions = {ActionTypeV2.KEEP_BEAT, ActionTypeV2.KEEP_BAR, ActionTypeV2.KEEP_PHRASE}
        n_keep_actions = sum(1 for a in self.episode_actions if a.action_type in keep_actions)
        keep_action_ratio = n_keep_actions / n_actions if n_actions > 0 else 0.0
        
        # Should have at least 20% keep actions (explicit decisions to keep good content)
        if keep_action_ratio < 0.05:
            penalties["no_keep_actions"] = -20.0  # Almost no keeps = bad
        elif keep_action_ratio < 0.15:
            penalties["no_keep_actions"] = -10.0  # Too few keeps
        elif keep_action_ratio < 0.25:
            penalties["no_keep_actions"] = -5.0   # Barely enough keeps
        else:
            penalties["no_keep_actions"] = 0.0    # Good balance
        
        # Penalty 4: OUTPUT DURATION - HARD CONSTRAINT ON >90%
        # The model MUST learn that >90% duration is completely unacceptable
        # CRITICAL FIX: Duration = kept + reordered + looped_extra + jumped_beats
        
        # Count reordered beats (they go to end of output, so they count!)
        total_reordered_beats = sum(n for _, n in self.edit_history.reordered_sections) if self.edit_history.reordered_sections else 0
        
        # Count jumped beats (JUMP_BACK replays content = adds to duration!)
        # Each jump from beat A to beat B replays (A - B) beats
        total_jumped_beats = sum(max(0, from_b - to_b) for from_b, to_b in self.edit_history.jump_points) if self.edit_history.jump_points else 0
        
        # Count loop extra beats from BOTH looped_beats (LOOP_BEAT) AND looped_sections (LOOP_BAR/PHRASE)
        # looped_beats: {beat_idx: n_times} - beat is kept, adds (n_times - 1) extra copies
        loop_extra_beats = sum(max(0, times - 1) for times in self.edit_history.looped_beats.values()) if self.edit_history.looped_beats else 0
        # looped_sections: [(start, end, n_times)] - section is kept, adds (n_times - 1) * section_length extra
        loop_extra_sections = sum((end - start) * max(0, times - 1) for start, end, times in self.edit_history.looped_sections) if self.edit_history.looped_sections else 0
        loop_extra = loop_extra_beats + loop_extra_sections
        
        # Output = kept beats + reordered beats + extra looped beats + jumped beats
        estimated_output_beats = n_kept + total_reordered_beats + loop_extra + total_jumped_beats
        duration_ratio = estimated_output_beats / n_beats if n_beats > 0 else 1.0
        self.episode_reward_breakdown["duration_ratio"] = duration_ratio
        self.episode_reward_breakdown["n_reordered"] = total_reordered_beats
        self.episode_reward_breakdown["n_jumped"] = total_jumped_beats
        
        # HARD CONSTRAINT: >90% duration = episode failure
        # This overrides ALL other rewards - model cannot ignore this
        if duration_ratio > 0.90:
            excess = duration_ratio - 0.90  # 0 to 0.10+
            # MASSIVE penalty: -150 base, scaling up to -300 for 100% duration
            # This ensures ANY positive rewards are wiped out
            penalties["excess_duration"] = -150.0 - (excess / 0.10) * 150.0
            self.episode_reward_breakdown["duration_violation"] = True
            logger.debug(f"DURATION VIOLATION: {duration_ratio:.1%} > 90% | penalty: {penalties['excess_duration']:.1f}")
        elif duration_ratio > 0.75:
            # Strong warning zone: 75-90% - significant penalty
            excess = duration_ratio - 0.75  # 0 to 0.15
            penalties["excess_duration"] = -(excess / 0.15) ** 1.5 * 80.0  # up to -80
        elif duration_ratio > 0.60:
            # Soft penalty zone: 60-75% - moderate penalty
            excess = duration_ratio - 0.60  # 0 to 0.15
            penalties["excess_duration"] = -(excess / 0.15) ** 1.5 * 30.0  # up to -30
        elif duration_ratio > 0.40:
            # Sweet spot: 40-60% - bonus!
            closeness = 1.0 - abs(duration_ratio - 0.50) / 0.10
            penalties["excess_duration"] = max(0, closeness * 15.0)  # up to +15
        elif duration_ratio > 0.25:
            # Acceptable: 25-40% - neutral
            penalties["excess_duration"] = 0.0
        else:
            # Too aggressive: <25% - penalty for cutting too much
            shortage = 0.25 - duration_ratio  # 0 to 0.25
            penalties["excess_duration"] = -(shortage / 0.25) ** 2 * 40.0  # up to -40
        
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
        
        # Component 4: Cut Quality (20 points max)
        # Reward for cutting LOW quality beats, keeping HIGH quality
        cut_quality = self._compute_cut_quality()
        components["cut_quality"] = np.sqrt(cut_quality) * 20.0
        
        # Component 5: Edit Structure (15 points max)
        # Reward for having multiple kept sections (not one big chunk or scattered)
        structure_score = self._compute_edit_structure_score()
        components["edit_structure"] = structure_score * 15.0
        
        # Component 6: Phrase Alignment (15 points max)
        phrase_score = self._compute_phrase_alignment()
        components["phrase_alignment"] = phrase_score * 15.0
        
        # Component 7: Action Diversity Bonus (5 points max)
        # Penalize using only one type of action (prevents section-only exploit)
        diversity_score = self._compute_action_diversity()
        components["action_diversity"] = diversity_score * 5.0
        
        # Component 8: Creative Reordering Bonus (20 points max)
        # REWARD for using jumps/loops that IMPROVE the edit quality
        reorder_score = self._compute_reordering_quality()
        components["reordering_quality"] = reorder_score * 20.0
        
        # Component 9: Loop Usage Bonus (10 points max)
        # POSITIVE BONUS for using loops - they're a valid creative tool!
        # Currently model avoids loops because they trigger duration penalty
        # But good loops (repeating high-energy sections) should be rewarded
        loop_bonus = self._compute_loop_usage_bonus()
        components["loop_bonus"] = loop_bonus * 10.0
        
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
        max_entropy = np.log(16)  # 16 possible actions
        entropy_score = entropy / max_entropy
        
        # Method 3: No single action dominates (>70%)
        max_action_ratio = max(action_counts.values()) / n_total
        dominance_score = 1.0 if max_action_ratio < 0.5 else (1.0 - max_action_ratio)
        
        return 0.3 * uniqueness_score + 0.4 * entropy_score + 0.3 * dominance_score
    
    def _compute_reordering_quality(self) -> float:
        """Compute quality of reordering actions (jumps, loops, reorders).
        
        REWARDS creative use of non-linear editing:
        1. Jumps that create better energy flow (jump to similar energy level)
        2. Loops that repeat high-quality sections
        3. Reorders that move good content to better positions
        4. Overall structural coherence of reordered output
        """
        if not self.episode_actions:
            return 0.0
        
        features = self.audio_state.beat_features
        if features is None or len(features) == 0:
            return 0.3  # Neutral if no features
        
        n_beats = len(features)
        
        # Collect jump, loop, and reorder actions
        jump_actions = []
        loop_actions = []
        reorder_actions = []
        
        for action in self.episode_actions:
            if action.action_type == ActionTypeV2.JUMP_BACK_4:
                jump_actions.append((action.beat_index, 4))
            elif action.action_type == ActionTypeV2.JUMP_BACK_8:
                jump_actions.append((action.beat_index, 8))
            elif action.action_type == ActionTypeV2.LOOP_BEAT:
                loop_actions.append((action.beat_index, 2))
            elif action.action_type == ActionTypeV2.LOOP_BAR:
                loop_actions.append((action.beat_index, 2))  # Bar = 4 beats looped 2x
            elif action.action_type == ActionTypeV2.LOOP_PHRASE:
                loop_actions.append((action.beat_index, 2))  # Phrase = 8 beats looped 2x
            elif action.action_type == ActionTypeV2.REORDER_BEAT:
                reorder_actions.append((action.beat_index, 1))
            elif action.action_type == ActionTypeV2.REORDER_BAR:
                reorder_actions.append((action.beat_index, action.n_beats_affected))
            elif action.action_type == ActionTypeV2.REORDER_PHRASE:
                reorder_actions.append((action.beat_index, action.n_beats_affected))
        
        scores = []
        
        # === JUMP QUALITY ===
        for from_beat, jump_amount in jump_actions:
            target_beat = max(0, from_beat - jump_amount)
            if 0 <= from_beat < n_beats and 0 <= target_beat < n_beats:
                source_feat = features[from_beat]
                target_feat = features[target_beat]
                source_energy = np.mean(np.abs(source_feat))
                target_energy = np.mean(np.abs(target_feat))
                energy_diff = abs(source_energy - target_energy)
                energy_similarity = np.exp(-energy_diff * 2.0)
                dot = np.dot(source_feat, target_feat)
                norm = np.linalg.norm(source_feat) * np.linalg.norm(target_feat) + 1e-8
                feature_similarity = (dot / norm + 1.0) / 2.0
                target_quality = min(1.0, target_energy / (np.mean(np.abs(features)) + 1e-8))
                jump_score = 0.4 * energy_similarity + 0.3 * feature_similarity + 0.3 * target_quality
                scores.append(jump_score)
        
        # === LOOP QUALITY ===
        for beat_idx, n_times in loop_actions:
            if 0 <= beat_idx < n_beats:
                beat_feat = features[beat_idx]
                beat_energy = np.mean(np.abs(beat_feat))
                avg_energy = np.mean(np.abs(features))
                energy_ratio = beat_energy / (avg_energy + 1e-8)
                loop_score = min(1.0, energy_ratio)
                if beat_idx > 0 and beat_idx < n_beats - 1:
                    neighbor_avg = (features[beat_idx - 1] + features[beat_idx + 1]) / 2
                    distinctiveness = np.linalg.norm(beat_feat - neighbor_avg)
                    distinctiveness_score = min(1.0, distinctiveness * 0.5)
                    loop_score = 0.6 * loop_score + 0.4 * distinctiveness_score
                scores.append(loop_score)
        
        # === REORDER QUALITY ===
        # Reward reordering of high-energy, distinctive sections (good content moved to end)
        for start_beat, n_beats_affected in reorder_actions:
            if 0 <= start_beat < n_beats:
                # Get average features for the reordered section
                end_beat = min(start_beat + n_beats_affected, n_beats)
                section_feats = features[start_beat:end_beat]
                if len(section_feats) > 0:
                    section_energy = np.mean(np.abs(section_feats))
                    avg_energy = np.mean(np.abs(features))
                    
                    # Higher score for reordering high-energy content (moving good stuff)
                    energy_quality = min(1.0, section_energy / (avg_energy + 1e-8))
                    
                    # Bonus if the section is coherent (low internal variance)
                    if len(section_feats) > 1:
                        internal_var = np.mean(np.var(section_feats, axis=0))
                        coherence_score = np.exp(-internal_var * 0.5)
                    else:
                        coherence_score = 0.5
                    
                    # Bonus for reordering at phrase boundaries (musically sensible)
                    boundary_bonus = 0.0
                    if start_beat % 8 == 0:  # Phrase boundary
                        boundary_bonus = 0.3
                    elif start_beat % 4 == 0:  # Bar boundary
                        boundary_bonus = 0.15
                    
                    reorder_score = 0.4 * energy_quality + 0.3 * coherence_score + 0.3 * (0.5 + boundary_bonus)
                    scores.append(reorder_score)
        
        # === BASE SCORE FOR ATTEMPTING REORDERING ===
        n_reorder_actions = len(jump_actions) + len(loop_actions) + len(reorder_actions)
        n_total_actions = len(self.episode_actions)
        
        if n_total_actions == 0:
            return 0.0
        
        reorder_ratio = n_reorder_actions / n_total_actions
        
        # Bonus for trying reordering at all (even if imperfect)
        if reorder_ratio == 0:
            attempt_bonus = 0.0
        elif reorder_ratio < 0.05:
            attempt_bonus = 0.3  # Small usage
        elif reorder_ratio < 0.20:
            attempt_bonus = 0.6  # Good usage range
        elif reorder_ratio < 0.30:
            attempt_bonus = 0.4  # Getting heavy
        else:
            attempt_bonus = 0.2  # Overusing
        
        if scores:
            quality_avg = np.mean(scores)
            return 0.5 * attempt_bonus + 0.5 * quality_avg
        else:
            return attempt_bonus * 0.5
    
    def _compute_loop_usage_bonus(self) -> float:
        """POSITIVE bonus for using loop actions creatively.
        
        Loops are a valid creative tool for music editing:
        1. Repeating a catchy hook or drop builds tension
        2. Looping the best part of a section extends high-energy moments
        3. Strategic loops can improve song structure
        
        Problem: Model currently avoids loops because they trigger duration penalty.
        Solution: Add explicit positive bonus for good loop usage.
        """
        if not self.edit_history.looped_beats:
            return 0.0  # No loops = no bonus (but also no penalty)
        
        features = self.audio_state.beat_features
        if features is None or len(features) == 0:
            return 0.2
        
        n_beats = len(features)
        avg_energy = np.mean(np.abs(features))
        
        loop_scores = []
        for beat_idx, loop_times in self.edit_history.looped_beats.items():
            if 0 <= beat_idx < n_beats:
                # Get the looped beat's energy
                beat_energy = np.mean(np.abs(features[beat_idx]))
                
                # Bonus for looping HIGH-ENERGY content (the good parts!)
                energy_ratio = beat_energy / (avg_energy + 1e-8)
                energy_quality = min(1.0, max(0.0, (energy_ratio - 0.7) / 0.6))  # Scale 0.7-1.3 to 0-1
                
                # Bonus for looping at musically sensible positions
                position_bonus = 0.0
                if beat_idx % 8 == 0:  # Phrase start - great for loops
                    position_bonus = 0.3
                elif beat_idx % 4 == 0:  # Bar start - good
                    position_bonus = 0.15
                
                # Don't over-loop the same spot
                repetition_penalty = max(0.0, 1.0 - (loop_times - 2) * 0.3)
                
                loop_score = (0.5 * energy_quality + 0.3 * position_bonus + 0.2) * repetition_penalty
                loop_scores.append(loop_score)
        
        if not loop_scores:
            return 0.0
        
        # Base bonus for using loops at all
        n_loops = len(loop_scores)
        n_total = len(self.episode_actions) if self.episode_actions else 1
        loop_ratio = n_loops / n_total
        
        # Sweet spot: 5-15% of actions being loops
        if loop_ratio == 0:
            usage_bonus = 0.0
        elif loop_ratio < 0.05:
            usage_bonus = 0.5  # Using loops, but could use more
        elif loop_ratio <= 0.15:
            usage_bonus = 1.0  # Sweet spot!
        elif loop_ratio <= 0.25:
            usage_bonus = 0.6  # Acceptable
        else:
            usage_bonus = 0.3  # Overusing
        
        quality_avg = np.mean(loop_scores)
        return 0.4 * usage_bonus + 0.6 * quality_avg
    
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
        """Build edited audio from edit history.
        
        Order: kept beats in sequence, then reordered sections appended at end.
        """
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
        
        # Append reordered sections at the end
        for start_beat, n_beats_affected in self.edit_history.reordered_sections:
            for i in range(n_beats_affected):
                beat_idx = start_beat + i
                if beat_idx >= len(beat_times):
                    continue
                
                start = beat_samples[beat_idx]
                end = beat_samples[beat_idx + 1] if beat_idx + 1 < len(beat_samples) else len(audio)
                
                segment = audio[start:end].copy()
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
