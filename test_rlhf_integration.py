#!/usr/bin/env python
"""Quick test to verify RLHF integration works end-to-end.

Tests:
1. Load learned reward model
2. Create dummy trajectory features
3. Compute learned rewards
4. Combine with dense rewards
5. Verify integration with trainer
"""

import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import logging

from rl_editor.config import Config, get_default_config
from rl_editor.learned_reward_integration import (
    LearnedRewardIntegration,
    LearnedRewardConfig
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_learned_reward_integration():
    """Test the learned reward integration."""
    
    logger.info("=" * 80)
    logger.info("RLHF INTEGRATION TEST")
    logger.info("=" * 80)
    
    # Test 1: Load configuration
    logger.info("\n[TEST 1] Loading configuration...")
    try:
        config = get_default_config()
        logger.info("✓ Configuration loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return False
    
    # Test 2: Initialize reward integration
    logger.info("\n[TEST 2] Initializing learned reward integration...")
    try:
        reward_config = LearnedRewardConfig(
            checkpoint_path="models/reward_model_v8_long/reward_model_final.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_learned_reward=True,
            learned_reward_weight=0.8,
            dense_reward_weight=0.2
        )
        
        reward_integration = LearnedRewardIntegration(config, reward_config)
        logger.info("✓ Reward integration initialized")
        logger.info(f"  Device: {reward_integration.device}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        return False
    
    # Test 3: Load reward model
    logger.info("\n[TEST 3] Loading learned reward model...")
    try:
        success = reward_integration.load_model()
        if not success:
            logger.error("✗ Failed to load model (file not found)")
            return False
        logger.info("✓ Reward model loaded successfully")
        logger.info(f"  Model config: {reward_integration.model_config}")
    except Exception as e:
        logger.error(f"✗ Exception during load: {e}")
        return False
    
    # Test 4: Create dummy trajectory
    logger.info("\n[TEST 4] Creating dummy trajectory...")
    try:
        n_beats = 32
        feature_dim = 125
        beat_features = np.random.randn(n_beats, feature_dim).astype(np.float32)
        
        logger.info("✓ Dummy trajectory created")
        logger.info(f"  Beat features shape: {beat_features.shape}")
    except Exception as e:
        logger.error(f"✗ Failed to create trajectory: {e}")
        return False
    
    # Test 5: Compute learned reward
    logger.info("\n[TEST 5] Computing learned reward...")
    try:
        learned_reward = reward_integration.compute_learned_reward(
            beat_features=beat_features,
            action_mask=None
        )
        logger.info("✓ Learned reward computed")
        logger.info(f"  Learned reward: {learned_reward:.4f}")
        logger.info(f"  Reward range: [{reward_config.clamp_reward[0]}, {reward_config.clamp_reward[1]}]")
    except Exception as e:
        logger.error(f"✗ Failed to compute reward: {e}")
        return False
    
    # Test 6: Combine with dense reward
    logger.info("\n[TEST 6] Combining learned + dense rewards...")
    try:
        dense_reward = 0.5  # Dummy dense reward
        trajectory = {
            "beat_features": beat_features,
            "action_mask": None
        }
        
        combined_reward = reward_integration.compute_trajectory_reward(
            trajectory=trajectory,
            dense_reward=dense_reward
        )
        
        logger.info("✓ Rewards combined successfully")
        logger.info(f"  Dense reward:    {dense_reward:.4f}")
        logger.info(f"  Learned reward:  {learned_reward:.4f}")
        logger.info(f"  Combined reward: {combined_reward:.4f}")
        logger.info(f"  Weights: learned={reward_config.learned_reward_weight}, dense={reward_config.dense_reward_weight}")
    except Exception as e:
        logger.error(f"✗ Failed to combine rewards: {e}")
        return False
    
    # Test 7: Batch processing
    logger.info("\n[TEST 7] Testing batch processing...")
    try:
        batch_rewards = []
        for i in range(5):
            beat_features = np.random.randn(32, 125).astype(np.float32)
            reward = reward_integration.compute_learned_reward(beat_features)
            batch_rewards.append(reward)
        
        logger.info("✓ Batch processing successful")
        logger.info(f"  Batch size: 5")
        logger.info(f"  Mean reward: {np.mean(batch_rewards):.4f}")
        logger.info(f"  Std reward:  {np.std(batch_rewards):.4f}")
        logger.info(f"  Min/Max:     {np.min(batch_rewards):.4f} / {np.max(batch_rewards):.4f}")
    except Exception as e:
        logger.error(f"✗ Batch processing failed: {e}")
        return False
    
    # Test 8: Checkpoint serialization
    logger.info("\n[TEST 8] Testing checkpoint serialization...")
    try:
        state = reward_integration.get_model_state()
        if state is None:
            logger.error("✗ Failed to get model state")
            return False
        
        logger.info("✓ Model state serialized")
        logger.info(f"  State keys: {list(state.keys())}")
        logger.info(f"  Config: {state['config']}")
    except Exception as e:
        logger.error(f"✗ Serialization failed: {e}")
        return False
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    logger.info("\nIntegration Status:")
    logger.info(f"  ✓ Learned reward model loaded")
    logger.info(f"  ✓ Reward computation working")
    logger.info(f"  ✓ Reward combining working")
    logger.info(f"  ✓ Batch processing working")
    logger.info(f"  ✓ Serialization working")
    logger.info("\n✓ Ready for RLHF training!")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_learned_reward_integration()
    sys.exit(0 if success else 1)
