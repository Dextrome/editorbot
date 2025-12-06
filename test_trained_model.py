#!/usr/bin/env python
"""Test the trained reward model."""

import torch
import json
from pathlib import Path
from rl_editor.train_reward_model import LearnedRewardModel

def test_trained_model():
    """Test the trained reward model checkpoint."""
    # Load the trained model
    checkpoint_path = Path("models/reward_model_v8_long/reward_model_final.pt")
    if checkpoint_path.exists():
        print(f"[OK] Checkpoint found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"[OK] Checkpoint loaded successfully")
        
        # Get configuration
        config = checkpoint['config']
        print(f"\nModel Configuration:")
        print(f"  Input dim: {config['input_dim']}")
        print(f"  Hidden dim: {config['hidden_dim']}")
        print(f"  Layers: {config['n_layers']}")
        print(f"  Heads: {config['n_heads']}")
        
        # Create model and load weights
        model = LearnedRewardModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"\n[OK] Model loaded and ready for inference")
        
        # Test inference
        print(f"\nTesting inference:")
        batch_size = 4
        n_beats = 32
        # Model expects: beat_features (batch, n_beats, input_dim)
        test_features = torch.randn(batch_size, n_beats, config['input_dim'])
        # action_ids (batch, n_beats) - dummy actions
        test_actions = torch.randint(0, 5, (batch_size, n_beats))
        
        with torch.no_grad():
            output = model(test_features, test_actions)
        print(f"  Features shape: {test_features.shape}")
        print(f"  Actions shape: {test_actions.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  [OK] Inference successful")
        
        # Check metrics
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"\nTraining Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        
        print(f"\n[SUCCESS] All tests passed!")
        return True
    else:
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        return False

if __name__ == "__main__":
    success = test_trained_model()
    exit(0 if success else 1)
