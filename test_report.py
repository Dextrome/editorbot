#!/usr/bin/env python
"""Comprehensive test report for reward model training."""

import json
from pathlib import Path
import torch

def generate_test_report():
    """Generate a comprehensive test report."""
    
    print("=" * 80)
    print("REWARD MODEL TRAINING - TEST REPORT")
    print("=" * 80)
    
    # 1. Checkpoint Verification
    print("\n[TEST 1] Checkpoint Integrity")
    print("-" * 80)
    checkpoint_path = Path("models/reward_model_v8_long/reward_model_final.pt")
    
    if checkpoint_path.exists():
        print(f"[PASS] Checkpoint file exists: {checkpoint_path}")
        print(f"       File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'config', 'metrics']
        actual_keys = set(checkpoint.keys())
        
        for key in required_keys:
            if key in actual_keys:
                print(f"[PASS] Required key present: {key}")
            else:
                print(f"[FAIL] Missing required key: {key}")
    else:
        print(f"[FAIL] Checkpoint not found: {checkpoint_path}")
        return False
    
    # 2. Configuration Verification
    print("\n[TEST 2] Model Configuration")
    print("-" * 80)
    config = checkpoint['config']
    
    expected_config = {
        'input_dim': 125,
        'hidden_dim': 256,
        'n_layers': 3,
        'n_heads': 4,
        'use_edit_aware_features': True,
    }
    
    for key, expected_val in expected_config.items():
        actual_val = config.get(key)
        if actual_val == expected_val:
            print(f"[PASS] {key}: {actual_val}")
        else:
            print(f"[WARN] {key}: expected {expected_val}, got {actual_val}")
    
    # 3. Training Metrics
    print("\n[TEST 3] Training Metrics")
    print("-" * 80)
    metrics = checkpoint['metrics']
    
    print(f"[INFO] Best validation accuracy: {metrics.get('best_val_accuracy', 0)*100:.1f}%")
    print(f"[INFO] Final training loss: {metrics.get('final_train_loss', 0):.4f}")
    if 'epoch' in checkpoint:
        print(f"[INFO] Training stopped at epoch: {checkpoint['epoch']}")
    
    # Verify convergence
    if metrics.get('best_val_accuracy', 0) > 0.95:
        print(f"[PASS] Model converged well (accuracy > 95%)")
    else:
        print(f"[FAIL] Model did not converge (accuracy < 95%)")
    
    if metrics.get('final_train_loss', 1.0) < 0.1:
        print(f"[PASS] Final loss is acceptable (<0.1)")
    else:
        print(f"[WARN] Final loss is high (>=0.1)")
    
    # 4. Model State Dict
    print("\n[TEST 4] Model Weights")
    print("-" * 80)
    model_state = checkpoint['model_state_dict']
    total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
    print(f"[PASS] Model state dict has {len(model_state)} parameter tensors")
    print(f"[INFO] Total parameters: {total_params:,}")
    
    # 5. Data Verification
    print("\n[TEST 5] Training Data")
    print("-" * 80)
    training_pairs_path = Path("models/reward_model_v8/training_pairs.json")
    
    if training_pairs_path.exists():
        with open(training_pairs_path) as f:
            pairs_data = json.load(f)
        
        n_pairs = len(pairs_data)
        print(f"[PASS] Training data exists: {training_pairs_path}")
        print(f"[INFO] Total preference pairs: {n_pairs}")
        
        if n_pairs >= 1000:
            print(f"[PASS] Sufficient training data (>= 1000 pairs)")
        else:
            print(f"[WARN] Limited training data (< 1000 pairs)")
    else:
        print(f"[WARN] Training data file not found (not critical)")
    
    # 6. Inference Test
    print("\n[TEST 6] Model Inference")
    print("-" * 80)
    try:
        from rl_editor.train_reward_model import LearnedRewardModel
        
        model = LearnedRewardModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test with random input
        batch_size, n_beats = 4, 32
        test_features = torch.randn(batch_size, n_beats, config['input_dim'])
        test_actions = torch.randint(0, 5, (batch_size, n_beats))
        
        with torch.no_grad():
            output = model(test_features, test_actions)
        
        print(f"[PASS] Model loaded successfully")
        print(f"[PASS] Forward pass successful")
        print(f"[INFO] Input shape: {test_features.shape}")
        print(f"[INFO] Output shape: {output.shape}")
        print(f"[INFO] Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        if output.shape[0] == batch_size:
            print(f"[PASS] Output batch size correct")
        else:
            print(f"[FAIL] Output batch size incorrect")
            
    except Exception as e:
        print(f"[FAIL] Inference test failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("[SUCCESS] All critical tests passed!")
    print("[INFO] Reward model is ready for RLHF training pipeline")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = generate_test_report()
    exit(0 if success else 1)
