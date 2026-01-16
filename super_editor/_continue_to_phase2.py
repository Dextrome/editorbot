"""
Monitor Phase 1 training and automatically start Phase 2 when complete.
"""
import os
import sys
import time
import re
from pathlib import Path

sys.path.insert(0, '.')

def wait_for_phase1():
    """Wait for Phase 1 training to complete."""
    log_file = Path('models/phase1_v3_training.log')
    checkpoint = Path('models/phase1_v3_label_aware/final.pt')
    
    print("Monitoring Phase 1 training...")
    last_epoch = 0
    
    while True:
        # Check if final checkpoint exists (training complete)
        if checkpoint.exists():
            print("\nPhase 1 training complete! Final checkpoint found.")
            return True
        
        # Check log for progress
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Find latest epoch
                epochs = re.findall(r'Epoch (\d+)/50', content)
                if epochs:
                    current_epoch = int(epochs[-1])
                    if current_epoch > last_epoch:
                        # Get latest val loss
                        val_losses = re.findall(r'Val   - Loss: ([\d.]+)', content)
                        val_loss = val_losses[-1] if val_losses else "?"
                        print(f"Phase 1 progress: Epoch {current_epoch}/50, Val Loss: {val_loss}")
                        last_epoch = current_epoch
                
                # Check for errors
                if 'Traceback' in content and 'Error' in content:
                    lines = content.split('\n')
                    error_start = -1
                    for i, line in enumerate(lines):
                        if 'Traceback' in line:
                            error_start = i
                    if error_start >= 0:
                        print("ERROR detected in Phase 1 training!")
                        print('\n'.join(lines[error_start:]))
                        return False
        
        time.sleep(30)  # Check every 30 seconds

def run_phase2():
    """Run Phase 2 training."""
    print("\n" + "="*60)
    print("Starting Phase 2: Edit Label Prediction")
    print("="*60)
    
    from super_editor.config import Phase2Config
    from super_editor.trainers import Phase2Trainer
    
    config = Phase2Config()
    config.num_workers = 0
    config.total_epochs = 100
    
    trainer = Phase2Trainer(
        config=config,
        recon_model_path='F:/editorbot/models/phase1_v3_label_aware/best.pt',
        data_dir='F:/editorbot/training_data/super_editor_cache',
        save_dir='F:/editorbot/models/phase2_v1',
    )
    
    trainer.train()
    print("Phase 2 training complete!")

if __name__ == '__main__':
    if wait_for_phase1():
        run_phase2()
    else:
        print("Phase 1 failed, not starting Phase 2")
        sys.exit(1)
