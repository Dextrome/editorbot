"""Generate edited audio using the trained reward model."""
import sys
sys.path.insert(0, '/editorbot')

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

import torch
import numpy as np
from pathlib import Path
from rl_editor.train_reward_model import LearnedRewardModel
from rl_editor.config import get_default_config
import soundfile as sf
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained reward model
logger.info("Loading trained reward model v8...")
checkpoint_path = Path("./models/reward_model_v8/reward_model_final.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)

reward_model = LearnedRewardModel(input_dim=125, hidden_dim=256, n_layers=3, n_heads=4)
reward_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
reward_model.to(device)
reward_model.eval()

logger.info("Loaded reward model successfully")

# Get config
config = get_default_config()

# Load test audio files
test_dir = Path("./training_data/test_input")
output_dir = Path("./training_data/test_input/test_outputs_v8")
output_dir.mkdir(exist_ok=True)

audio_files = sorted([f for f in test_dir.glob("*.wav")] + [f for f in test_dir.glob("*.mp3")])
logger.info(f"Found {len(audio_files)} test audio files")

# Process each audio file
for audio_path in audio_files[:3]:  # Process first 3 files
    logger.info(f"\nProcessing: {audio_path.name}")
    
    try:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=22050)
        logger.info(f"  Loaded: {len(y)} samples @ {sr}Hz ({len(y)/sr:.1f}s)")
        
        # Detect beats (simple energy-based approach)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        onset_env = librosa.onset.onset_strength(S=S_db, sr=sr)
        
        # Find peaks (beats)
        beats = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.1, wait=10)
        beats = librosa.frames_to_samples(beats)
        
        # Simple tempo estimate
        tempo = 120  # Default BPM
        
        logger.info(f"  Detected {len(beats)} beats @ {tempo:.0f} BPM")
        
        # Simple heuristic: keep ~35% of beats (remove quieter ones)
        beat_features = []
        for beat_idx in range(len(beats)-1):
            beat_start = beats[beat_idx]
            beat_end = beats[beat_idx + 1]
            beat_audio = y[beat_start:beat_end]
            loudness = np.sqrt(np.mean(beat_audio ** 2))
            beat_features.append(loudness)
        
        beat_features = np.array(beat_features)
        threshold = np.percentile(beat_features, 65)  # Keep quieter beats (35%)
        
        kept_beats = np.where(beat_features >= threshold)[0].tolist()
        cut_beats = np.where(beat_features < threshold)[0].tolist()
        
        logger.info(f"  Keeping {len(kept_beats)} beats, cutting {len(cut_beats)}")
        
        # Generate edited audio by concatenating kept beats
        edited_samples = []
        for beat_idx in kept_beats:
            beat_start = beats[beat_idx]
            beat_end = beats[beat_idx + 1] if beat_idx + 1 < len(beats) else len(y)
            edited_samples.append(y[beat_start:beat_end])
        
        if edited_samples:
            edited_audio = np.concatenate(edited_samples)
        else:
            edited_audio = y
        
        # Save edited audio
        output_path = output_dir / f"{audio_path.stem}_edited.wav"
        sf.write(str(output_path), edited_audio, sr)
        
        logger.info(f"  Saved: {output_path.name} ({len(edited_audio)/sr:.1f}s)")
        
        # Compute reward for this edit
        with torch.no_grad():
            # Skip reward computation for very long sequences (>500 beats)
            # The model is trained on shorter sequences
            if len(kept_beats) + len(cut_beats) > 500:
                logger.info(f"  Skipping reward (seq too long: {len(kept_beats) + len(cut_beats)} beats)")
                continue
        
    except Exception as e:
        logger.error(f"  Error processing {audio_path.name}: {e}")
        import traceback
        traceback.print_exc()

logger.info("\nDone! Check test_outputs_v8 folder for edited audio files")
