import numpy as np
from rl_editor.config import get_default_config
from rl_editor.state import AudioState
from rl_editor.environment import AudioEditingEnvFactored
from rl_editor.actions import ActionType, ActionSize, ActionAmount

config = get_default_config()
# Simple beat_times and beat_features
n_beats = 16
beat_times = np.arange(n_beats)
# Use full feature dim (121) to avoid shape issues
beat_features = np.zeros((n_beats, 121), dtype=np.float32)

audio_state = AudioState(
    beat_index=0,
    beat_times=beat_times,
    beat_features=beat_features,
)

env = AudioEditingEnvFactored(config, audio_state=audio_state)
obs, info = env.reset(options={"audio_state": audio_state})
print('reset info:', info)

# Build KEEP action: type KEEP, size BEAT, amount NEUTRAL
action = (ActionType.KEEP.value, ActionSize.BEAT.value, ActionAmount.NEUTRAL.value)

# Step once
obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.int64))
print('step1 reward, info:', reward, info.get('temporal_penalty'), info.get('step_reward'))

# Step second action at immediate next beat (simulate micro-edit)
obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.int64))
print('step2 reward, info:', reward, info.get('temporal_penalty'), info.get('step_reward'))

# Print episode breakdown if ended
if terminated or truncated:
    print('episode_end breakdown:', info.get('reward_breakdown'))
else:
    print('not ended')
