"""
Subprocess-based vectorized environment for true multiprocessing.
Each environment runs in its own process, bypassing the GIL.

Key optimization: Auxiliary targets are computed locally in each subprocess,
avoiding expensive IPC overhead for large numpy arrays.
"""
import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import numpy as np
from typing import List, Tuple, Optional, Any, Dict, Callable
import cloudpickle
import logging
import time

logger = logging.getLogger(__name__)


def _worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn_pickled: bytes,
    worker_id: int,
):
    """Worker process that runs a single environment.
    
    Each worker maintains its own auxiliary target computer for efficient
    local computation without IPC overhead.
    """
    parent_remote.close()
    env_fn = cloudpickle.loads(env_fn_pickled)
    env = env_fn()
    
    # Local auxiliary target computer (avoids IPC for large arrays)
    from rl_editor.auxiliary_tasks import AuxiliaryTargetComputer, AuxiliaryConfig
    aux_computer = AuxiliaryTargetComputer(AuxiliaryConfig())
    current_audio_id = None
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                
                # Compute auxiliary targets locally (fast, no IPC for big arrays)
                if env.audio_state is not None and env.audio_state.beat_features is not None:
                    beat_times = env.audio_state.beat_times
                    beat_features = env.audio_state.beat_features
                    beat_idx = env.current_beat
                    
                    # Use cached computation
                    aux_targets = aux_computer.get_targets(
                        audio_id=current_audio_id or f"worker_{worker_id}",
                        beat_times=beat_times,
                        beat_features=beat_features,
                        beat_indices=np.array([beat_idx]),
                    )
                    # Convert to IPC-friendly format
                    aux_dict = {}
                    for k, v in aux_targets.items():
                        if k == "reconstruction_mask":
                            continue  # Skip mask
                        if k == "reconstruction":
                            # Reconstruction is a feature array - convert to list
                            if len(v) > 0:
                                aux_dict[k] = v[0].tolist() if hasattr(v[0], 'tolist') else list(v[0])
                            else:
                                aux_dict[k] = []
                        else:
                            # Scalar values (tempo bin, energy bin, phrase boundary)
                            if len(v) > 0:
                                val = v[0]
                                aux_dict[k] = float(val) if np.isscalar(val) else float(val.item())
                            else:
                                aux_dict[k] = 0.0
                    info["aux_targets"] = aux_dict
                
                remote.send((obs, reward, terminated, truncated, info))
                
            elif cmd == "reset":
                obs, info = env.reset()
                # Clear cache on reset (new track)
                aux_computer.clear_cache()
                remote.send((obs, info))
                
            elif cmd == "get_action_mask":
                mask = env.get_action_mask()
                remote.send(mask)
                
            elif cmd == "set_audio_state":
                env.set_audio_state(data)
                # Update audio ID for caching
                current_audio_id = f"worker_{worker_id}_{id(data)}"
                aux_computer.clear_cache()  # New audio, clear cache
                remote.send(True)
                
            elif cmd == "get_attr":
                attr = getattr(env, data, None)
                remote.send(attr)
                
            elif cmd == "close":
                remote.close()
                break
                
            else:
                raise ValueError(f"Unknown command: {cmd}")
                
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")
        import traceback
        traceback.print_exc()
        try:
            remote.send(("error", str(e)))
        except:
            pass
    finally:
        if hasattr(env, 'close'):
            env.close()


class SubprocessVecEnvV2:
    """
    Vectorized environment using subprocesses for true parallelism.
    
    Each environment runs in its own process, allowing true parallel
    execution that bypasses Python's GIL.
    """
    
    def __init__(
        self,
        env_fns: List[Callable],
        start_method: str = "spawn",
    ):
        """
        Args:
            env_fns: List of callables that create environments
            start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
        """
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        # Set up multiprocessing context
        ctx = mp.get_context(start_method)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        
        # Start worker processes
        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            # Pickle the env function for transfer to subprocess
            env_fn_pickled = cloudpickle.dumps(env_fn)
            
            process = ctx.Process(
                target=_worker,
                args=(work_remote, remote, env_fn_pickled, i),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()  # Close in parent process
        
        logger.info(f"Started {self.n_envs} subprocess environments")
    
    def step_async(self, actions: List[int]):
        """Send step commands to all environments asynchronously."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True
    
    def step_wait(self) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict]]:
        """Wait for all environments to complete their steps."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs_list = [r[0] for r in results]
        rewards = [r[1] for r in results]
        terminateds = [r[2] for r in results]
        truncateds = [r[3] for r in results]
        infos = [r[4] for r in results]
        
        return obs_list, rewards, terminateds, truncateds, infos
    
    def step_all(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict]]:
        """Step all environments synchronously."""
        self.step_async(actions)
        return self.step_wait()
    
    def reset_all(self) -> Tuple[List[np.ndarray], List[dict]]:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(("reset", None))
        
        results = [remote.recv() for remote in self.remotes]
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        
        return obs_list, info_list
    
    def get_action_masks(self) -> List[np.ndarray]:
        """Get action masks from all environments."""
        for remote in self.remotes:
            remote.send(("get_action_mask", None))
        
        masks = [remote.recv() for remote in self.remotes]
        return masks
    
    def reset_env(self, i: int) -> Tuple[np.ndarray, dict]:
        """Reset a single environment."""
        self.remotes[i].send(("reset", None))
        obs, info = self.remotes[i].recv()
        return obs, info
    
    def set_audio_states(self, audio_states: List[Any]):
        """Set audio states for all environments."""
        for remote, state in zip(self.remotes, audio_states):
            remote.send(("set_audio_state", state))
        
        # Wait for confirmation
        for remote in self.remotes:
            remote.recv()
    
    def close(self):
        """Close all environments and processes."""
        if self.closed:
            return
        
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except:
                pass
        
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
        
        self.closed = True
        logger.info("Closed all subprocess environments")
    
    def __del__(self):
        self.close()


def make_subprocess_vec_env(
    config,
    n_envs: int,
    learned_reward_model: Optional[Any] = None,
) -> SubprocessVecEnvV2:
    """
    Factory function to create subprocess vectorized environment.
    
    Args:
        config: Configuration object
        n_envs: Number of parallel environments
        learned_reward_model: Optional learned reward model (currently not used in V2 env)
        
    Returns:
        SubprocessVecEnvV2 instance
    """
    from rl_editor.environment_v2 import AudioEditingEnvV2
    
    # Note: learned_reward_model is not passed to V2 env (reward computed in trainer)
    def make_env():
        return AudioEditingEnvV2(config)
    
    env_fns = [make_env for _ in range(n_envs)]
    
    return SubprocessVecEnvV2(env_fns, start_method="spawn")
