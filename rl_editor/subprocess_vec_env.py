"""
Subprocess-based vectorized environment for true multiprocessing.
Each environment runs in its own process, bypassing the GIL.

Supports factored action space: actions are tuples (type, size, amount).
Auxiliary targets are computed locally in each subprocess to avoid IPC overhead.
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
            
            if cmd == "batched_rollout":
                # Run N steps with pre-computed actions (batched to reduce IPC)
                # actions is list of (type, size, amount) tuples
                actions, n_steps = data
                results = []
                for step_idx in range(n_steps):
                    action = actions[step_idx] if step_idx < len(actions) else (0, 0, 0)
                    # Convert tuple to array for factored env
                    action_array = np.array(action, dtype=np.int64)
                    obs, reward, terminated, truncated, info = env.step(action_array)
                    
                    # Compute auxiliary targets locally
                    aux_dict = {}
                    if env.audio_state is not None and env.audio_state.beat_features is not None:
                        beat_times = env.audio_state.beat_times
                        beat_features = env.audio_state.beat_features
                        beat_idx = env.current_beat
                        aux_targets = aux_computer.get_targets(
                            audio_id=current_audio_id or f"worker_{worker_id}",
                            beat_times=beat_times,
                            beat_features=beat_features,
                            beat_indices=np.array([beat_idx]),
                        )
                        for k, v in aux_targets.items():
                            if k == "reconstruction_mask":
                                continue
                            if k == "reconstruction":
                                if len(v) > 0:
                                    aux_dict[k] = v[0].tolist() if hasattr(v[0], 'tolist') else list(v[0])
                                else:
                                    aux_dict[k] = []
                            else:
                                if len(v) > 0:
                                    val = v[0]
                                    aux_dict[k] = float(val) if np.isscalar(val) else float(val.item())
                                else:
                                    aux_dict[k] = 0.0
                    info["aux_targets"] = aux_dict
                    
                    # Get factored action masks for next step
                    type_mask, size_mask, amount_mask = env.get_action_masks()
                    
                    results.append((
                        obs.tolist(), 
                        reward, 
                        terminated, 
                        truncated, 
                        info, 
                        type_mask.tolist(),
                        size_mask.tolist(),
                        amount_mask.tolist(),
                    ))
                    
                    if terminated or truncated:
                        # Reset and continue
                        obs, reset_info = env.reset()
                        aux_computer.clear_cache()
                
                remote.send(results)
                
            elif cmd == "step":
                # data is (type, size, amount) tuple
                action_array = np.array(data, dtype=np.int64)
                obs, reward, terminated, truncated, info = env.step(action_array)
                
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
                
            elif cmd == "get_action_masks":
                # Return factored masks
                type_mask, size_mask, amount_mask = env.get_action_masks()
                remote.send((type_mask, size_mask, amount_mask))
                
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


class SubprocessVecEnv:
    """
    Vectorized environment using subprocesses for true parallelism.
    
    Each environment runs in its own process, allowing true parallel
    execution that bypasses Python's GIL.
    
    Supports factored action space with (type, size, amount) tuples.
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
        
        logger.info(f"Started {self.n_envs} subprocess environments (factored actions)")
    
    def step_async(self, actions: List[Tuple[int, int, int]]):
        """Send step commands to all environments asynchronously.
        
        Args:
            actions: List of (type, size, amount) tuples
        """
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
    
    def step_all(self, actions: List[Tuple[int, int, int]]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict]]:
        """Step all environments synchronously.
        
        Args:
            actions: List of (type, size, amount) tuples
        """
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
    
    def get_action_masks(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Get factored action masks from all environments.
        
        Returns:
            Tuple of (type_masks, size_masks, amount_masks)
        """
        for remote in self.remotes:
            remote.send(("get_action_masks", None))
        
        results = [remote.recv() for remote in self.remotes]
        type_masks = [r[0] for r in results]
        size_masks = [r[1] for r in results]
        amount_masks = [r[2] for r in results]
        
        return type_masks, size_masks, amount_masks
    
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
    
    def batched_rollout(self, actions_per_env: List[List[Tuple[int, int, int]]], n_steps: int):
        """Run batched rollouts in all environments in parallel.
        
        Each subprocess runs n_steps with pre-computed actions, reducing IPC from
        O(n_envs * n_steps) to O(n_envs).
        
        Args:
            actions_per_env: List of action lists, one per environment.
                            Each action is (type, size, amount) tuple.
            n_steps: Number of steps to run
        """
        # Send batched rollout commands to all workers (parallel)
        for remote, actions in zip(self.remotes, actions_per_env):
            remote.send(("batched_rollout", (actions, n_steps)))
        
        # Collect results (blocking, but all workers run in parallel)
        all_results = []
        for remote in self.remotes:
            env_results = remote.recv()
            processed = []
            for obs, reward, term, trunc, info, type_mask, size_mask, amount_mask in env_results:
                processed.append((
                    np.array(obs),
                    reward,
                    term,
                    trunc,
                    info,
                    np.array(type_mask),
                    np.array(size_mask),
                    np.array(amount_mask),
                ))
            all_results.append(processed)
        
        return all_results

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
) -> SubprocessVecEnv:
    """
    Factory function to create subprocess vectorized environment.
    
    Args:
        config: Configuration object
        n_envs: Number of parallel environments
        learned_reward_model: Optional learned reward model for RLHF
        
    Returns:
        SubprocessVecEnv instance with factored action support
    """
    from rl_editor.environment import AudioEditingEnvFactored
    
    def make_env():
        return AudioEditingEnvFactored(config, learned_reward_model=learned_reward_model)
    
    env_fns = [make_env for _ in range(n_envs)]
    
    return SubprocessVecEnv(env_fns, start_method="spawn")

