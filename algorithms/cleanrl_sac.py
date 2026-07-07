# Multi-Agent SAC Navigation — adapted from optimised_MARL/cleanrl_sac.py
# Uses the pure-Python Fossen-dynamics usv_nav_env instead of HoloOcean.
#
# Key changes vs. the original:
#   - No HoloOcean / multiprocessing worker needed (env is pure Python + fast)
#   - HeterogeneousSwarmVectorEnv replaced with a simple SyncVectorEnv
#   - SwarmToCleanRLWrapper adapted for usv_nav_env's dict API
#   - All neural-net architectures (Actor, Critic, DeepSetOAB, EntitySetEncoder)
#     kept 100% identical
#   - SAC training loop kept 100% identical
#   - Curriculum scheduling kept identical

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from algorithms.buffers import ReplayBuffer
from envs.usv_nav_env import MultiAgentNavEnv


# ═══════════════════════════════════════════════════════════════════════════════
# WRAPPER — translates usv_nav_env's dict API into CleanRL's flat-array API
# ═══════════════════════════════════════════════════════════════════════════════
class SwarmToCleanRLWrapper(gym.Wrapper):
    """
    Translates the swarm dictionary API into CleanRL's vector-like array API.
    Agents interact in one shared world; this wrapper is only for shared-policy
    SAC batching and replay storage.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = env.num_agents
        self.obs_layout = env.obs_layout

        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = 0

        # Expose single spaces for CleanRL's neural-network initialisation
        self.single_observation_space = env.single_obs_space
        self.single_action_space      = env.single_action_space

        # Trick CleanRL into treating this as a standard vector env
        self.observation_space = gym.spaces.Box(
            low=np.tile(self.single_observation_space.low,  (self.num_envs, 1)),
            high=np.tile(self.single_observation_space.high, (self.num_envs, 1)),
        )
        self.action_space = gym.spaces.Box(
            low=np.tile(self.single_action_space.low,  (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
        )
        self.worker_idx = 0

    def reset(self, **kwargs):
        self.episode_returns.fill(0.0)
        self.episode_lengths = 0
        obs_dict, info_dict = self.env.reset(**kwargs)
        
        obs_list = []
        for i in range(self.num_envs):
            aid = f"vessel{i}"
            if aid in obs_dict:
                obs_list.append(obs_dict[aid])
            else:
                obs_list.append(np.zeros(self.single_observation_space.shape, dtype=np.float32))
        
        obs_array = np.stack(obs_list)
        return obs_array, info_dict

    def step(self, action_array):
        # Only send actions for agents that are still alive in the environment
        action_dict = {f"vessel{i}": action_array[i] for i in range(self.num_envs) if f"vessel{i}" in self.env.agents}

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)

        obs_array   = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype=np.float32)
        rew_array   = np.zeros(self.num_envs, dtype=np.float32)
        term_array  = np.ones(self.num_envs, dtype=bool)   # default to True (dead)
        trunc_array = np.zeros(self.num_envs, dtype=bool)
        
        for i in range(self.num_envs):
            aid = f"vessel{i}"
            if aid in obs_dict:
                obs_array[i]   = obs_dict[aid]
                rew_array[i]   = rew_dict.get(aid, 0.0)
                term_array[i]  = term_dict.get(aid, False)
                trunc_array[i] = trunc_dict.get(aid, False)
            # If not in dict, it stays dead (term_array[i] = True)

        info_dict["dist_to_goal"] = np.array(
            [info_dict.get(f"vessel{i}", {}).get("dist_to_goal", 0.0)
             for i in range(self.num_envs)],
            dtype=np.float32,
        )

        if "reward_metrics" in info_dict:
            pass # already handled cleanly by the new PZ environment

        self.episode_returns += rew_array
        self.episode_lengths += 1

        dones = term_array | trunc_array

        # ── Global auto-reset ────────────────────────────────────────────────
        world_term  = False
        world_trunc = False

        if dones.all() or trunc_array.any():
            info_dict["final_observation"] = obs_array.copy()
            info_dict["raw_terminations"]  = term_array.copy()
            info_dict["raw_truncations"]   = trunc_array.copy()
            info_dict["world_reset"]       = True
            info_dict["episode"] = {
                "r": self.episode_returns.copy(),
                "l": np.full(self.num_envs, self.episode_lengths, dtype=np.int32),
            }

            if trunc_array.all():
                world_trunc = True
            else:
                world_term = True

            obs_dict, reset_info = self.env.reset()
            self.episode_returns.fill(0.0)
            self.episode_lengths = 0
            
            obs_list = []
            for i in range(self.num_envs):
                aid = f"vessel{i}"
                if aid in obs_dict:
                    obs_list.append(obs_dict[aid])
                else:
                    obs_list.append(np.zeros(self.single_observation_space.shape, dtype=np.float32))
            obs_array = np.stack(obs_list)
            
            info_dict.update(reset_info)

        info_dict["_world_term"]  = world_term
        info_dict["_world_trunc"] = world_trunc

        return obs_array, rew_array, term_array, trunc_array, info_dict

    def configure_curriculum(self, **kwargs):
        self.env.configure_curriculum(**kwargs)

    def get_goals(self):
        return self.env.goals


import multiprocessing as mp
import cloudpickle

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rew, term, trunc, info = env.step(data)
                remote.send((obs, rew, term, trunc, info))
            elif cmd == "reset":
                obs, info = env.reset(seed=data.get("seed"), options=data.get("options"))
                remote.send((obs, info))
            elif cmd == "configure_curriculum":
                env.configure_curriculum(**data)
                remote.send(None)
            elif cmd == "get_goals":
                remote.send(env.get_goals())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Got unrecognized cmd {cmd}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Worker Error] {e}")
        remote.send(("ERROR", e))
    finally:
        env.close()

class CloudpickleWrapper:
    def __init__(self, x): self.fn = x
    def __getstate__(self): return cloudpickle.dumps(self.fn)
    def __setstate__(self, ob): self.fn = cloudpickle.loads(ob)

class AsyncSwarmVectorEnv:
    """
    Runs multiple independent SwarmToCleanRLWrapper envs in separate processes.
    """
    def __init__(self, env_fns):
        self.num_worlds = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_worlds)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

        sample = env_fns[0]()
        self.single_observation_space = sample.single_observation_space
        self.single_action_space      = sample.single_action_space
        self.obs_layout               = sample.obs_layout
        self.lidar_range              = getattr(sample.unwrapped, "lidar_range", 50.0)
        self.map_size                 = getattr(sample.unwrapped, "map_size", 200.0)
        self.env_agent_counts         = [sample.num_envs for _ in range(self.num_worlds)]
        self.total_agents             = sum(self.env_agent_counts)
        self.num_envs                 = self.total_agents

        self.observation_space = gym.spaces.Box(
            low=np.tile(self.single_observation_space.low,  (self.total_agents, 1)),
            high=np.tile(self.single_observation_space.high, (self.total_agents, 1)),
        )
        self.action_space = gym.spaces.Box(
            low=np.tile(self.single_action_space.low,  (self.total_agents, 1)),
            high=np.tile(self.single_action_space.high, (self.total_agents, 1)),
        )
        sample.close()

    def reset(self, seed=None, options=None):
        for i, remote in enumerate(self.remotes):
            env_seed = seed + i if seed is not None else None
            remote.send(("reset", {"seed": env_seed, "options": options}))
        
        obs_list, info_dicts = [], []
        for remote in self.remotes:
            obs, info = remote.recv()
            obs_list.append(obs)
            info_dicts.append(info)
        return np.concatenate(obs_list, axis=0), self._merge_infos(info_dicts)

    def step(self, actions):
        idx = 0
        for i, remote in enumerate(self.remotes):
            count = self.env_agent_counts[i]
            remote.send(("step", actions[idx:idx+count]))
            idx += count

        obs_list, rew_list, term_list, trunc_list, info_dicts = [], [], [], [], []
        for remote in self.remotes:
            result = remote.recv()
            if isinstance(result, tuple) and len(result) == 2 and result[0] == "ERROR":
                raise Exception(f"Worker crashed: {result[1]}")
            obs, rew, term, trunc, info = result
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_dicts.append(info)

        return (
            np.concatenate(obs_list,  axis=0),
            np.concatenate(rew_list,  axis=0),
            np.concatenate(term_list, axis=0),
            np.concatenate(trunc_list,axis=0),
            self._merge_infos(info_dicts),
        )

    def configure_curriculum(self, **kwargs):
        for remote in self.remotes:
            remote.send(("configure_curriculum", kwargs))
        for remote in self.remotes:
            remote.recv()

    def get_goals(self):
        for remote in self.remotes:
            remote.send(("get_goals", None))
        goals = {}
        for remote in self.remotes:
            g = remote.recv()
            goals.update(g)
        return goals

    @property
    def goals(self):
        return self.get_goals()

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()

    def _merge_infos(self, info_dicts):
        merged = {}
        dist_arrays = [
            d.get("dist_to_goal", np.zeros(c))
            for d, c in zip(info_dicts, self.env_agent_counts)
        ]
        merged["dist_to_goal"] = np.concatenate(dist_arrays, axis=0)

        merged["_world_term"]  = any(d.get("_world_term",  False) for d in info_dicts)
        merged["_world_trunc"] = any(d.get("_world_trunc", False) for d in info_dicts)

        metrics_dicts = [d.get("reward_metrics", {}) for d in info_dicts if "reward_metrics" in d]
        if metrics_dicts:
            keys = metrics_dicts[0].keys()
            merged["reward_metrics"] = {}
            for k in keys:
                vals = [d.get(k, 0.0) for d in metrics_dicts if k in d and d.get(k) is not None]
                merged["reward_metrics"][k] = sum(vals) / len(vals) if vals else 0.0

        if any(d.get("world_reset", False) for d in info_dicts):
            merged["world_reset"] = True
            final_obs, raw_terms, raw_truncs, ep_r, ep_l = [], [], [], [], []
            for d, c in zip(info_dicts, self.env_agent_counts):
                if d.get("world_reset", False):
                    final_obs.extend(d["final_observation"])
                    raw_terms.extend(d["raw_terminations"])
                    raw_truncs.extend(d["raw_truncations"])
                    ep_r.extend(d["episode"]["r"])
                    ep_l.extend(d["episode"]["l"])
                else:
                    final_obs.extend([None] * c)
                    raw_terms.extend([None] * c)
                    raw_truncs.extend([None] * c)
                    ep_r.extend([0.0] * c)
                    ep_l.extend([0]   * c)
            merged["final_observation"] = final_obs
            merged["raw_terminations"]  = raw_terms
            merged["raw_truncations"]   = raw_truncs
            merged["episode"]           = {"r": ep_r, "l": ep_l}

        return merged

# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE VECTOR ENV — runs N independent copies of the env in the same process
# ═══════════════════════════════════════════════════════════════════════════════
class SimpleSwarmVectorEnv:
    """
    Runs multiple independent SwarmToCleanRLWrapper envs in a single process.
    No multiprocessing — pure Python is fast enough without HoloOcean overhead.
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_worlds = len(self.envs)

        sample = self.envs[0]
        self.single_observation_space = sample.single_observation_space
        self.single_action_space      = sample.single_action_space
        self.obs_layout               = sample.obs_layout
        self.lidar_range              = getattr(sample.unwrapped, "lidar_range", 50.0)
        self.map_size                 = getattr(sample.unwrapped, "map_size", 200.0)

        self.env_agent_counts = [e.num_envs for e in self.envs]
        self.total_agents     = sum(self.env_agent_counts)
        self.num_envs         = self.total_agents

        self.observation_space = gym.spaces.Box(
            low=np.tile(self.single_observation_space.low,  (self.total_agents, 1)),
            high=np.tile(self.single_observation_space.high, (self.total_agents, 1)),
        )
        self.action_space = gym.spaces.Box(
            low=np.tile(self.single_action_space.low,  (self.total_agents, 1)),
            high=np.tile(self.single_action_space.high, (self.total_agents, 1)),
        )

    def reset(self, seed=None, options=None):
        obs_list, info_dicts = [], []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            info_dicts.append(info)
        return np.concatenate(obs_list, axis=0), self._merge_infos(info_dicts)

    def step(self, actions):
        obs_list, rew_list, term_list, trunc_list, info_dicts = [], [], [], [], []
        idx = 0
        for i, env in enumerate(self.envs):
            count = self.env_agent_counts[i]
            obs, rew, term, trunc, info = env.step(actions[idx:idx+count])
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_dicts.append(info)
            idx += count

        return (
            np.concatenate(obs_list,  axis=0),
            np.concatenate(rew_list,  axis=0),
            np.concatenate(term_list, axis=0),
            np.concatenate(trunc_list,axis=0),
            self._merge_infos(info_dicts),
        )

    def _merge_infos(self, info_dicts):
        merged = {}
        dist_arrays = [
            d.get("dist_to_goal", np.zeros(c))
            for d, c in zip(info_dicts, self.env_agent_counts)
        ]
        merged["dist_to_goal"] = np.concatenate(dist_arrays, axis=0)

        merged["_world_term"]  = any(d.get("_world_term",  False) for d in info_dicts)
        merged["_world_trunc"] = any(d.get("_world_trunc", False) for d in info_dicts)

        metrics_dicts = [d.get("reward_metrics", {}) for d in info_dicts if "reward_metrics" in d]
        if metrics_dicts:
            keys = metrics_dicts[0].keys()
            merged["reward_metrics"] = {}
            for k in keys:
                vals = [d.get(k, 0.0) for d in metrics_dicts if k in d and d.get(k) is not None]
                merged["reward_metrics"][k] = sum(vals) / len(vals) if vals else 0.0

        if any(d.get("world_reset", False) for d in info_dicts):
            merged["world_reset"] = True
            final_obs, raw_terms, raw_truncs, ep_r, ep_l = [], [], [], [], []
            for d, c in zip(info_dicts, self.env_agent_counts):
                if d.get("world_reset", False):
                    final_obs.extend(d["final_observation"])
                    raw_terms.extend(d["raw_terminations"])
                    raw_truncs.extend(d["raw_truncations"])
                    ep_r.extend(d["episode"]["r"])
                    ep_l.extend(d["episode"]["l"])
                else:
                    final_obs.extend([None] * c)
                    raw_terms.extend([None] * c)
                    raw_truncs.extend([None] * c)
                    ep_r.extend([0.0] * c)
                    ep_l.extend([0]   * c)
            merged["final_observation"] = final_obs
            merged["raw_terminations"]  = raw_terms
            merged["raw_truncations"]   = raw_truncs
            merged["episode"]           = {"r": ep_r, "l": ep_l}

        return merged

    def configure_curriculum(self, **kwargs):
        for env in self.envs:
            env.configure_curriculum(**kwargs)

    @property
    def goals(self):
        return self.envs[0].get_goals()

    def close(self):
        for env in self.envs:
            env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME STACK (optional LSTM burn-in)
# ═══════════════════════════════════════════════════════════════════════════════
class SwarmFrameStack:
    def __init__(self, env, num_stack=5):
        self.env       = env
        self.num_stack = num_stack
        self.frames    = deque(maxlen=num_stack)

        self.single_observation_space = gym.spaces.Box(
            low=np.tile(env.single_observation_space.low,  (num_stack, 1)),
            high=np.tile(env.single_observation_space.high, (num_stack, 1)),
        )
        self.single_action_space  = env.single_action_space
        self.obs_layout           = env.obs_layout
        self.total_agents         = env.total_agents
        self.num_envs             = env.num_envs
        self.env_agent_counts     = getattr(env, "env_agent_counts", [])
        self.lidar_range          = getattr(env, "lidar_range", 50.0)
        self.map_size             = getattr(env, "map_size",  200.0)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        self.frames.append(obs)
        return self._get_obs(), rew, term, trunc, info

    def _get_obs(self):
        return np.stack(self.frames, axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""
    evaluate: bool = False
    """if toggled, disables all training and runs pure deterministic inference"""
    log_rewards_csv: bool = False
    """log individual reward components to a csv file during training"""

    # ── Algorithm ──────────────────────────────────────────────────────────────
    env_id: str = "MultiAgentNav-v0"
    """the environment id of the task"""
    num_worlds: int = 2
    """number of parallel env instances"""
    total_timesteps: int = 40_000
    """total timesteps of the experiments"""
    num_envs: int = 2
    """the number of parallel game environments"""
    max_episode_steps: int = 10000
    """max steps per episode"""
    num_agents: int = 6
    """number of agents per env"""
    num_moving_obstacles: int = 10
    """number of scripted moving obstacle vessels"""
    sim_ticks_per_sec: int = 10
    """simulation steps per second (unused — kept for API compat)"""    
    ticks_per_step: int = 1
    """env steps per RL action"""
    max_static_obstacles_spawned: int = 150
    """hard cap on static obstacles spawned per reset"""
    shared_goal: bool = False
    """give all agents the same goal instead of individual goals"""
    show_viewport: bool = False
    """unused — kept for API compat"""
    show_visualizer: bool = False
    """show the PyGame per-agent local-view dashboard during training"""

    # ── Curriculum ────────────────────────────────────────────────────────────
    curriculum: bool = True
    """start with easier obstacle settings and ramp to the full task"""
    curriculum_steps: int = 35_000
    """number of global steps used to ramp the curriculum"""
    initial_static_obstacle_keep_prob: float = 0.03
    """fraction of static obstacles spawned at the start of curriculum (near-zero = clear start)"""
    final_static_obstacle_keep_prob: float = 1.0
    """fraction of static obstacles at end of curriculum"""
    initial_moving_obstacle_speed_scale: float = 0.0
    """moving obstacle speed multiplier at the start of curriculum (0 = stationary)"""
    final_moving_obstacle_speed_scale: float = 1.0
    """max speed scale of moving obstacles"""
    max_tracked_entities: int = 20
    """max number of neighbors/obstacles tracked by the Deep Set"""
    initial_goal_dist_max: float = 600.0
    """max goal distance (m) at start of curriculum (closer = easier)"""
    final_goal_dist_max: float = 800.0
    """max goal distance (m) after curriculum finishes"""
    initial_spawn_box_size: float = 25.0
    """spawn box half-width (m) at start of curriculum (larger = agents spread out)"""
    final_spawn_box_size: float = 15.0
    """spawn box size (m) at end of curriculum"""
    initial_cluster_prob: float = 0.0
    """probability of clustered goals at start of curriculum"""
    final_cluster_prob: float = 0.5
    """probability of clustered goals at end of curriculum"""
    initial_cluster_radius: float = 30.0
    """cluster radius (m) at start"""
    final_cluster_radius: float = 10.0
    """cluster radius (m) at end"""
    warm_start_exploration: bool = True
    """use a noisy goal-seeking controller before SAC updates begin"""
    warm_start_noise: float = 0.15
    """standard deviation of steering noise in warm-start exploration"""
    warm_start_throttle: float = 0.5
    """nominal throttle used by warm-start exploration"""

    # ── SAC ───────────────────────────────────────────────────────────────────
    buffer_size: int = int(2e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 3000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    utd_ratio: int = 1
    """number of gradient updates per environment step (Update-To-Data ratio)"""


# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def linear_schedule(start, end, step, duration):
    if duration <= 0:
        return end
    mix = min(max(step / duration, 0.0), 1.0)
    return (1.0 - mix) * start + mix * end


def apply_curriculum(env, args, global_step):
    if not args.curriculum:
        return args.final_static_obstacle_keep_prob, args.final_moving_obstacle_speed_scale

    kw = {
        "static_obstacle_keep_prob":   linear_schedule(
            args.initial_static_obstacle_keep_prob,
            args.final_static_obstacle_keep_prob,
            global_step, args.curriculum_steps),
        "moving_obstacle_speed_scale": linear_schedule(
            args.initial_moving_obstacle_speed_scale,
            args.final_moving_obstacle_speed_scale,
            global_step, args.curriculum_steps),
        "goal_dist_max":               linear_schedule(
            args.initial_goal_dist_max,
            args.final_goal_dist_max,
            global_step, args.curriculum_steps),
        "spawn_box_size":              linear_schedule(
            args.initial_spawn_box_size,
            args.final_spawn_box_size,
            global_step, args.curriculum_steps),
        "cluster_prob":                linear_schedule(
            args.initial_cluster_prob,
            args.final_cluster_prob,
            global_step, args.curriculum_steps),
        "cluster_radius":              linear_schedule(
            args.initial_cluster_radius,
            args.final_cluster_radius,
            global_step, args.curriculum_steps),
    }
    env.configure_curriculum(**kw)
    return kw["static_obstacle_keep_prob"], kw["moving_obstacle_speed_scale"]


def get_warm_start_actions(obs, env, args):
    """Simple goal-seeking warm-start controller."""
    layout      = env.obs_layout
    ego_spec    = layout["ego"]
    lidar_spec  = layout["lidar"]
    ego_start   = ego_spec["start"]
    lidar_slice = slice(lidar_spec["start"], lidar_spec["start"] + lidar_spec["dim"])
    lidar_range = getattr(env, "lidar_range", 50.0)

    actions = []
    for agent_obs in obs:
        yaw      = np.arctan2(agent_obs[ego_start],     agent_obs[ego_start + 1])
        goal_yaw = np.arctan2(agent_obs[ego_start + 2], agent_obs[ego_start + 3])
        hd_err   = np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw))

        lidar          = agent_obs[lidar_slice]
        lidar_dists    = lidar[0::2]
        front_clear    = float(np.min(np.concatenate([lidar_dists[:6], lidar_dists[-6:]])) * lidar_range)
        left_clear     = float(np.mean(lidar_dists[8:24]))
        right_clear    = float(np.mean(lidar_dists[-24:-8]))

        avoidance = 0.0
        throttle  = args.warm_start_throttle if abs(hd_err) < 1.2 else 0.1

        if front_clear < 12.0:
            avoidance = 0.6 if left_clear > right_clear else -0.6
            throttle  = 0.05

        steering = 1.1 * hd_err + avoidance
        steering += np.random.normal(0.0, args.warm_start_noise)
        throttle += np.random.normal(0.0, args.warm_start_noise * 0.3)

        actions.append([
            float(np.clip(throttle, -0.15, 0.75)),
            float(np.clip(steering, -0.9,  0.9)),
        ])
    return np.asarray(actions, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURES  (identical to optimised_MARL)
# ═══════════════════════════════════════════════════════════════════════════════
class DeepSetOAB(nn.Module):
    """PointNet-style LiDAR processor accepting [distance, angle] pairs."""
    def __init__(self, lidar_dim=64, out_features=256):
        super().__init__()
        self.num_points = lidar_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, 32), nn.Mish(),
            nn.Linear(32, out_features), nn.Mish(),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.num_points, 2)
        features = self.mlp(x)
        pooled, _ = torch.max(features, dim=1)
        return pooled


class EntitySetEncoder(nn.Module):
    """Deep Sets Encoder for permutation-invariant entity sets."""
    def __init__(self, feature_dim, query_dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feature_dim - 1, embed_dim), nn.Mish(),
            nn.Linear(embed_dim, embed_dim),        nn.Mish(),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, entities, query_features):
        mask = entities[:, :, 0] > 0.5
        h    = self.phi(entities[:, :, 1:])
        h    = h * mask.unsqueeze(-1)
        active_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = h.sum(dim=1) / active_counts
        out    = self.rho(pooled)
        has_entities = mask.any(dim=1, keepdim=True)
        return torch.where(has_entities, out, torch.zeros_like(out))


class CriticBackbone(nn.Module):
    """Centralized Critic Feature Extractor (CTDE)."""
    def __init__(self, env):
        super().__init__()
        self.layout = env.obs_layout

        ego_dim    = self.layout["ego"]["dim"]
        goal_dim   = self.layout["goal"]["dim"]
        global_dim = self.layout.get("global_state", {"dim": 0})["dim"]

        self.kin_net  = nn.Sequential(nn.Linear(ego_dim, 64),  nn.LayerNorm(64),  nn.Mish(), nn.Linear(64, 64), nn.Mish())
        self.goal_net = nn.Sequential(nn.Linear(goal_dim, 32), nn.LayerNorm(32), nn.Mish())

        entity_feature_dim = self.layout["auv_entities"]["feature_dim"]
        ego_query_dim      = 64 + 32
        self.auv_net            = EntitySetEncoder(entity_feature_dim, query_dim=ego_query_dim, embed_dim=64)
        self.moving_obstacle_net= EntitySetEncoder(entity_feature_dim, query_dim=ego_query_dim, embed_dim=64)
        self.lidar_net          = DeepSetOAB(lidar_dim=64, out_features=64)
        self.fusion_norm        = nn.LayerNorm(128)
        self.global_net = nn.Sequential(nn.Linear(global_dim, 64), nn.LayerNorm(64), nn.Mish(), nn.Linear(64, 64), nn.Mish())

        combined_dim = 64 + 32 + 64 + 64 + 64 + 64
        self.lstm       = nn.LSTM(input_size=combined_dim, hidden_size=256, batch_first=True)
        self.output_dim = 256

    def _vector(self, x, name):
        spec = self.layout[name]
        return x[:, spec["start"]: spec["start"] + spec["dim"]]

    def _entities(self, x, name):
        spec = self.layout[name]
        flat = x[:, spec["start"]: spec["start"] + spec["dim"]]
        return flat.reshape(x.shape[0], spec["count"], spec["feature_dim"])

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(batch_size * seq_len, -1)

        kin_feat    = self.kin_net(self._vector(x_flat, "ego"))
        goal_feat   = self.goal_net(self._vector(x_flat, "goal"))
        ego_query   = torch.cat([kin_feat, goal_feat], dim=1)
        auv_feat    = self.auv_net(self._entities(x_flat, "auv_entities"),  ego_query)
        moving_feat = self.moving_obstacle_net(self._entities(x_flat, "moving_obstacles"), ego_query)
        lidar_feat  = self.lidar_net(self._vector(x_flat, "lidar"))
        global_feat = self.global_net(self._vector(x_flat, "global_state"))

        fused       = self.fusion_norm(torch.cat([kin_feat, lidar_feat], dim=1))
        combined    = torch.cat([fused, goal_feat, auv_feat, moving_feat, global_feat], dim=1)
        combined_seq= combined.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(combined_seq)
        return lstm_out[:, -1, :]


class ActorBackbone(nn.Module):
    """Decentralized Actor Feature Extractor (local observation only)."""
    def __init__(self, env):
        super().__init__()
        self.layout = env.obs_layout

        ego_dim  = self.layout["ego"]["dim"]
        goal_dim = self.layout["goal"]["dim"]

        self.kin_net  = nn.Sequential(nn.Linear(ego_dim, 64),  nn.LayerNorm(64),  nn.Mish(), nn.Linear(64, 64), nn.Mish())
        self.goal_net = nn.Sequential(nn.Linear(goal_dim, 32), nn.LayerNorm(32), nn.Mish())

        entity_feature_dim = self.layout["auv_entities"]["feature_dim"]
        ego_query_dim      = 64 + 32
        self.auv_net            = EntitySetEncoder(entity_feature_dim, query_dim=ego_query_dim, embed_dim=64)
        self.moving_obstacle_net= EntitySetEncoder(entity_feature_dim, query_dim=ego_query_dim, embed_dim=64)
        self.lidar_net          = DeepSetOAB(lidar_dim=64, out_features=64)
        self.fusion_norm        = nn.LayerNorm(128)

        combined_dim = 64 + 32 + 64 + 64 + 64
        self.lstm       = nn.LSTM(input_size=combined_dim, hidden_size=256, batch_first=True)
        self.output_dim = 256

    def _vector(self, x, name):
        spec = self.layout[name]
        return x[:, spec["start"]: spec["start"] + spec["dim"]]

    def _entities(self, x, name):
        spec = self.layout[name]
        flat = x[:, spec["start"]: spec["start"] + spec["dim"]]
        return flat.reshape(x.shape[0], spec["count"], spec["feature_dim"])

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(batch_size * seq_len, -1)

        kin_feat    = self.kin_net(self._vector(x_flat, "ego"))
        goal_feat   = self.goal_net(self._vector(x_flat, "goal"))
        ego_query   = torch.cat([kin_feat, goal_feat], dim=1)
        auv_feat    = self.auv_net(self._entities(x_flat, "auv_entities"),  ego_query)
        moving_feat = self.moving_obstacle_net(self._entities(x_flat, "moving_obstacles"), ego_query)
        lidar_feat  = self.lidar_net(self._vector(x_flat, "lidar"))

        fused       = self.fusion_norm(torch.cat([kin_feat, lidar_feat], dim=1))
        combined    = torch.cat([fused, goal_feat, auv_feat, moving_feat], dim=1)
        combined_seq= combined.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(combined_seq)
        return lstm_out[:, -1, :]


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = CriticBackbone(env)
        action_dim = int(np.prod(env.single_action_space.shape))
        self.q_net = nn.Sequential(
            nn.Linear(self.backbone.output_dim + action_dim, 256), nn.Mish(),
            nn.Linear(256, 128), nn.Mish(),
            nn.Linear(128, 1),
        )

    def forward(self, x, a):
        fusion = self.backbone(x)
        q_in   = torch.cat([fusion, a], dim=1)
        return self.q_net(q_in)


LOG_STD_MAX =  2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone     = ActorBackbone(env)
        self.decision_net = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 256), nn.Mish(),
            nn.Linear(256, 128), nn.Mish(),
        )
        action_dim = int(np.prod(env.single_action_space.shape))
        self.fc_mean   = nn.Linear(128, action_dim)
        self.fc_logstd = nn.Linear(128, action_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        fusion  = self.backbone(x)
        hidden  = self.decision_net(fusion)
        mean    = self.fc_mean(hidden)
        log_std = torch.clamp(self.fc_logstd(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def get_deterministic_action(self, x):
        mean, _ = self(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.evaluate:
        run_name = f"EVAL__{run_name}"

    if args.track and not args.evaluate:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # ── Seeding ───────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Speed up matmul on GPUs with tensor cores (suppresses the UserWarning)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"[INFO] PyTorch device: {device}")

    # ── Build envs ────────────────────────────────────────────────────────────
    def make_env(seed, idx, swarm_size):
        def thunk():
            base_env = MultiAgentNavEnv(
                num_agents=swarm_size,
                num_moving_obstacles=args.num_moving_obstacles,
                max_steps=args.max_episode_steps,
                max_static_obstacles_spawned=args.max_static_obstacles_spawned,
                shared_goal=args.shared_goal,
                max_auv_entities=args.max_tracked_entities,
                max_static_obstacles=args.max_tracked_entities,
                max_moving_obstacles=args.max_tracked_entities,
                static_obstacle_keep_prob=(
                    args.initial_static_obstacle_keep_prob
                    if args.curriculum else args.final_static_obstacle_keep_prob
                ),
                moving_obstacle_speed_scale=(
                    args.initial_moving_obstacle_speed_scale
                    if args.curriculum else args.final_moving_obstacle_speed_scale
                ),
            )
            return SwarmToCleanRLWrapper(base_env)
        return thunk

    swarm_sizes = [args.num_agents for _ in range(args.num_worlds)]
    # envs = SimpleSwarmVectorEnv(
    #     [make_env(args.seed + i, i, swarm_sizes[i]) for i in range(args.num_worlds)]
    # )
    envs = AsyncSwarmVectorEnv(
        [make_env(args.seed + i, i, swarm_sizes[i]) for i in range(args.num_worlds)]
    )
    envs = SwarmFrameStack(envs, num_stack=5)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action spaces supported"

    # ── Neural networks ───────────────────────────────────────────────────────
    def apply_orthogonal_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    actor_raw = Actor(envs).to(device)
    actor_raw.apply(apply_orthogonal_init)
    actor = torch.compile(actor_raw)

    qf1_raw = SoftQNetwork(envs).to(device)
    qf1_raw.apply(apply_orthogonal_init)
    qf1 = torch.compile(qf1_raw)

    qf2_raw = SoftQNetwork(envs).to(device)
    qf2_raw.apply(apply_orthogonal_init)
    qf2 = torch.compile(qf2_raw)

    qf1_target_raw = SoftQNetwork(envs).to(device)
    qf1_target_raw.load_state_dict(qf1_raw.state_dict())
    qf1_target = torch.compile(qf1_target_raw)

    qf2_target_raw = SoftQNetwork(envs).to(device)
    qf2_target_raw.load_state_dict(qf2_raw.state_dict())
    qf2_target = torch.compile(qf2_target_raw)

    q_optimizer     = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr,     weight_decay=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()),                          lr=args.policy_lr, weight_decay=1e-4)

    # AMP Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.cuda)

    if args.evaluate:
        args.learning_starts = float("inf")
        args.warm_start_exploration = False
        args.show_visualizer = True

    # ── Entropy tuning ────────────────────────────────────────────────────────
    if args.autotune:
        target_entropy = -0.5 * torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        alpha      = log_alpha.exp().item()
        a_optimizer= optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # ── Optional checkpoint pre-load ──────────────────────────────────────────
    _SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path= None   # set this path to resume from a checkpoint

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["actor_state_dict"])
            qf1.load_state_dict(ckpt["qf1_state_dict"])
            qf2.load_state_dict(ckpt["qf2_state_dict"])
            qf1_target.load_state_dict(ckpt["qf1_target_state_dict"])
            qf2_target.load_state_dict(ckpt["qf2_target_state_dict"])
            actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
            q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
            if args.autotune and "log_alpha" in ckpt:
                with torch.no_grad():
                    log_alpha.copy_(ckpt["log_alpha"])
                a_optimizer.load_state_dict(ckpt["a_optimizer_state_dict"])
        else:
            actor.load_state_dict(ckpt)
        print("[INFO] Checkpoint loaded.")

    # ── Replay buffer ─────────────────────────────────────────────────────────
    # =========================================================================
    # PRIORITIZED EXPERIENCE REPLAY (PER) INITIALIZATION - FOR FUTURE USE
    # =========================================================================
    # To enable PER, replace the ReplayBuffer below with:
    # from buffers import PrioritizedReplayBuffer
    # rb = PrioritizedReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     n_envs=envs.num_envs,
    #     alpha=0.6,
    # )
    # =========================================================================
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=envs.num_envs,
        handle_timeout_termination=False,
    )

    # ── Visualizer ────────────────────────────────────────────────────────────
    if args.show_visualizer:
        from visualization.visualiser import Swarm2DVisualizer
        visualizer = Swarm2DVisualizer(
            num_agents=envs.env_agent_counts[0],
            lidar_range=getattr(envs, "lidar_range", 50.0),
        )
    else:
        visualizer = None

    # ── CSV logging ───────────────────────────────────────────────────────────
    import csv
    csv_file   = None
    csv_writer = None
    if args.log_rewards_csv:
        csv_file   = open("rewards_log.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["global_step", "phi_dist", "phi_heading", "phi_swarm", "phi_obstacle", "pbrs_shaping", "r_total"])

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[INFO] Starting environment...")
    obs, _     = envs.reset(seed=args.seed)
    is_alive   = np.ones(envs.num_envs, dtype=bool)
    start_time = time.time()
    ep_count   = 0
    best_return= -np.inf

    import tqdm
    desc_text = "Inference Steps" if args.evaluate else "Training Steps"

    for global_step in tqdm.tqdm(range(args.total_timesteps), desc=desc_text):
        static_keep_prob, moving_speed_scale = apply_curriculum(envs, args, global_step)

        # ── Select actions ────────────────────────────────────────────────────
        if global_step < args.learning_starts:
            if args.warm_start_exploration:
                current_obs = obs[:, -1, :] if obs.ndim == 3 else obs
                actions     = get_warm_start_actions(current_obs, envs, args)
            else:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = actor.get_action(torch.from_numpy(obs).to(device, non_blocking=True))
            actions    = actions.detach().cpu().numpy()

        actions = actions * is_alive.reshape(-1, 1)

        # ── Env step ─────────────────────────────────────────────────────────
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # ── Visualizer ───────────────────────────────────────────────────────
        if visualizer is not None:
            first_env_count = envs.env_agent_counts[0]
            obs_dict = {f"vessel{i}": next_obs[i] for i in range(first_env_count)}
            try:
                visualizer.render(obs_dict, infos, envs.goals)
            except Exception as e:
                print(f"[Visualizer Error] {e}")

        # ── CSV logging ───────────────────────────────────────────────────────
        if args.log_rewards_csv and csv_writer and "reward_metrics" in infos and infos["reward_metrics"]:
            rm = infos["reward_metrics"]
            csv_writer.writerow([
                global_step,
                rm.get("phi_dist", 0.0),
                rm.get("phi_heading", 0.0),
                rm.get("phi_swarm", 0.0),
                rm.get("phi_obstacle", 0.0),
                rm.get("pbrs_shaping", 0.0),
                rm.get("r_total", 0.0),
            ])
            csv_file.flush()

        # ── Episode stats ─────────────────────────────────────────────────────
        if "episode" in infos:
            ep_count += 1
            episode_returns = np.asarray(infos["episode"]["r"], dtype=np.float32)
            episode_lengths = np.asarray(infos["episode"]["l"], dtype=np.int32)
            distances       = np.asarray(infos.get("dist_to_goal", np.zeros(envs.total_agents)), dtype=np.float32)

            valid = episode_returns[episode_lengths > 0]
            ep_return  = float(np.mean(valid))  if len(valid) else 0.0
            min_return = float(np.min(valid))   if len(valid) else 0.0
            max_return = float(np.max(valid))   if len(valid) else 0.0
            ep_len     = int(episode_lengths[0])
            dist       = float(np.mean(distances))
            min_dist   = float(np.min(distances))
            terminal_label = ", ".join(
                f"{ag}:{ev}"
                for ag, ev in zip(
                    infos.get("terminal_agents", []),
                    infos.get("terminal_events", []),
                )
            ) or "none"

            if ep_return > best_return:
                best_return = ep_return
                os.makedirs(f"runs/{run_name}", exist_ok=True)
                torch.save(actor.state_dict(), f"runs/{run_name}/actor_best.pth")

            if ep_count % 100 == 0:
                ckpt = {
                    "global_step":               global_step,
                    "ep_count":                  ep_count,
                    "best_return":               best_return,
                    "actor_state_dict":          actor.state_dict(),
                    "qf1_state_dict":            qf1.state_dict(),
                    "qf2_state_dict":            qf2.state_dict(),
                    "qf1_target_state_dict":     qf1_target.state_dict(),
                    "qf2_target_state_dict":     qf2_target.state_dict(),
                    "actor_optimizer_state_dict":actor_optimizer.state_dict(),
                    "q_optimizer_state_dict":    q_optimizer.state_dict(),
                }
                if args.autotune:
                    ckpt["log_alpha"]             = log_alpha
                    ckpt["a_optimizer_state_dict"]= a_optimizer.state_dict()
                os.makedirs(f"runs/{run_name}", exist_ok=True)
                torch.save(ckpt, f"runs/{run_name}/full_checkpoint_ep{ep_count}.pth")
                print(f"Full checkpoint saved at EP {ep_count}")

            print(
                f"EP {ep_count} | step={global_step} | mean_return={ep_return:.2f} "
                f"| min_return={min_return:.2f} | len={ep_len} | event={terminal_label} "
                f"| mean_dist={dist:.2f} | min_dist={min_dist:.2f} "
                f"| static={static_keep_prob:.2f} | moving={moving_speed_scale:.2f} "
                f"| best_mean={best_return:.2f}"
            )
            if writer:
                writer.add_scalar("charts/episodic_return",     ep_return,  global_step)
                writer.add_scalar("charts/episodic_return_min", min_return, global_step)
                writer.add_scalar("charts/episodic_return_max", max_return, global_step)
                writer.add_scalar("charts/episodic_length",     ep_len,     global_step)
                writer.add_scalar("charts/dist_to_goal_final",  dist,       global_step)
                writer.add_scalar("charts/dist_to_goal_min",    min_dist,   global_step)
                writer.add_scalar("charts/static_obstacle_keep_prob",   static_keep_prob,   global_step)
                writer.add_scalar("charts/moving_obstacle_speed_scale", moving_speed_scale, global_step)

        # ── Replay buffer ─────────────────────────────────────────────────────
        real_next_obs = next_obs.copy()
        dones         = terminations | truncations

        if "final_observation" in infos:
            for idx in range(envs.num_envs):
                if dones[idx] and infos["final_observation"][idx] is not None:
                    real_next_obs[idx] = infos["final_observation"][idx]
            is_alive = np.ones(envs.num_envs, dtype=bool)

        active_mask_for_buffer = is_alive.copy()
        is_alive = is_alive & ~dones

        if not args.evaluate:
            clipped_rewards = np.clip(rewards, -500.0, 500.0)
            rb.add(obs, real_next_obs, actions, clipped_rewards, dones, infos,
                   active=active_mask_for_buffer)

        obs = next_obs

        # ── SAC Updates ───────────────────────────────────────────────────────
        if global_step > args.learning_starts:
            for _ in range(args.utd_ratio):
                data                = rb.sample(args.batch_size)
                actor_loss_value    = None
                alpha_loss_value    = None

                with torch.cuda.amp.autocast(enabled=args.cuda):
                    with torch.no_grad():
                        next_actions, next_log_pi = actor.get_action(data.next_observations)
                        qf1_next = qf1_target(data.next_observations, next_actions)
                        qf2_next = qf2_target(data.next_observations, next_actions)
                        min_qf_next    = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                        next_q_value   = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next.view(-1)

                    qf1_a = qf1(data.observations, data.actions).view(-1)
                    qf2_a = qf2(data.observations, data.actions).view(-1)
                    valid_samples = data.active_masks.sum().clamp(min=1.0)
                    qf1_loss = (F.mse_loss(qf1_a, next_q_value, reduction="none") * data.active_masks.view(-1)).sum() / valid_samples
                    qf2_loss = (F.mse_loss(qf2_a, next_q_value, reduction="none") * data.active_masks.view(-1)).sum() / valid_samples
                    qf_loss  = qf1_loss + qf2_loss
                    
                    # =========================================================================
                    # PRIORITIZED EXPERIENCE REPLAY (PER) UPDATE - FOR FUTURE USE
                    # =========================================================================
                    # 1. Update your sample call above to: data, indices, is_weights = rb.sample(args.batch_size)
                    # 2. Modify qf1_loss and qf2_loss above to multiply by is_weights:
                    #    qf1_loss = (F.mse_loss(...) * data.active_masks.view(-1) * is_weights).sum() / valid_samples
                    # 3. Calculate absolute TD-error to feedback into the tree:
                    #    td_errors = (next_q_value - qf1_a).detach().abs().cpu().numpy()
                    # 4. Update the buffer priorities:
                    #    rb.update_priorities(indices, td_errors)
                    # =========================================================================

                q_optimizer.zero_grad(set_to_none=True)
                scaler.scale(qf_loss).backward()
                scaler.unscale_(q_optimizer)
                nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), 5.0)
                scaler.step(q_optimizer)

                if global_step % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        with torch.cuda.amp.autocast(enabled=args.cuda):
                            pi, log_pi = actor.get_action(data.observations)
                            qf1_pi     = qf1(data.observations, pi)
                            qf2_pi     = qf2(data.observations, pi)
                            min_qf_pi  = torch.min(qf1_pi, qf2_pi)
                            actor_loss = (((alpha * log_pi) - min_qf_pi) * data.active_masks).sum() / valid_samples

                        actor_optimizer.zero_grad(set_to_none=True)
                        scaler.scale(actor_loss).backward()
                        scaler.unscale_(actor_optimizer)
                        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                        scaler.step(actor_optimizer)
                        actor_loss_value = actor_loss.item()

                        if args.autotune:
                            with torch.cuda.amp.autocast(enabled=args.cuda):
                                with torch.no_grad():
                                    _, log_pi = actor.get_action(data.observations)
                                alpha_loss = ((-log_alpha.exp() * (log_pi + target_entropy)) * data.active_masks).sum() / valid_samples
                            a_optimizer.zero_grad(set_to_none=True)
                            scaler.scale(alpha_loss).backward()
                            scaler.step(a_optimizer)
                            with torch.no_grad():
                                log_alpha.clamp_(min=-4.6)
                            alpha = log_alpha.exp().item()
                            alpha_loss_value = alpha_loss.item()

                scaler.update()

            if global_step % args.target_network_frequency == 0:
                for param, tgt in zip(qf1.parameters(), qf1_target.parameters()):
                    tgt.data.copy_(args.tau * param.data + (1 - args.tau) * tgt.data)
                for param, tgt in zip(qf2.parameters(), qf2_target.parameters()):
                    tgt.data.copy_(args.tau * param.data + (1 - args.tau) * tgt.data)

            if global_step % 100 == 0 and writer:
                if actor_loss_value is not None:
                    writer.add_scalar("losses/actor_loss", actor_loss_value, global_step)
                    writer.add_scalar("losses/alpha",      alpha,            global_step)
                writer.add_scalar("losses/qf1_values", qf1_a.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss",   qf1_loss.item(),     global_step)
                writer.add_scalar("losses/qf2_loss",   qf2_loss.item(),     global_step)
                writer.add_scalar("losses/qf_loss",    qf_loss.item() / 2.0,global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if alpha_loss_value is not None:
                    writer.add_scalar("losses/alpha_loss", alpha_loss_value, global_step)

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if global_step % 50_000 == 0 and not args.evaluate:
            periodic_ckpt = {
                "global_step":               global_step,
                "actor_state_dict":          actor.state_dict(),
                "qf1_state_dict":            qf1.state_dict(),
                "qf2_state_dict":            qf2.state_dict(),
                "qf1_target_state_dict":     qf1_target.state_dict(),
                "qf2_target_state_dict":     qf2_target.state_dict(),
                "actor_optimizer_state_dict":actor_optimizer.state_dict(),
                "q_optimizer_state_dict":    q_optimizer.state_dict(),
            }
            if args.autotune:
                periodic_ckpt["log_alpha"]             = log_alpha
                periodic_ckpt["a_optimizer_state_dict"]= a_optimizer.state_dict()
            os.makedirs(f"runs/{run_name}", exist_ok=True)
            torch.save(periodic_ckpt, f"runs/{run_name}/actor_{global_step}.pth")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("Saving Replay Buffer to disk (this might take a minute depending on size)...")
    rb.save(f"runs/{run_name}/replay_buffer.npz")
    print(f"Replay Buffer saved to runs/{run_name}/replay_buffer.npz")
    
    envs.close()
    if csv_file:
        csv_file.close()
    if writer:
        writer.close()
    print("Training complete.")
