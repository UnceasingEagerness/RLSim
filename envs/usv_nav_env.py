"""
usv_nav_env.py
==============
Multi-Agent Navigation Environment using Fossen USV Dynamics.

This environment replaces `surfacevessel_env.py` from the optimised_MARL
codebase.  Instead of HoloOcean, it uses the pure-Python `USVDynamics`
model from the usv_pe_simulator, making it self-contained and fast.

Task
----
Each vessel must reach its individual goal while avoiding:
  * Static circular obstacles scattered across the map
  * Moving obstacles (scripted patrol-point USVs)
  * Other agents in the swarm

Observation Layout (mirrors optimised_MARL)
-------------------------------------------
  ego            : 7 dims  — [sin(yaw), cos(yaw), sin(goal_yaw), cos(goal_yaw),
                               throttle, steering, yaw_rate]
  goal           : 2 dims  — [normalized_dist, heading_cos_error]
  auv_entities   : max_auv_entities * 5 dims  — padded [active, rx, ry, rvx, rvy]
  static_obs     : max_static_obstacles * 5 dims
  moving_obs     : max_moving_obstacles * 5 dims
  lidar          : 128 dims — 64 beams × [distance_norm, angle_norm]
  global_state   : max_auv_entities * 5 dims  — CTDE global state

Action Space
------------
  [throttle_norm ∈ [-1,1], steering_norm ∈ [-1,1]]
  → mapped to Fossen thruster commands [tau_u, tau_r]
"""

import sys
import os

# Automatically add the simulator root to sys.path so 'env' module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'usv_pe_simulator')))

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Locate the usv_pe_simulator dynamics model ────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR    = os.path.join(_THIS_DIR, "..", "..", "usv_pe_simulator")
sys.path.insert(0, _SIM_DIR)
from envs.usv_dynamics import USVDynamics

from envs.perception_utils import PerceptionModule


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic LiDAR
# ─────────────────────────────────────────────────────────────────────────────
def _synthetic_lidar(ego_pos, ego_yaw, static_obstacles, moving_obstacles,
                      lidar_range=50.0, num_beams=64):
    """
    Ray-cast against circular obstacles to produce a 64-beam LiDAR array.

    Returns distances in metres, capped at lidar_range.
    """
    distances = np.full(num_beams, lidar_range, dtype=np.float32)
    beam_angles = np.arange(num_beams) * (2 * np.pi / num_beams)

    all_obstacles = []
    for obs in static_obstacles:
        all_obstacles.append((obs["pos"][:2], obs["radius"]))
    for obs in moving_obstacles:
        all_obstacles.append((obs["pos"][:2], obs.get("radius", 2.5)))

    ex, ey = ego_pos[0], ego_pos[1]
    for bi, rel_angle in enumerate(beam_angles):
        world_angle = ego_yaw + rel_angle
        dx = math.cos(world_angle)
        dy = math.sin(world_angle)
        min_dist = lidar_range
        for (cx, cy), r in all_obstacles:
            fx, fy = cx - ex, cy - ey
            # Ray-circle intersection
            b = 2 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - r * r
            disc = b * b - 4 * c          # a=1 because |d|=1
            if disc < 0:
                continue
            t = (-b - math.sqrt(disc)) / 2.0
            if 0.01 < t < min_dist:
                min_dist = t
        distances[bi] = min_dist

    return distances


# ─────────────────────────────────────────────────────────────────────────────
# Episode Logger
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeLogger:
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _RESET  = "\033[0m"

    def __init__(self):
        self.ep_num = 0
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._returns = {}
        self._steps   = 0

    def reset(self, agent_ids):
        self.ep_num += 1
        self._returns = {a: 0.0 for a in agent_ids}
        self._steps   = 0

    def step(self, reward_dict):
        self._steps += 1
        for aid, r in reward_dict.items():
            self._returns[aid] = self._returns.get(aid, 0.0) + r

    def log_termination(self, agent_id, outcome, dist_to_goal,
                        min_lidar_m, min_neighbor_m, step_count):
        colour = {
            "GOAL":        self._GREEN,
            "COLLISION":   self._RED,
            "STUCK":       self._YELLOW,
            "TIMEOUT":     self._CYAN,
        }.get(outcome, self._RESET)

        nbor_str = f"{min_neighbor_m:6.1f}m" if min_neighbor_m is not None else "   —  "
        ret = self._returns.get(agent_id, 0.0)
        print(
            f"[EP {self.ep_num:4d} | step {step_count:4d} | len {self._steps:4d}] "
            f"{colour}{outcome:<14s}{self._RESET} "
            f"agent={agent_id:<9s} "
            f"dist={dist_to_goal:7.1f}m  "
            f"lidar_min={min_lidar_m:5.1f}m  "
            f"nbor_min={nbor_str}  "
            f"return={ret:+9.1f}",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reward Calculator
# ─────────────────────────────────────────────────────────────────────────────
class NavReward:
    """
    Mathematically airtight Potential-Based Reward Shaping (PBRS) function.
    Total Reward = (1.0 * Phi_now) - Phi_prev + R_step
    """
    # Coefficients
    GOAL_RADIUS      = 3.0       # metres — goal capture distance
    COLLISION_RADIUS = 1.5       # metres — hard collision

    def __init__(self, map_size=1500.0):
        self.map_size = map_size
        self.prev_phi = None
        self.last_event = None
        self.last_metrics = {}

    def reset(self):
        self.prev_phi = None
        self.last_event = None
        self.last_metrics = {}

    def compute(self, pos, yaw, goal_pos, lidar_distances, swarm_context):
        """
        pos            : [x, y, z]   (z unused)
        yaw            : float radians
        goal_pos       : [x, y, 0]
        lidar_distances: np.array(64,) in metres
        swarm_context  : np.array(N, 5) [active, rx, ry, rvx, rvy]
        Returns (reward, terminated, event_str)
        """
        p = pos[:2]
        g = goal_pos[:2]
        dist = float(np.linalg.norm(g - p))
        
        current_yaw_rad = yaw
        goal_yaw_rad = math.atan2(g[1] - p[1], g[0] - p[0])
        
        # Calculate heading error in degrees
        heading_error_rad = (goal_yaw_rad - current_yaw_rad + math.pi) % (2 * math.pi) - math.pi
        heading_error_deg = float(np.rad2deg(heading_error_rad))
        
        # 1. Base PBRS Potentials
        phi_dist = -5.0 * dist
        phi_heading = -20.0 * abs(heading_error_deg)
        
        # 2. Swarm Potential
        phi_swarm = 0.0
        closest_neighbor_dist = np.inf
        agent_collision = False
        
        if swarm_context is not None and len(swarm_context) > 0:
            active = swarm_context[swarm_context[:, 0] > 0.5]
            if len(active):
                nbor_dists = np.linalg.norm(active[:, 1:3] * self.map_size, axis=1)
                closest_neighbor_dist = float(np.min(nbor_dists))
                if closest_neighbor_dist < self.COLLISION_RADIUS:
                    agent_collision = True
                    
        if closest_neighbor_dist < np.inf:
            phi_swarm = -100.0 / max(closest_neighbor_dist, 0.1)
            
        # 3. Obstacle Potential
        min_lidar = float(np.min(lidar_distances)) if len(lidar_distances) > 0 else np.inf
        
        phi_obstacle = 0.0
        if min_lidar < np.inf:
            phi_obstacle = -100.0 / max(min_lidar, 0.1)
            
        # Total PBRS Potential
        phi_now = phi_dist + phi_heading + phi_swarm + phi_obstacle
        
        if self.prev_phi is None:
            self.prev_phi = phi_now
            
        # 4. Apply PBRS Equation
        r_step = -1.0
        reward = phi_now - self.prev_phi + r_step
        self.prev_phi = phi_now
        
        # 5. Sparse Terminal Penalties / Rewards
        terminated = False
        
        if dist < self.GOAL_RADIUS:
            reward += 1000.0
            self.last_event = "GOAL"
            terminated = True
        elif min_lidar < self.COLLISION_RADIUS:
            reward -= 1000.0
            self.last_event = "COLLISION"
            terminated = True
        elif agent_collision:
            reward -= 1000.0
            self.last_event = "COLLISION"
            terminated = True
        else:
            self.last_event = None
            
        # Logging metrics
        self.last_metrics = {
            "dist_to_goal": dist,
            "heading_error_deg": heading_error_deg,
            "min_lidar": min_lidar,
            "closest_neighbor": closest_neighbor_dist if closest_neighbor_dist < np.inf else 0.0,
            "phi_dist": float(phi_dist),
            "phi_heading": float(phi_heading),
            "phi_swarm": float(phi_swarm),
            "phi_obstacle": float(phi_obstacle),
            "pbrs_shaping": float(reward - r_step),
            "r_total": float(reward),
        }
            
        return float(reward), terminated, self.last_event


# ─────────────────────────────────────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────────────────────────────────────
class MultiAgentNavEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "MultiAgentNav_v0"}
    
    def __init__(
        self,
        num_agents              = 4,
        num_auvs                = 0,
        max_steps               = 2000,
        shared_goal             = False,
        num_moving_obstacles    = 2,
        max_auv_entities        = 8,
        max_static_obstacles    = 8,
        max_moving_obstacles    = 8,
        static_obstacle_keep_prob  = 1.0,
        moving_obstacle_speed_scale= 1.0,
        map_size                = 1500.0,
        lidar_range             = 50.0,
        num_lidar_beams         = 64,
        max_static_obstacles_spawned = 40,
        show_viewport           = False,
        reward_kwargs           = None,
    ):
        super().__init__()

        self.num_agents                  = num_agents
        self.num_auvs                    = num_auvs
        self.possible_agents             = [f"vessel{i}" for i in range(num_agents)] + [f"auv{i}" for i in range(num_auvs)]
        self.agents                      = self.possible_agents[:]
        self.max_steps                   = max_steps
        self.shared_goal                 = shared_goal
        self.num_moving_obstacles        = num_moving_obstacles
        self.max_auv_entities            = max_auv_entities
        self.max_static_obstacles_cap    = max_static_obstacles
        self.max_moving_obstacles_cap    = max_moving_obstacles
        self.static_obstacle_keep_prob   = float(np.clip(static_obstacle_keep_prob, 0.0, 1.0))
        self.moving_obstacle_speed_scale = float(np.clip(moving_obstacle_speed_scale, 0.0, 1.0))
        self.map_size                    = map_size
        self.lidar_range                 = lidar_range
        self.num_lidar_beams             = num_lidar_beams
        self.max_static_obstacles_spawned= max_static_obstacles_spawned
        self.step_count                  = 0
        self.episode_count               = 0

        # Curriculum-controlled parameters
        self.goal_dist_max   = 80.0
        self.spawn_box_size  = 30.0
        self.cluster_prob    = 0.0
        self.cluster_radius  = 30.0

        # Physics
        self.dt = 0.1
        self._vessels = {}
        for aid in self.possible_agents:
            if "auv" in aid:
                from envs.auv_dynamics import AUVDynamics
                self._vessels[aid] = AUVDynamics(dt=self.dt)
            else:
                self._vessels[aid] = USVDynamics(dt=self.dt)
                
        self._moving_obs_dynamics = [
            USVDynamics(dt=self.dt) for _ in range(num_moving_obstacles)
        ]

        # Reward calculators
        self.reward_calculators = {aid: NavReward(map_size=self.map_size) for aid in self.possible_agents}

        # Perception (Kalman Filter smoothing)
        self.perception = PerceptionModule(dt=self.dt, map_size=map_size)

        # Obstacle storage
        self.static_obstacles  = []        # list of {"pos": [x,y,0], "radius": r}
        self.moving_obstacles  = []        # list of {"pos": [x,y,0], "vel": v, ...}
        self._moving_obs_goals = []        # patrol waypoints
        self._moving_obs_back  = []        # direction flag (forward/backward)

        # Previous actions for observation
        self._prev_actions = {aid: np.zeros(2, dtype=np.float32) for aid in self.possible_agents}
        self._agent_terminated = {aid: False for aid in self.possible_agents}

        self.goals  = {aid: np.zeros(3, dtype=np.float32) for aid in self.possible_agents}
        self.logger = EpisodeLogger()

        # Pre-computed LiDAR angles (normalised)
        lidar_angles_deg = np.arange(num_lidar_beams) * (360.0 / num_lidar_beams)
        self._lidar_angles_rad_norm = (np.deg2rad(lidar_angles_deg) - np.pi) / np.pi

        # Observation layout
        self.ego_dim          = 7
        self.goal_dim         = 2
        self.entity_feature_dim = 5         # [active, rx, ry, rvx, rvy]
        self.lidar_dim        = num_lidar_beams * 2   # [dist, angle] × beams
        self._build_obs_layout()

        self.obs_buffers = {
            aid: np.zeros(self.obs_dim, dtype=np.float32)
            for aid in self.possible_agents
        }

        # Gym spaces
        self.single_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_spaces = {aid: self.single_obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: self.single_action_space for aid in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ─── Observation layout ───────────────────────────────────────────────────
    def _build_obs_layout(self):
        start = 0
        self.obs_layout = {}

        def add_vector(name, dim):
            nonlocal start
            self.obs_layout[name] = {"start": start, "dim": dim}
            start += dim

        def add_entities(name, count):
            nonlocal start
            dim = count * self.entity_feature_dim
            self.obs_layout[name] = {
                "start": start, "dim": dim,
                "count": count, "feature_dim": self.entity_feature_dim,
            }
            start += dim

        add_vector("ego",             self.ego_dim)
        add_vector("goal",            self.goal_dim)
        add_entities("auv_entities",  self.max_auv_entities)
        add_entities("static_obstacles", self.max_static_obstacles_cap)
        add_entities("moving_obstacles", self.max_moving_obstacles_cap)
        add_vector("lidar",           self.lidar_dim)
        add_vector("global_state",    self.max_auv_entities * 5)

        self.obs_dim = start
        self.obs_layout["obs_dim"] = self.obs_dim

    def _slice(self, name):
        spec = self.obs_layout[name]
        return slice(spec["start"], spec["start"] + spec["dim"])

    # ─── Curriculum ──────────────────────────────────────────────────────────
    def configure_curriculum(
        self,
        static_obstacle_keep_prob=None,
        moving_obstacle_speed_scale=None,
        goal_dist_max=None,
        spawn_box_size=None,
        cluster_prob=None,
        cluster_radius=None,
    ):
        if static_obstacle_keep_prob is not None:
            self.static_obstacle_keep_prob = float(np.clip(static_obstacle_keep_prob, 0.0, 1.0))
        if moving_obstacle_speed_scale is not None:
            self.moving_obstacle_speed_scale = float(np.clip(moving_obstacle_speed_scale, 0.0, 1.0))
        if goal_dist_max is not None:
            self.goal_dist_max = float(np.clip(goal_dist_max, 20.0, self.map_size * 0.9))
        if spawn_box_size is not None:
            self.spawn_box_size = float(np.clip(spawn_box_size, 5.0, 50.0))
        if cluster_prob is not None:
            self.cluster_prob = float(np.clip(cluster_prob, 0.0, 1.0))
        if cluster_radius is not None:
            self.cluster_radius = float(np.clip(cluster_radius, 5.0, 50.0))

    # ─── Reset ───────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.step_count    = 0
        self.episode_count += 1
        self._agent_terminated = {aid: False for aid in self.possible_agents}

        for aid in self.possible_agents:
            self._prev_actions[aid] = np.zeros(2, dtype=np.float32)
            self.reward_calculators[aid].reset()

        self.perception.reset()
        self.logger.reset(self.possible_agents)

        # Sample goals
        sampled_goals = self._sample_goals()
        for aid in self.possible_agents:
            self.goals[aid] = sampled_goals[aid]

        # Spawn static obstacles
        self._spawn_static_obstacles()

        # Spawn moving obstacles
        self._spawn_moving_obstacles()

        # Place agents in a ring / random within spawn box
        half = self.spawn_box_size / 2.0
        min_sep = max(4.0, self.spawn_box_size * 0.2)
        placed = []
        for aid in self.possible_agents:
            for _ in range(200):
                cand = np.random.uniform(-half, half, size=2)
                if all(np.linalg.norm(cand - c) > min_sep for c in placed):
                    break
            else:
                angle = 2 * np.pi * len(placed) / max(self.num_agents, 1)
                cand = np.array([half * 0.8 * math.cos(angle),
                                 half * 0.8 * math.sin(angle)])
            placed.append(cand)
            yaw = np.random.uniform(-np.pi, np.pi)
            if "auv" in aid:
                depth = np.random.uniform(-20.0, -5.0)
                self._vessels[aid].reset(initial_eta=[float(cand[0]), float(cand[1]), depth, 0.0, 0.0, float(yaw)])
            else:
                self._vessels[aid].reset(initial_eta=[float(cand[0]), float(cand[1]), float(yaw)])

        all_states = self._all_agent_states()
        self.perception.update_entities(all_states)

        obs_dict  = {aid: self._get_obs(aid, all_states) for aid in self.agents}
        info_dict = {aid: self._get_info(aid) for aid in self.agents}

        return obs_dict, info_dict

    # ─── Step ────────────────────────────────────────────────────────────────
    def step(self, action_dict):
        self.step_count    += 1
        truncated_all       = self.step_count >= self.max_steps

        # 1. Apply actions to agents
        for aid, action in action_dict.items():
            if self._agent_terminated.get(aid, False):
                continue
                
            if "auv" in aid:
                # 4D Action space for AUV: [surge, pitch, yaw, heave]
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                throttle = float(np.clip(action[0], -1.0, 1.0)) if action.size > 0 else 0.0
                pitch_st = float(np.clip(action[1], -1.0, 1.0)) if action.size > 1 else 0.0
                yaw_st   = float(np.clip(action[2], -1.0, 1.0)) if action.size > 2 else 0.0
                heave    = float(np.clip(action[3], -1.0, 1.0)) if action.size > 3 else 0.0
                
                tau_X = throttle * 150.0
                tau_Z = heave * 50.0
                tau_M = pitch_st * 30.0
                tau_N = yaw_st * 30.0
                tau = [tau_X, 0.0, tau_Z, 0.0, tau_M, tau_N]
                
                self._vessels[aid].step(tau)
                self._prev_actions[aid] = np.array([throttle, yaw_st], dtype=np.float32) # For observation backwards compatibility
                
            else:
                throttle, steering = self._decode_action(action)
                tau_u, tau_r = self._action_to_fossen(throttle, steering)
                tau = [tau_u, 0.0, tau_r]
                self._vessels[aid].step(tau)
                self._prev_actions[aid] = np.array([throttle, steering], dtype=np.float32)

        # 2. Step moving obstacles
        self._step_moving_obstacles()

        # 3. Update perception
        all_states = self._all_agent_states()
        entity_states = dict(all_states)
        for i, mob in enumerate(self.moving_obstacles):
            entity_states[f"moving_obstacle{i}"] = {
                "pos": mob["pos"].copy(),
                "vel": mob["vel"].copy(),
                "yaw": mob["yaw"],
                "class": "moving_obstacle",
            }
        self.perception.update_entities(entity_states)

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = {}, {}, {}, {}, {}

        for aid in self.agents:
            vessel = self._vessels[aid]
            
            if "auv" in aid:
                pos = np.array([vessel.eta[0], vessel.eta[1], vessel.eta[2]], dtype=np.float32)
                yaw = float(vessel.eta[5])
                lidar_dists = np.full(self.num_lidar_beams, self.lidar_range, dtype=np.float32) # No lidar for AUV
            else:
                pos = np.array([vessel.eta[0], vessel.eta[1], 0.0], dtype=np.float32)
                yaw = float(vessel.eta[2])
                lidar_dists = self._compute_lidar(pos, yaw)
                
            swarm_ctx   = self.perception.extract_entity_context(
                aid, target_classes=("auv",)
            )

            reward, terminated, event = self.reward_calculators[aid].compute(
                pos=pos,
                yaw=yaw,
                goal_pos=self.goals[aid],
                lidar_distances=lidar_dists,
                swarm_context=swarm_ctx,
            )

            if terminated:
                self._agent_terminated[aid] = True

            obs_dict[aid]  = self._get_obs(aid, all_states)
            rew_dict[aid]  = reward
            term_dict[aid] = terminated
            trunc_dict[aid]= truncated_all
            info_dict[aid] = self._get_info(aid)
            info_dict[aid]["event"] = event or ""
            info_dict[aid]["reward_metrics"] = dict(
                self.reward_calculators[aid].last_metrics
            )

        self.logger.step(rew_dict)

        # Log terminal events
        any_term = any(term_dict.values())
        if any_term or truncated_all:
            for aid in self.possible_agents:
                if not truncated_all and not term_dict.get(aid, False):
                    continue
                if info_dict[aid].get("event") == "ALREADY_TERMINATED":
                    continue
                vessel    = self._vessels[aid]
                pos2d     = np.array([vessel.eta[0], vessel.eta[1]])
                dist      = float(np.linalg.norm(self.goals[aid][:2] - pos2d))
                lidar_d   = self._compute_lidar(
                    np.array([pos2d[0], pos2d[1], 0.0]), vessel.eta[2]
                )
                min_lidar = float(np.min(lidar_d))
                ctx       = self.perception.extract_entity_context(aid, target_classes=("auv",))
                if ctx is not None and len(ctx):
                    active = ctx[ctx[:, 0] > 0.5]
                    min_nbr = float(np.min(np.linalg.norm(
                        active[:, 1:3] * self.map_size, axis=1
                    ))) if len(active) else None
                else:
                    min_nbr = None

                outcome = "TIMEOUT" if truncated_all else (
                    self.reward_calculators[aid].last_event or "TERMINATED"
                )
                self.logger.log_termination(
                    aid, outcome, dist, min_lidar, min_nbr, self.step_count
                )

        # Aggregate info fields that the wrapper expects
        dist_to_goal = np.array(
            [info_dict[aid].get("dist_to_goal", 0.0) for aid in self.possible_agents],
            dtype=np.float32,
        )
        info_dict["dist_to_goal"] = dist_to_goal

        metrics_list = [
            info_dict[aid].get("reward_metrics", {})
            for aid in self.possible_agents
            if info_dict.get(aid, {}).get("reward_metrics")
        ]
        if metrics_list:
            keys = metrics_list[0].keys()
            avg_metrics = {}
            for k in keys:
                vals = [m.get(k, 0.0) for m in metrics_list if m.get(k) is not None]
                avg_metrics[k] = sum(vals) / len(vals) if vals else 0.0
            info_dict["reward_metrics"] = avg_metrics

        # PettingZoo API Requirement: Remove dead agents from self.agents
        self.agents = [aid for aid in self.agents if not term_dict[aid] and not trunc_dict[aid]]

        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    # ─── Observation Construction ─────────────────────────────────────────────
    def _get_obs(self, aid, all_states):
        vessel = self._vessels[aid]
        if "auv" in aid:
            pos = np.array([vessel.eta[0], vessel.eta[1], vessel.eta[2]], dtype=np.float32)
            yaw = float(vessel.eta[5])
            nu  = vessel.nu          # [u, v, w, p, q, r]
            yaw_rate = nu[5]
        else:
            pos = np.array([vessel.eta[0], vessel.eta[1], 0.0], dtype=np.float32)
            yaw = float(vessel.eta[2])
            nu  = vessel.nu          # [surge, sway, yaw_rate]
            yaw_rate = nu[2]

        prev   = self._prev_actions.get(aid, np.zeros(2, dtype=np.float32))
        throttle_cmd = float(prev[0])
        steering_cmd = float(prev[1])

        goal_pos    = self.goals[aid]
        goal_yaw    = math.atan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])

        # ── Ego (7) ───────────────────────────────────────────────────────────
        self.obs_buffers[aid][self._slice("ego")] = [
            math.sin(yaw), math.cos(yaw),
            math.sin(goal_yaw), math.cos(goal_yaw),
            throttle_cmd, steering_cmd,
            float(np.clip(yaw_rate, -1.0, 1.0)),
        ]

        # ── Goal (2) ─────────────────────────────────────────────────────────
        dist_to_goal  = float(np.linalg.norm(goal_pos[:2] - pos[:2]))
        norm_dist     = float(np.clip(dist_to_goal / (self.map_size * 0.5), 0.0, 1.0))
        heading_cos   = float(math.cos(yaw - goal_yaw))
        self.obs_buffers[aid][self._slice("goal")] = [norm_dist, heading_cos]

        # ── Entity contexts ───────────────────────────────────────────────────
        auv_ctx     = self.perception.extract_entity_context(aid, target_classes=("auv",))
        moving_ctx  = self.perception.extract_entity_context(aid, target_classes=("moving_obstacle",))
        static_ctx  = self._static_obstacle_context(aid, all_states)

        auv_pad     = self._pad_entities(auv_ctx,    self.max_auv_entities)
        static_pad  = self._pad_entities(static_ctx, self.max_static_obstacles_cap)
        moving_pad  = self._pad_entities(moving_ctx, self.max_moving_obstacles_cap)

        self.obs_buffers[aid][self._slice("auv_entities")]    = auv_pad.reshape(-1)
        self.obs_buffers[aid][self._slice("static_obstacles")]= static_pad.reshape(-1)
        self.obs_buffers[aid][self._slice("moving_obstacles")]= moving_pad.reshape(-1)

        # ── LiDAR (128) ──────────────────────────────────────────────────────
        if "auv" in aid:
            # AUVs do not use LiDAR currently - pass zeros to match network dimension
            lidar_feats = np.zeros(self.lidar_dim, dtype=np.float32)
        else:
            lidar_dists = self._compute_lidar(pos, yaw)
            lidar_feats = np.stack([
                lidar_dists / self.lidar_range,
                self._lidar_angles_rad_norm,
            ], axis=1).astype(np.float32).flatten()
            
        self.obs_buffers[aid][self._slice("lidar")] = lidar_feats

        # ── Global state for CTDE (max_auv_entities × 5) ─────────────────────
        global_state = np.zeros(self.max_auv_entities * 5, dtype=np.float32)
        for i, a in enumerate(self.possible_agents):
            if i >= self.max_auv_entities:
                break
            if a in all_states:
                s = all_states[a]
                global_state[i*5 : i*5+2] = s["pos"][:2]
                global_state[i*5+2 : i*5+4] = s["vel"][:2]
                global_state[i*5+4] = s["yaw"]
        self.obs_buffers[aid][self._slice("global_state")] = global_state

        return self.obs_buffers[aid].copy()

    # ─── Entity helpers ───────────────────────────────────────────────────────
    def _all_agent_states(self):
        states = {}
        for aid in self.possible_agents:
            v = self._vessels[aid]
            if "auv" in aid:
                # Need to convert body velocities to earth velocities for AUV
                u, v_sway, w = v.nu[0], v.nu[1], v.nu[2]
                phi, theta, psi = v.eta[3], v.eta[4], v.eta[5]
                # Simplified 2D earth velocity for MARL compatibility
                c_psi, s_psi = math.cos(psi), math.sin(psi)
                vx = c_psi * u - s_psi * v_sway
                vy = s_psi * u + c_psi * v_sway
                
                states[aid] = {
                    "pos": np.array([v.eta[0], v.eta[1], v.eta[2]], dtype=np.float32),
                    "vel": np.array([vx, vy, 0.0], dtype=np.float32),
                    "yaw": float(psi),
                    "class": "auv",
                }
            else:
                states[aid] = {
                    "pos": np.array([v.eta[0], v.eta[1], 0.0], dtype=np.float32),
                    "vel": v.get_earth_velocity().astype(np.float32),
                    "yaw": float(v.eta[2]),
                    "class": "auv", # using 'auv' class name to trigger entity context gathering
                }
        return states

    def _static_obstacle_context(self, ego_id, all_states):
        if ego_id not in all_states:
            return np.zeros((0, self.entity_feature_dim), dtype=np.float32)
        ego_pos = all_states[ego_id]["pos"][:2]
        ego_yaw = float(all_states[ego_id]["yaw"])
        features = []
        for obs in self.static_obstacles:
            rel = obs["pos"][:2] - ego_pos
            rel = _rot_body(rel, ego_yaw)
            features.append([
                1.0,
                float(np.clip(rel[0] / self.map_size, -1.0, 1.0)),
                float(np.clip(rel[1] / self.map_size, -1.0, 1.0)),
                0.0, 0.0,
            ])
        if not features:
            return np.zeros((0, self.entity_feature_dim), dtype=np.float32)
        return np.array(features, dtype=np.float32)

    def _pad_entities(self, context, max_entities):
        padded = np.zeros((max_entities, self.entity_feature_dim), dtype=np.float32)
        if context is None or len(context) == 0 or max_entities == 0:
            return padded
        context = np.asarray(context, dtype=np.float32)
        active  = context[context[:, 0] > 0.5]
        if not len(active):
            return padded
        order    = np.argsort(np.linalg.norm(active[:, 1:3], axis=1))
        selected = active[order[:max_entities]]
        padded[:len(selected)] = selected
        return padded

    # ─── LiDAR ───────────────────────────────────────────────────────────────
    def _compute_lidar(self, pos, yaw):
        # Compile obstacles into a fast Numba-friendly NumPy array
        num_static = len(self.static_obstacles)
        num_moving = len(self.moving_obstacles)
        obs_arr = np.zeros((num_static + num_moving, 3), dtype=np.float32)
        
        for i, obs in enumerate(self.static_obstacles):
            obs_arr[i, 0] = obs["pos"][0]
            obs_arr[i, 1] = obs["pos"][1]
            obs_arr[i, 2] = obs["radius"]
            
        for i, obs in enumerate(self.moving_obstacles):
            obs_arr[num_static + i, 0] = obs["pos"][0]
            obs_arr[num_static + i, 1] = obs["pos"][1]
            obs_arr[num_static + i, 2] = obs.get("radius", 2.5)
            
        from envs.numba_utils import numba_synthetic_lidar
        return numba_synthetic_lidar(
            ego_pos=pos,
            ego_yaw=yaw,
            obstacles_array=obs_arr,
            lidar_range=self.lidar_range,
            num_beams=self.num_lidar_beams,
        )

    # ─── Action helpers ───────────────────────────────────────────────────────
    def _decode_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 2:
            return 0.0, 0.0
        return float(np.clip(action[0], -1.0, 1.0)), float(np.clip(action[1], -1.0, 1.0))

    def _action_to_fossen(self, throttle_norm, steering_norm):
        """Map normalised [-1,1] action to Fossen thrust commands."""
        tau_u = throttle_norm * 80.0    # surge force (N)
        tau_r = steering_norm * 15.0    # yaw torque (N·m)
        return tau_u, tau_r

    # ─── Moving obstacles ─────────────────────────────────────────────────────
    def _spawn_moving_obstacles(self):
        self.moving_obstacles  = []
        self._moving_obs_goals = []
        self._moving_obs_back  = []
        for i in range(self.num_moving_obstacles):
            angle = 2 * np.pi * i / max(self.num_moving_obstacles, 1)
            r     = self.map_size * 0.35
            start = np.array([r * math.cos(angle), r * math.sin(angle), 0.0], dtype=np.float32)
            goal  = -start.copy()
            yaw   = math.atan2(goal[1] - start[1], goal[0] - start[0])
            self._moving_obs_dynamics[i].reset(
                initial_eta=[float(start[0]), float(start[1]), float(yaw)]
            )
            self.moving_obstacles.append({
                "pos":    start.copy(),
                "vel":    np.zeros(2, dtype=np.float32),
                "yaw":    yaw,
                "radius": 2.5,
            })
            self._moving_obs_goals.append(goal.copy())
            self._moving_obs_back.append(False)

    def _step_moving_obstacles(self):
        for i, dyn in enumerate(self._moving_obs_dynamics):
            if i >= self.num_moving_obstacles:
                break
            pos    = np.array([dyn.eta[0], dyn.eta[1], 0.0], dtype=np.float32)
            target = self._moving_obs_goals[i][:2]
            dist   = float(np.linalg.norm(target - pos[:2]))
            if dist < 8.0:
                # Swap waypoint
                self._moving_obs_back[i] = not self._moving_obs_back[i]
                r = self.map_size * 0.35
                angle = 2 * np.pi * i / max(self.num_moving_obstacles, 1)
                if self._moving_obs_back[i]:
                    self._moving_obs_goals[i] = np.array([
                        r * math.cos(angle), r * math.sin(angle), 0.0
                    ], dtype=np.float32)
                else:
                    self._moving_obs_goals[i] = np.array([
                        -r * math.cos(angle), -r * math.sin(angle), 0.0
                    ], dtype=np.float32)
                target = self._moving_obs_goals[i][:2]

            yaw     = dyn.eta[2]
            des_yaw = math.atan2(target[1] - dyn.eta[1], target[0] - dyn.eta[0])
            hd_err  = math.atan2(math.sin(des_yaw - yaw), math.cos(des_yaw - yaw))
            tau_u   = 30.0 * self.moving_obstacle_speed_scale
            tau_r   = 8.0  * hd_err * self.moving_obstacle_speed_scale
            dyn.step([tau_u, 0.0, tau_r])

            v = dyn.get_earth_velocity()
            self.moving_obstacles[i] = {
                "pos":    np.array([dyn.eta[0], dyn.eta[1], 0.0], dtype=np.float32),
                "vel":    v.astype(np.float32),
                "yaw":    float(dyn.eta[2]),
                "radius": 2.5,
            }

    # ─── Static obstacles ─────────────────────────────────────────────────────
    def _spawn_static_obstacles(self):
        self.static_obstacles = []
        rng = self.np_random
        # Wider grid = fewer obstacles, easier navigation
        grid_spacing = 20.0
        half         = self.map_size / 2.0
        xs = np.arange(-half + grid_spacing, half, grid_spacing)
        ys = np.arange(-half + grid_spacing, half, grid_spacing)
        pts = [(float(x), float(y)) for x in xs for y in ys]
        rng.shuffle(pts)

        # Keep agents' spawn zone AND a corridor to goals free
        safe_r = max(self.spawn_box_size * 1.5 + 5.0, 25.0)

        for (x, y) in pts:
            if (self.max_static_obstacles_spawned is not None and
                    len(self.static_obstacles) >= self.max_static_obstacles_spawned):
                break
            if rng.random() > self.static_obstacle_keep_prob:
                continue
            if math.sqrt(x*x + y*y) < safe_r:
                continue
            # Check goal clearance
            too_close = any(
                float(np.linalg.norm(self.goals[aid][:2] - np.array([x, y]))) < 8.0
                for aid in self.possible_agents
            )
            if too_close:
                continue
            ox   = x + float(rng.uniform(-2.0, 2.0))
            oy   = y + float(rng.uniform(-2.0, 2.0))
            robs = float(rng.uniform(1.0, 3.0))
            self.static_obstacles.append({
                "pos":    np.array([ox, oy, 0.0], dtype=np.float32),
                "radius": robs,
            })

    # ─── Goal sampling ────────────────────────────────────────────────────────
    def _sample_goals(self):
        if self.shared_goal:
            g = self._sample_one_goal()
            return {aid: g.copy() for aid in self.possible_agents}

        goals, accepted = {}, []
        is_clustered = self.np_random.uniform() < self.cluster_prob

        if is_clustered:
            center = self._sample_one_goal()
            for aid in self.possible_agents:
                for _ in range(100):
                    off = self.np_random.uniform(-self.cluster_radius, self.cluster_radius, 2)
                    cand = np.array([center[0]+off[0], center[1]+off[1], 0.0], dtype=np.float32)
                    if all(np.linalg.norm(cand[:2]-o[:2]) >= 6.0 for o in accepted):
                        break
                goals[aid] = cand
                accepted.append(cand)
        else:
            min_sep = 30.0
            for aid in self.possible_agents:
                for _ in range(100):
                    cand = self._sample_one_goal()
                    if all(np.linalg.norm(cand[:2]-o[:2]) >= min_sep for o in accepted):
                        break
                goals[aid] = cand
                accepted.append(cand)

        return goals

    def _sample_one_goal(self):
        rng  = self.np_random
        dist = rng.uniform(max(20.0, self.goal_dist_max * 0.6), self.goal_dist_max)
        ang  = rng.uniform(0, 2 * np.pi)
        return np.array([dist * math.cos(ang), dist * math.sin(ang), 0.0], dtype=np.float32)

    # ─── Info ─────────────────────────────────────────────────────────────────
    def _get_info(self, aid):
        v = self._vessels[aid]
        pos2d = np.array([v.eta[0], v.eta[1]])
        vel   = v.get_earth_velocity()
        return {
            "dist_to_goal":  float(np.linalg.norm(self.goals[aid][:2] - pos2d)),
            "speed":         float(np.linalg.norm(vel)),
            "throttle_cmd":  float(self._prev_actions[aid][0]),
            "steering_cmd":  float(self._prev_actions[aid][1]),
            "step_count":    self.step_count,
            "pos":           np.array([v.eta[0], v.eta[1], 0.0], dtype=np.float32),
        }

    def close(self):
        pass


# ─── Utility ──────────────────────────────────────────────────────────────────
def _rot_body(vec_xy, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([c * vec_xy[0] + s * vec_xy[1],
                     -s * vec_xy[0] + c * vec_xy[1]], dtype=np.float32)


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    env = MultiAgentNavEnv(num_agents=4, num_moving_obstacles=2)
    obs, info = env.reset(seed=42)
    print(f"Obs dim: {env.obs_dim}")
    print(f"Obs layout keys: {list(env.obs_layout.keys())}")
    for _ in range(10):
        actions = {aid: env.single_action_space.sample() for aid in env.agent_ids}
        obs, rew, term, trunc, info = env.step(actions)
    print("Sanity check passed ✓")
    env.close()
