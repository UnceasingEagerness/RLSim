import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict

# Import our JAX physics components
from env.jax_dynamics import USVState, USVParams, rk4_step
from env.jax_lidar import jax_synthetic_lidar

@struct.dataclass
class EnvState:
    """The complete state of the environment."""
    usv_state: USVState     # Batched [N]
    goal_pos: jnp.ndarray   # [N, 2]
    obstacles: jnp.ndarray  # [M, 3] matrix of [x, y, radius]
    step_count: int
    time: float
    prev_dist: jnp.ndarray  # [N] Previous distance to goal for progress reward
    history_obs: jnp.ndarray # [N, T, 92] Historical buffer for LSTM Frame Stacking
    # ── Encirclement fields (used when params.encircle_mode=True; zero otherwise) ─
    target_pos: jnp.ndarray     # [2]  ring center  (= goal in nav mode)
    target_vel: jnp.ndarray     # [2]  target velocity (used for moving_target)
    prev_ring_dist: jnp.ndarray # [N]  |dist_to_target - d_E|  (unused in nav mode)
    prev_phi_cap: jnp.ndarray   # []   previous uniformity potential

@struct.dataclass
class EnvParams:
    """Static configuration for the environment."""
    num_agents: int = 5
    seq_len: int = 10           # Frame Stacking for Spatio-Temporal LSTM
    max_steps: int = 2000       # Shorter episodes = faster replay buffer diversity
    map_size: float = 3000.0    # 3km x 3km map
    num_obstacles: int = struct.field(pytree_node=False, default=600) # Ultra-dense minefield
    lidar_range: float = 50.0
    num_lidar_beams: int = struct.field(pytree_node=False, default=64)
    goal_radius: float = 15.0
    collision_radius: float = 4.0
    usv_params: USVParams = USVParams()
    map_bank_size: int = struct.field(pytree_node=False, default=1000)
    goals_bank: jnp.ndarray = struct.field(default_factory=lambda: jnp.empty((0,)))
    obstacles_bank: jnp.ndarray = struct.field(default_factory=lambda: jnp.empty((0,)))
    # ── Encirclement mode (off by default — fully backwards compatible) ────────
    encircle_mode: bool = struct.field(pytree_node=False, default=False)
    encircle_radius: float = 80.0        # d_E: orbit ring radius (m)
    orbit_lead_angle: float = 0.4        # radians CCW ahead on ring (carrot)
    formation_reward_scale: float = 1.0  # l6 from paper Eq.14 — punishes clustering
    orbit_reward_scale: float = 2.0      # reward CCW tangential velocity on ring
    # Curriculum Progression
    global_progress: float = 0.0
    moving_target: bool = struct.field(pytree_node=False, default=False) # Pursuit-evasion
    target_speed: float = 2.0            # meters per second for moving target

class JaxUSVEnv:
    """
    A pure JAX implementation of the Multi-Agent Nav Environment.
    Simulates N agents interacting in the same physical space.

    Supports two operating modes via params.encircle_mode:
      False (default) — standard navigation to independent goals
      True            — encirclement task: agents orbit a static target at radius d_E
    """
    def __init__(self):
        self.default_params = EnvParams()

    # _compute_phi removed. PBRS scrapped due to unbounded reward variance.
    # Using simple progress + heading reward instead.

    @staticmethod
    def _compute_lead_ring_points(pos: jnp.ndarray,
                                   target_pos: jnp.ndarray,
                                   encircle_radius: float,
                                   orbit_lead_angle: float) -> jnp.ndarray:
        """
        Encirclement helper: compute the 'lead' point on the ring for each agent.

        The lead point is orbit_lead_angle radians CCW ahead of the agent's current
        angular position around target_pos. Acts as a moving carrot that induces
        continuous CCW orbiting once agents reach the ring radius.

        Args:
            pos:              [N, 2]
            target_pos:       [2]
            encircle_radius:  float
            orbit_lead_angle: float (radians CCW)
        Returns:
            lead_points:      [N, 2]
        """
        diff          = pos - target_pos[None, :]                     # [N, 2]
        current_angle = jnp.arctan2(diff[:, 1], diff[:, 0])          # [N]
        lead_angle    = current_angle + orbit_lead_angle              # [N]
        lead_points   = target_pos[None, :] + encircle_radius * jnp.stack(
            [jnp.cos(lead_angle), jnp.sin(lead_angle)], axis=1
        )                                                             # [N, 2]
        return lead_points

    @staticmethod
    def generate_map_bank(key: jax.random.PRNGKey, num_agents: int, num_obstacles: int, map_size: float, map_bank_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Pre-computes a bank of maps (goals and obstacles) using Rejection Sampling."""
        def generate_single_map(k):
            N = num_agents
            hm = map_size / 2.0
            k_goal, k_obs = jax.random.split(k)
            
            # --- Place Goals ---
            def place_goals(k, n, hm, min_radius=300.0, max_radius=600.0):
                def body_fn(i, val):
                    k, g_xy = val
                    def try_fn(j, try_val):
                        k, candidate, found = try_val
                        k, sk = jax.random.split(k)
                        new_xy = jax.random.uniform(sk, shape=(2,), minval=-max_radius, maxval=max_radius)
                        dist = jnp.linalg.norm(new_xy)
                        valid = (dist >= min_radius) & (dist <= max_radius)
                        candidate = jnp.where(jnp.logical_and(~found, valid), new_xy, candidate)
                        found = jnp.logical_or(found, valid)
                        return k, candidate, found
                    k, final_candidate, _ = jax.lax.fori_loop(0, 500, try_fn, (k, jnp.zeros(2), False))
                    g_xy = g_xy.at[i].set(final_candidate)
                    return k, g_xy
                k, sk = jax.random.split(k)
                obs_xy = jnp.zeros((n, 2))
                _, final_goals = jax.lax.fori_loop(0, n, body_fn, (k, obs_xy))
                return final_goals
                
            goal_pos = place_goals(k_goal, N, hm)
            
            # --- Place Obstacles ---
            obs_r = jax.random.uniform(k_obs, shape=(num_obstacles, 1), minval=3.0, maxval=10.0)
            def place_obstacles(k, n, hm, goals, min_obs_dist=50.0, min_goal_dist=50.0, safe_spawn_radius=150.0):
                def body_fn(i, val):
                    k, obs_xy = val
                    def try_fn(j, try_val):
                        k, candidate, found = try_val
                        k, sk = jax.random.split(k)
                        new_xy = jax.random.uniform(sk, shape=(2,), minval=-hm, maxval=hm)
                        mask = jnp.arange(n) < i
                        obs_dists = jnp.linalg.norm(obs_xy - new_xy, axis=1)
                        valid_obs = jnp.all(jnp.logical_or(~mask, obs_dists >= min_obs_dist))
                        goal_dists = jnp.linalg.norm(goals - new_xy, axis=1)
                        valid_goals = jnp.all(goal_dists >= min_goal_dist)
                        spawn_dist = jnp.linalg.norm(new_xy)
                        valid_spawn = spawn_dist >= safe_spawn_radius
                        valid = valid_obs & valid_goals & valid_spawn
                        candidate = jnp.where(jnp.logical_and(~found, valid), new_xy, candidate)
                        found = jnp.logical_or(found, valid)
                        return k, candidate, found
                    k, final_candidate, _ = jax.lax.fori_loop(0, 200, try_fn, (k, jnp.zeros(2), False))
                    obs_xy = obs_xy.at[i].set(final_candidate)
                    return k, obs_xy
                    
                def try_first(j, try_val):
                    k, candidate, found = try_val
                    k, sk = jax.random.split(k)
                    new_xy = jax.random.uniform(sk, shape=(2,), minval=-hm, maxval=hm)
                    goal_dists = jnp.linalg.norm(goals - new_xy, axis=1)
                    valid_goals = jnp.all(goal_dists >= min_goal_dist)
                    spawn_dist = jnp.linalg.norm(new_xy)
                    valid_spawn = spawn_dist >= safe_spawn_radius
                    valid = valid_goals & valid_spawn
                    candidate = jnp.where(jnp.logical_and(~found, valid), new_xy, candidate)
                    found = jnp.logical_or(found, valid)
                    return k, candidate, found

                k, initial_pt, _ = jax.lax.fori_loop(0, 200, try_first, (k, jnp.zeros(2), False))
                obs_xy = jnp.zeros((n, 2)).at[0].set(initial_pt)
                _, final_obs = jax.lax.fori_loop(1, n, body_fn, (k, obs_xy))
                return final_obs
                
            k_obs, key_place = jax.random.split(k_obs)
            obs_xy = place_obstacles(key_place, num_obstacles, hm, goal_pos)
            obstacles = jnp.concatenate([obs_xy, obs_r], axis=1)
            
            return goal_pos, obstacles

        # Vectorize across the bank size
        keys = jax.random.split(key, map_bank_size)
        vmap_generate = jax.vmap(generate_single_map)
        goals_bank, obstacles_bank = vmap_generate(keys)
        return goals_bank, obstacles_bank

    @staticmethod
    def reset(key: jax.random.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
        """Resets the environment with N agents using O(1) Map Bank lookup."""
        key_pos, key_idx = jax.random.split(key, 2)
        N = params.num_agents

        init_pos = jax.random.uniform(key_pos, shape=(N, 2), minval=-50.0, maxval=50.0)
        init_yaw = jax.random.uniform(key_pos, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)

        eta = jnp.concatenate([init_pos, init_yaw[:, None]], axis=1)
        nu  = jnp.zeros((N, 3))
        usv_state = USVState(eta=eta, nu=nu)

        # O(1) Map Bank Lookup
        idx = jax.random.randint(key_idx, shape=(), minval=0, maxval=params.map_bank_size)
        goal_pos = params.goals_bank[idx]
        obstacles = params.obstacles_bank[idx]

        init_dist = jnp.linalg.norm(goal_pos - init_pos, axis=1)

        # ── Encirclement: use slot-0 of goals_bank as ring center ─────────────────
        # In nav mode:     target_pos is unused (set to zeros)
        # In encircle mode: target_pos = goal_pos[0] (the ring centre)
        target_pos    = jnp.where(params.encircle_mode, goal_pos[0],   jnp.zeros(2))
        
        # Moving target logic
        key_tvel = jax.random.split(key_idx, 1)[0]
        target_angle = jax.random.uniform(key_tvel, minval=-jnp.pi, maxval=jnp.pi)
        base_vel = jnp.array([jnp.cos(target_angle) * params.target_speed, 
                              jnp.sin(target_angle) * params.target_speed])
        target_vel = jnp.where(params.moving_target, base_vel, jnp.zeros(2))
        
        init_ring_dist = jnp.where(params.encircle_mode,
                                   jnp.abs(jnp.linalg.norm(init_pos - target_pos[None,:], axis=1)
                                           - params.encircle_radius),
                                   jnp.zeros(params.num_agents))
        # OPTION B (paper-faithful): all agents navigate to the SAME target center.
        # The emergent spreading comes purely from r_formation (Paper Eq.14)
        # and orbiting from r_orbit — NOT from any geometric ring-point carrot.
        goal_pos_enc = jnp.broadcast_to(target_pos[None, :], (N, 2))
        goal_pos = jnp.where(params.encircle_mode, goal_pos_enc, goal_pos)

        state = EnvState(
            usv_state=usv_state,
            goal_pos=goal_pos,
            obstacles=obstacles,
            step_count=0,
            time=0.0,
            prev_dist=init_dist,
            history_obs=jnp.zeros((N, params.seq_len, 92)), # Placeholder
            target_pos=target_pos,
            target_vel=target_vel,
            prev_ring_dist=init_ring_dist,
            prev_phi_cap=jnp.array(0.0),
        )
        
        # Calculate the initial single-frame observation
        base_obs = JaxUSVEnv.get_base_obs(state, params) # [N, 92]
        
        # Replicate the initial frame backwards in time to fill the history buffer
        history_obs = jnp.repeat(jnp.expand_dims(base_obs, axis=1), params.seq_len, axis=1) # [N, 10, 92]
        state = state.replace(history_obs=history_obs)

        # Return the flat sequence
        obs = history_obs.reshape((N, -1))
        return obs, state

    @staticmethod
    def step(key: jax.random.PRNGKey, state: EnvState, action: jnp.ndarray, params: EnvParams) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Steps the dynamics forward for all N agents. action shape: [N, 2]"""

        N = params.num_agents

        throttle = jnp.clip(action[:, 0], -1.0, 1.0)
        steering = jnp.clip(action[:, 1], -1.0, 1.0)

        tau_u = throttle * 250.0
        tau_r = steering * 100.0
        tau = jnp.stack([tau_u, jnp.zeros(N), tau_r], axis=1)

        vmap_rk4 = jax.vmap(rk4_step, in_axes=(0, 0, None))
        new_usv_state = vmap_rk4(state.usv_state, tau, params.usv_params)

        pos = new_usv_state.eta[:, :2]  # [N, 2]
        yaw = new_usv_state.eta[:, 2]   # [N]

        # ── Termination Conditions ───────────────────────────────────────────────
        curr_dist = jnp.linalg.norm(state.goal_pos - pos, axis=1)  # [N]
        reached_goal = curr_dist < params.goal_radius

        vmap_lidar = jax.vmap(jax_synthetic_lidar, in_axes=(0, 0, None, None, None))
        lidar_dists = vmap_lidar(pos, yaw, state.obstacles, params.lidar_range, params.num_lidar_beams)  # [N, 64]
        min_dist_obs = jnp.min(lidar_dists, axis=1)
        collision_obs = min_dist_obs < 2.0

        pos_diff = pos[:, None, :] - pos[None, :, :]
        agent_dists = jnp.linalg.norm(pos_diff, axis=-1)
        agent_dists = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, agent_dists)
        min_agent_dist = jnp.min(agent_dists, axis=1)  # [N]
        collision_agent = min_agent_dist < params.collision_radius

        collision = collision_obs | collision_agent
        timeout = jnp.full((N,), state.step_count >= params.max_steps)
        done = reached_goal | timeout | collision

        # ── Simple, Stable Reward Function (No PBRS) ─────────────────────────────
        # 1. Progress: reward moving toward goal, penalise moving away
        progress = state.prev_dist - curr_dist          # +ve = closer, -ve = farther
        r_progress = progress * 5.0                     # strong dense gradient

        # 2. Adaptive heading alignment — scales with proximity but capped at 2.0
        #    Far away (500m): weight ≈ 1.0  → gentle nudge to face goal
        #    Close up  (20m): weight = 2.0  → strong precision docking signal
        #    Cap at 2.0 prevents yaw term from dominating progress and causing spinning
        goal_dir = jnp.arctan2(
            state.goal_pos[:, 1] - pos[:, 1],
            state.goal_pos[:, 0] - pos[:, 0]
        )
        heading_error = jnp.abs((goal_dir - yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        adaptive_yaw_weight = jnp.minimum(2.0, 500.0 / (curr_dist + 20.0))
        r_heading = jnp.cos(heading_error) * adaptive_yaw_weight

        # 3. Step penalty — keeps episodes short, discourages loitering
        r_step = -0.5
        
        # 4. Action penalty — punishes violent swerving (steering is in [-1, 1])
        r_action = -0.2 * (steering ** 2)

        reward = r_progress + r_heading + r_step + r_action

        # 5. Sparse terminals — override shaping entirely on terminal steps
        #    so the full ±500/200 signal always reaches the Q-function cleanly
        reward = jnp.where(reached_goal, 500.0,  reward)
        reward = jnp.where(collision,   -200.0,  reward)

        new_state = state.replace(
            usv_state=new_usv_state,
            step_count=state.step_count + 1,
            time=state.time + params.usv_params.dt,
            prev_dist=curr_dist
        )

        # Build the base info dict here so the encirclement block can append to it
        info = {
            "reached_goal":   reached_goal,
            "collision":      collision,
            "collision_obs":  collision_obs,
            "collision_agent": collision_agent,
            "timeout":        timeout,
            "dist_to_goal":   curr_dist
        }

        # ── Encirclement mode: override goal, reward, done ────────────────────────
        # This entire block is a compile-time branch (encircle_mode is a static field).
        # When encircle_mode=False every line below is dead code — zero runtime cost.
        if params.encircle_mode:
            nu = new_usv_state.nu  # [N, 3] body-frame velocities

            # Geometry
            diff_to_target = pos - state.target_pos[None, :]           # [N, 2]
            dist_to_target = jnp.linalg.norm(diff_to_target, axis=1)   # [N]
            curr_ring_dist = jnp.abs(dist_to_target - params.encircle_radius)  # [N]

            # --- Encirclement reward terms (Transformer MARL Paper §III-B) ---

            # (a) Target approach reward (PBRS)
            # NOTE: We use gamma=1.0 here instead of the paper's 0.99. 
            # If gamma < 1.0 and \Phi is negative, the agent receives a positive reward 
            # for staying still (D * (1 - gamma)). This causes the agents to run away 
            # to the corners of the map to farm infinite reward. gamma=1.0 fixes this perfectly.
            phi_dis_prev = -state.prev_ring_dist
            phi_dis_curr = -curr_ring_dist
            gamma_pbrs = 1.0
            r_enc_approach = gamma_pbrs * phi_dis_curr - phi_dis_prev

            # (b) Encirclement Uniformity Score (EUS)
            target_dir = jnp.arctan2(
                pos[:, 1] - state.target_pos[1],
                pos[:, 0] - state.target_pos[0]
            )
            sorted_angles = jnp.sort(target_dir) # [N]
            first_gap = (sorted_angles[0] + 2 * jnp.pi) - sorted_angles[-1]
            other_gaps = sorted_angles[1:] - sorted_angles[:-1]
            gaps = jnp.concatenate([jnp.array([first_gap]), other_gaps]) # [N]
            
            ideal_gap = 2 * jnp.pi / N
            norm_gaps = gaps / ideal_gap
            sigma_gaps = jnp.std(norm_gaps)
            
            curr_phi_cap = jnp.exp(-sigma_gaps)
            r_enc_dense = jnp.full((N,), curr_phi_cap - state.prev_phi_cap)

            # (c) Collision Avoidance Penalty (Continuous)
            d_coll = 6.0
            min_dist_any = jnp.minimum(min_agent_dist, min_dist_obs) # [N]
            r_collision = jnp.where(min_dist_any < d_coll, 
                                    -((d_coll - min_dist_any)**2), 
                                    0.0)

            # (d) Terminal Capture Reward
            on_ring   = curr_ring_dist < params.goal_radius
            encircled = jnp.all(on_ring)
            r_cap = jnp.where(encircled, 20.0, 0.0)

            # Combine components (Weighted sum)
            # --- Curriculum Reward Annealing ---
            p = params.global_progress
            
            def sigmoid(x, center, steepness=15.0):
                return 1.0 / (1.0 + jnp.exp(-steepness * (x - center)))

            # Progress drops smoothly (Cosine decay) 4.0 -> 0.5
            k_progress = 0.5 + 3.5 * 0.5 * (1.0 + jnp.cos(p * jnp.pi))
            
            # Safety drops slightly 5.0 -> 4.0
            k_safety = 5.0 - 1.0 * p
            
            # Formation rises early (around p=0.3) 0.0 -> 5.0
            # GATED: Only activate formation if a majority of agents are within 2.5 * encircle_radius
            gate_dist = 2.5 * params.encircle_radius
            is_near_target = jnp.mean(dist_to_target < gate_dist) > 0.5
            k_form = jnp.where(is_near_target, 5.0 * sigmoid(p, 0.3), 0.0)
            
            # Capture rises late (around p=0.7) 0.0 -> 8.0
            k_cap = 8.0 * sigmoid(p, 0.7)
            
            # Time penalty (Linear) -0.2 -> -1.0
            k_time = -0.2 - 0.8 * p

            total_enc_reward = (k_cap * r_cap + 
                                k_form * r_enc_dense + 
                                k_progress * r_enc_approach + 
                                k_safety * r_collision + 
                                k_time)
            
            # Global Team Reward (average across all agents)
            global_reward = jnp.mean(total_enc_reward)
            enc_reward = jnp.full((N,), global_reward)

            # Termination: all agents on ring simultaneously = success
            enc_done = encircled | timeout | collision

            # Override nav outputs with encirclement outputs.
            reward = enc_reward
            done   = enc_done
            
            # Update target position if moving
            new_target_pos = state.target_pos + state.target_vel * params.usv_params.dt
            new_target_pos = jnp.where(params.moving_target, new_target_pos, state.target_pos)
            new_goal_pos   = jnp.broadcast_to(new_target_pos[None, :], (N, 2))
            new_goal_pos   = jnp.where(params.encircle_mode, new_goal_pos, state.goal_pos)
            
            new_state = new_state.replace(
                goal_pos       = new_goal_pos,
                target_pos     = new_target_pos,
                target_vel     = state.target_vel,
                prev_ring_dist = curr_ring_dist,
                prev_phi_cap   = curr_phi_cap,
            )
            info["on_ring"]         = on_ring
            info["encircled"]       = encircled
            info["ring_dist"]       = curr_ring_dist
            info["k_progress"]      = jnp.full((N,), k_progress)
            info["k_form"]          = jnp.full((N,), k_form)
            info["k_cap"]           = jnp.full((N,), k_cap)
            info["r_enc_dense"]     = r_enc_dense
            info["curr_phi_cap"]    = curr_phi_cap
            info["gate_active"]     = jnp.full((N,), is_near_target)
            info["max_escape_gap"]  = jnp.full((N,), jnp.max(gaps))
            
            centroid = jnp.mean(pos, axis=0)
            radii = jnp.linalg.norm(pos - centroid, axis=1)
            info["radius_mean"]     = jnp.full((N,), jnp.mean(radii))
            info["radius_std"]      = jnp.full((N,), jnp.std(radii))
            info["encircle_ratio"]  = jnp.full((N,), jnp.mean(on_ring.astype(jnp.float32)))
        else:
            # Nav mode: carry forward unchanged (unused)
            new_state = new_state.replace(
                target_pos     = state.target_pos,
                target_vel     = state.target_vel,
                prev_ring_dist = state.prev_ring_dist,
                prev_phi_cap   = state.prev_phi_cap,
            )

        
        # Get the new single frame
        new_base_obs = JaxUSVEnv.get_base_obs(new_state, params) # [N, 92]
        
        # Roll the history buffer (shift past frames left, append new frame at the end)
        # Slice [N, 1:10, 92] and concat with [N, 1, 92] -> [N, 10, 92]
        rolled_history = jnp.concatenate([
            state.history_obs[:, 1:, :], 
            jnp.expand_dims(new_base_obs, axis=1)
        ], axis=1)
        
        new_state = new_state.replace(history_obs=rolled_history)
        
        # Return the flat sequence
        obs = rolled_history.reshape((N, -1))

        return obs, new_state, reward, done, info

    @staticmethod
    def get_base_obs(state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Constructs a SINGLE multi-agent observation vector [N, 92]."""
        N = params.num_agents
        pos = state.usv_state.eta[:, :2]
        yaw = state.usv_state.eta[:, 2]
        nu = state.usv_state.nu
        
        # 1. Ego State Features [N, 8]
        sin_yaw = jnp.sin(yaw)
        cos_yaw = jnp.cos(yaw)
        
        rel_goal = state.goal_pos - pos
        dist_to_goal = jnp.linalg.norm(rel_goal, axis=1)
        angle_to_goal = jnp.arctan2(rel_goal[:, 1], rel_goal[:, 0]) - yaw
        
        ego_feats = jnp.stack([
            sin_yaw, cos_yaw,
            nu[:, 0], nu[:, 1], nu[:, 2],
            # Normalize by 1000.0 (fixed, NOT map_size) so policy generalizes
            # to any mission scale: 0.05 = 50m docking, 0.8 = 800m transit.
            jnp.clip(dist_to_goal / 1000.0, 0.0, 1.0),
            jnp.sin(angle_to_goal), jnp.cos(angle_to_goal)
        ], axis=1)
        
        # 2. LiDAR [N, 64]
        vmap_lidar = jax.vmap(jax_synthetic_lidar, in_axes=(0, 0, None, None, None))
        lidar_dists = vmap_lidar(pos, yaw, state.obstacles, params.lidar_range, params.num_lidar_beams)
        lidar_norm = lidar_dists / params.lidar_range
        
        # 3. Dynamic Neighbor Tracking (Deep Sets AUV Entities) [N, N-1 * 5]
        def get_neighbor_features(ego_idx):
            # We cannot use boolean masking pos[mask] because JAX requires static shapes.
            idx = jnp.arange(N - 1)
            neighbor_idx = jnp.where(idx >= ego_idx, idx + 1, idx)
            
            ego_pos = pos[ego_idx]
            ego_yaw = yaw[ego_idx]
            ego_vel = nu[ego_idx, :2]
            
            c, s = jnp.cos(ego_yaw), jnp.sin(ego_yaw)
            R_inv = jnp.array([[c, s], [-s, c]])
            
            other_pos = pos[neighbor_idx]
            other_vel = nu[neighbor_idx, :2]
            
            rel_pos = other_pos - ego_pos
            rel_vel = other_vel - ego_vel
            
            rel_pos_body = jnp.dot(rel_pos, R_inv.T)
            rel_vel_body = jnp.dot(rel_vel, R_inv.T)
            
            active_flag = jnp.ones((N-1, 1))
            
            neighbor_feats = jnp.concatenate([active_flag, rel_pos_body, rel_vel_body], axis=1).flatten()
            return neighbor_feats
            
        neighbor_feats = jax.vmap(get_neighbor_features)(jnp.arange(N))
        
        return jnp.concatenate([ego_feats, lidar_norm, neighbor_feats], axis=1)
