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

@struct.dataclass
class EnvParams:
    """Static configuration for the environment."""
    num_agents: int = 5
    max_steps: int = 2000       # Shorter episodes = faster replay buffer diversity
    map_size: float = 5000.0    # Massive 5km x 5km map
    num_obstacles: int = struct.field(pytree_node=False, default=300) # Dense minefield
    lidar_range: float = 50.0
    num_lidar_beams: int = struct.field(pytree_node=False, default=64)
    goal_radius: float = 15.0
    collision_radius: float = 4.0
    usv_params: USVParams = USVParams()

class JaxUSVEnv:
    """
    A pure JAX implementation of the Multi-Agent Nav Environment.
    Simulates N agents interacting in the same physical space.
    """
    def __init__(self):
        self.default_params = EnvParams()

    # _compute_phi removed. PBRS scrapped due to unbounded reward variance.
    # Using simple progress + heading reward instead.

    @staticmethod
    def reset(key: jax.random.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
        """Resets the environment with N agents."""
        key_pos, key_goal, key_obs = jax.random.split(key, 3)

        N = params.num_agents

        # Spawn agents in a tighter box so they start closer to their goals
        init_pos = jax.random.uniform(key_pos, shape=(N, 2), minval=-50.0, maxval=50.0)
        init_yaw = jax.random.uniform(key_pos, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)

        eta = jnp.concatenate([init_pos, init_yaw[:, None]], axis=1)  # [N, 3]
        nu  = jnp.zeros((N, 3))
        usv_state = USVState(eta=eta, nu=nu)

        half_map = params.map_size / 2.0
        
        # Goals inside 1000m radius but outside 600m radius (Annulus spawn)
        def place_goals(k, n, hm, min_radius=600.0, max_radius=1000.0):
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
                    
                # 500 tries to find a valid spot
                k, final_candidate, _ = jax.lax.fori_loop(0, 500, try_fn, (k, jnp.zeros(2), False))
                g_xy = g_xy.at[i].set(final_candidate)
                return k, g_xy

            k, sk = jax.random.split(k)
            obs_xy = jnp.zeros((n, 2))
            _, final_goals = jax.lax.fori_loop(0, n, body_fn, (k, obs_xy))
            return final_goals

        key_goal, key_place_goal = jax.random.split(key_goal)
        goal_pos = place_goals(key_place_goal, N, half_map, min_radius=600.0, max_radius=1000.0)

        # Static Obstacles — dense minefield with minimum distance guarantees
        num_obstacles = params.num_obstacles
        obs_r  = jax.random.uniform(key_obs, shape=(num_obstacles, 1), minval=3.0, maxval=10.0)
        
        # JAX Rejection Sampling for extreme safety guarantees
        def place_obstacles(k, n, hm, goals, min_obs_dist=50.0, min_goal_dist=50.0, safe_spawn_radius=150.0):
            def body_fn(i, val):
                k, obs_xy = val
                def try_fn(j, try_val):
                    k, candidate, found = try_val
                    k, sk = jax.random.split(k)
                    new_xy = jax.random.uniform(sk, shape=(2,), minval=-hm, maxval=hm)
                    
                    # 1. Dist to other obstacles
                    mask = jnp.arange(n) < i
                    obs_dists = jnp.linalg.norm(obs_xy - new_xy, axis=1)
                    valid_obs = jnp.all(jnp.logical_or(~mask, obs_dists >= min_obs_dist))
                    
                    # 2. Dist to all goals
                    goal_dists = jnp.linalg.norm(goals - new_xy, axis=1)
                    valid_goals = jnp.all(goal_dists >= min_goal_dist)
                    
                    # 3. Dist to spawn center
                    spawn_dist = jnp.linalg.norm(new_xy)
                    valid_spawn = spawn_dist >= safe_spawn_radius
                    
                    valid = valid_obs & valid_goals & valid_spawn
                    
                    candidate = jnp.where(jnp.logical_and(~found, valid), new_xy, candidate)
                    found = jnp.logical_or(found, valid)
                    return k, candidate, found
                    
                # 200 tries to find a valid spot
                k, final_candidate, _ = jax.lax.fori_loop(0, 200, try_fn, (k, jnp.zeros(2), False))
                obs_xy = obs_xy.at[i].set(final_candidate)
                return k, obs_xy

            # First obstacle must also be valid w.r.t goals and spawn
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
            
        key_obs, key_place = jax.random.split(key_obs)
        obs_xy = place_obstacles(key_place, num_obstacles, half_map, goal_pos)
        
        obstacles = jnp.concatenate([obs_xy, obs_r], axis=1)

        # Initial distances for progress reward
        init_dist = jnp.linalg.norm(goal_pos - init_pos, axis=1)  # [N]

        state = EnvState(
            usv_state=usv_state,
            goal_pos=goal_pos,
            obstacles=obstacles,
            step_count=0,
            time=0.0,
            prev_dist=init_dist
        )

        obs = JaxUSVEnv.get_obs(state, params)
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

        obs = JaxUSVEnv.get_obs(new_state, params)

        info = {
            "reached_goal": reached_goal,
            "collision":    collision,
            "timeout":      timeout,
            "dist_to_goal": curr_dist
        }

        return obs, new_state, reward, done, info

    @staticmethod
    def get_obs(state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Constructs the multi-agent observation vector for all N agents [N, Obs_Dim]."""
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
