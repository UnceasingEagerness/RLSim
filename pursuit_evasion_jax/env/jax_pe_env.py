import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict

def ssa(angle):
    """Smallest signed angle between -pi and pi."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

@struct.dataclass
class PEEnvParams:
    num_pursuers: int = 3
    num_evaders: int = 1
    num_agents: int = 4 # Total agents
    max_steps: int = 1000
    map_size: float = 100.0
    capture_radius: float = 5.0
    sensor_range: float = 20.0
    num_obstacles: int = 8
    
    # Physics constraints (Asymmetric for Evader at index 3)
    # Throttle bounds: Pursuers [0, 80], Evader [0, 48] (0.6 * 80)
    throttle_scales: tuple = (80.0, 80.0, 80.0, 48.0)
    # Steering bounds: Pursuers [-15, 15], Evader [-18, 18] (1.2 * 15)
    steering_scales: tuple = (15.0, 15.0, 15.0, 18.0)

@struct.dataclass
class PEEnvState:
    eta: jnp.ndarray # [4, 3] -> (x, y, yaw)
    nu: jnp.ndarray  # [4, 3] -> (u, v, r)
    static_obstacles: jnp.ndarray # [O, 3] -> (x, y, radius)
    step_count: int

class JaxPursuitEvasionEnv:
    def __init__(self):
        self.default_params = PEEnvParams()
        self.sonar_angles = jnp.linspace(-jnp.pi/3, jnp.pi/3, 11) # -60 to 60 deg

    def usv_dynamics_step(self, eta: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray, dt: float = 0.1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Vectorized Fossen 3-DOF USV Dynamics."""
        # Updated to standard WAM-V like parameters (previously damping was 215.0 which capped speed to 0.3 m/s)
        m11, m22, m33 = 25.8, 33.8, 2.76
        d11, d22, d33 = 2.0, 7.0, 0.5

        u, v, r = nu[:, 0], nu[:, 1], nu[:, 2]
        tau_u, tau_v, tau_r = tau[:, 0], tau[:, 1], tau[:, 2]

        # Dropping Coriolis forces for numerical stability (Euler integration at dt=0.1 explodes with realistic masses)
        u_dot = (tau_u - d11 * u) / m11
        v_dot = (tau_v - d22 * v) / m22
        r_dot = (tau_r - d33 * r) / m33

        next_nu = jnp.stack([u + u_dot * dt, v + v_dot * dt, r + r_dot * dt], axis=-1)
        # Safety clip to guarantee no NaNs ever
        next_nu = jnp.clip(next_nu, -25.0, 25.0)

        yaw = eta[:, 2]
        cos_psi = jnp.cos(yaw)
        sin_psi = jnp.sin(yaw)

        x_dot = next_nu[:, 0] * cos_psi - next_nu[:, 1] * sin_psi
        y_dot = next_nu[:, 0] * sin_psi + next_nu[:, 1] * cos_psi
        yaw_dot = next_nu[:, 2]

        next_eta = jnp.stack([
            eta[:, 0] + x_dot * dt,
            eta[:, 1] + y_dot * dt,
            ssa(eta[:, 2] + yaw_dot * dt)
        ], axis=-1)

        return next_eta, next_nu

    def reset(self, key: jax.random.PRNGKey, params: PEEnvParams) -> Tuple[jnp.ndarray, PEEnvState]:
        key, e_key, obs_key = jax.random.split(key, 3)
        
        # Pursuer fixed spawn (10, 10), (10, 20), (10, 30) facing pi/4
        p_eta = jnp.array([
            [10.0, 10.0, jnp.pi/4],
            [10.0, 20.0, jnp.pi/4],
            [10.0, 30.0, jnp.pi/4]
        ])
        
        # Evader random spawn in [40, 80] facing pi/4
        e_pos = jax.random.uniform(e_key, shape=(1, 2), minval=40.0, maxval=80.0)
        e_eta = jnp.concatenate([e_pos, jnp.array([[jnp.pi/4]])], axis=-1)
        
        eta = jnp.concatenate([p_eta, e_eta], axis=0)
        nu = jnp.zeros((params.num_agents, 3))
        
        # Obstacles (Simplified random placement for now)
        obs_pos = jax.random.uniform(obs_key, shape=(params.num_obstacles, 2), minval=20.0, maxval=80.0)
        obs_rad = jax.random.uniform(obs_key, shape=(params.num_obstacles, 1), minval=3.0, maxval=8.0)
        static_obstacles = jnp.concatenate([obs_pos, obs_rad], axis=-1)
        
        state = PEEnvState(eta=eta, nu=nu, static_obstacles=static_obstacles, step_count=0)
        obs = self.get_obs(state, params)
        return obs, state

    def step(self, key: jax.random.PRNGKey, state: PEEnvState, action: jnp.ndarray, params: PEEnvParams):
        """
        action: [4, 2] -> (throttle_cmd [-1, 1], steering_cmd [-1, 1])
        """
        # Convert to arrays and apply asymmetrical dynamics
        throttle = jnp.clip(action[:, 0], -1.0, 1.0) * jnp.array(params.throttle_scales)
        steering = jnp.clip(action[:, 1], -1.0, 1.0) * jnp.array(params.steering_scales)
        
        tau = jnp.stack([throttle, jnp.zeros(params.num_agents), steering], axis=-1)
        
        next_eta, next_nu = self.usv_dynamics_step(state.eta, state.nu, tau)
        
        # Boundary clipping
        next_eta = next_eta.at[:, 0].set(jnp.clip(next_eta[:, 0], 0.0, params.map_size))
        next_eta = next_eta.at[:, 1].set(jnp.clip(next_eta[:, 1], 0.0, params.map_size))
        
        next_state = state.replace(eta=next_eta, nu=next_nu, step_count=state.step_count + 1)
        
        # Collision detection (Obstacles & Agents)
        obs = self.get_obs(next_state, params) # Extracts LiDAR and positions
        
        # We need lidar distances for collision checking. 
        # But to avoid recomputing, we can extract them from the observations
        # Or recompute efficiently
        from env.jax_lidar import jax_synthetic_lidar
        vmap_lidar = jax.vmap(jax_synthetic_lidar, in_axes=(0, 0, None, None, None))
        lidar_dists = vmap_lidar(next_eta[:, :2], next_eta[:, 2], state.static_obstacles, params.sensor_range, 64)
        min_dist_obs = jnp.min(lidar_dists, axis=1)
        collision_obs = min_dist_obs < 2.0
        
        pos_diff = next_eta[:, None, :2] - next_eta[None, :, :2]
        agent_dists = jnp.linalg.norm(pos_diff, axis=-1)
        agent_dists = jnp.where(jnp.eye(params.num_agents, dtype=bool), jnp.inf, agent_dists)
        min_agent_dist = jnp.min(agent_dists, axis=1)
        collision_agent = min_agent_dist < 4.0
        
        collision = collision_obs | collision_agent
        
        from env.jax_pe_reward import compute_rewards
        rewards = compute_rewards(next_eta[:3, :2], next_eta[3, :2], collision[:3], collision[3])
        
        # Check termination
        d_p_to_e = jnp.linalg.norm(next_eta[:3, :2] - next_eta[3, :2], axis=1)
        is_captured = jnp.all(d_p_to_e < params.capture_radius)
        timeout = next_state.step_count >= params.max_steps
        
        # Episode terminates if captured, timeout, or any collision occurs
        done_flag = is_captured | timeout | jnp.any(collision)
        done = jnp.full((params.num_agents,), done_flag)
        
        return obs, next_state, rewards, done, {"collision": collision, "captured": is_captured}

    def get_obs(self, state: PEEnvState, params: PEEnvParams) -> jnp.ndarray:
        from env.jax_lidar import jax_synthetic_lidar
        
        eta = state.eta
        pos = eta[:, :2]
        yaw = eta[:, 2]
        
        e_pos = pos[3]
        e_yaw = yaw[3]
        
        # Precompute pursuer to evader distances
        d_p_to_e = jnp.linalg.norm(pos[:3] - e_pos, axis=1)
        d_mean = jnp.mean(d_p_to_e)
        
        # LiDAR for all agents
        vmap_lidar = jax.vmap(jax_synthetic_lidar, in_axes=(0, 0, None, None, None))
        lidar_dists = vmap_lidar(pos, yaw, state.static_obstacles, params.sensor_range, 64)
        
        # --- Pursuer Observations (vmap over 0, 1, 2) ---
        def get_pursuer_obs(p_idx):
            p_pos = pos[p_idx]
            p_yaw = yaw[p_idx]
            
            # O_out
            bearing_T = jnp.arctan2(e_pos[1] - p_pos[1], e_pos[0] - p_pos[0])
            theta_aT = ssa(p_yaw - bearing_T)
            da = d_p_to_e[p_idx] - params.capture_radius
            O_out = jnp.array([theta_aT, da, d_mean])
            
            # O_in (2 neighbors)
            idx_array = jnp.arange(2)
            # if p_idx=0 -> neighbors 1,2. p_idx=1 -> 0,2. p_idx=2 -> 0,1.
            nbor_indices = jnp.where(idx_array >= p_idx, idx_array + 1, idx_array)
            
            def get_nbor_feats(n_idx):
                n_pos = pos[n_idx]
                n_yaw = yaw[n_idx]
                dab = jnp.linalg.norm(p_pos - n_pos)
                bearing_ab = jnp.arctan2(n_pos[1] - p_pos[1], n_pos[0] - p_pos[0])
                theta_ab = ssa(p_yaw - bearing_ab)
                theta_ba = ssa(n_yaw - bearing_ab)
                delta_dab = d_p_to_e[p_idx] - d_p_to_e[n_idx]
                
                v_ego = p_pos - e_pos
                v_nbor = n_pos - e_pos
                angle_ego = jnp.arctan2(v_ego[1], v_ego[0])
                angle_nbor = jnp.arctan2(v_nbor[1], v_nbor[0])
                gamma_ab = jnp.abs(ssa(angle_ego - angle_nbor))
                
                return jnp.array([dab, theta_ab, theta_ba, delta_dab, gamma_ab])
                
            O_in = jax.vmap(get_nbor_feats)(nbor_indices).flatten()
            
            return jnp.concatenate([O_out, O_in, lidar_dists[p_idx]])
            
        pursuer_obs = jax.vmap(get_pursuer_obs)(jnp.arange(3)) # [3, 77]
        
        # --- Evader Observation ---
        def get_evader_nbor(p_idx):
            p_pos = pos[p_idx]
            bearing_p = jnp.arctan2(p_pos[1] - e_pos[1], p_pos[0] - e_pos[0])
            phi_a = ssa(e_yaw - bearing_p)
            return jnp.array([phi_a, d_p_to_e[p_idx]])
            
        O_out_e = jax.vmap(get_evader_nbor)(jnp.arange(3)).flatten() # [6]
        evader_obs_raw = jnp.concatenate([O_out_e, lidar_dists[3]]) # [70]
        
        # Pad Evader obs with 7 zeros to match the 77-D shape of Pursuers
        evader_obs = jnp.pad(evader_obs_raw, (0, 7)) # [77]
        
        obs = jnp.concatenate([pursuer_obs, jnp.expand_dims(evader_obs, axis=0)], axis=0) # [4, 77]
        return obs
