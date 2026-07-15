import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict

# Import our JAX physics components
from jax_envs.jax_dynamics import USVState, USVParams, rk4_step
from jax_envs.jax_lidar import jax_synthetic_lidar

@struct.dataclass
class EnvState:
    """The complete state of the environment."""
    usv_state: USVState
    goal_pos: jnp.ndarray
    obstacles: jnp.ndarray  # [N, 3] matrix of [x, y, radius]
    step_count: int
    time: float

@struct.dataclass
class EnvParams:
    """Static configuration for the environment."""
    max_steps: int = 2000
    map_size: float = 1500.0
    lidar_range: float = 50.0
    num_lidar_beams: int = 64
    goal_radius: float = 20.0
    usv_params: USVParams = USVParams()

class JaxUSVEnv:
    """
    A pure JAX implementation of the Multi-Agent Nav Environment.
    Follows the gymnax functional API paradigm for massive vectorization.
    """
    def __init__(self):
        self.default_params = EnvParams()

    @jax.jit
    def reset(self, key: jax.random.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
        """Resets the environment to an initial state."""
        key_pos, key_goal, key_obs = jax.random.split(key, 3)
        
        # Random initial position within a 200m box
        init_pos = jax.random.uniform(key_pos, shape=(2,), minval=-100.0, maxval=100.0)
        init_yaw = jax.random.uniform(key_pos, minval=-jnp.pi, maxval=jnp.pi)
        eta = jnp.array([init_pos[0], init_pos[1], init_yaw])
        nu = jnp.zeros(3)
        usv_state = USVState(eta=eta, nu=nu)
        
        # Random goal position
        goal_pos = jax.random.uniform(key_goal, shape=(2,), minval=-300.0, maxval=300.0)
        
        # Generate some random static obstacles
        # For JAX, the shape of arrays MUST be static, so we always generate N obstacles.
        num_obstacles = 10
        obs_xy = jax.random.uniform(key_obs, shape=(num_obstacles, 2), minval=-300.0, maxval=300.0)
        obs_r = jax.random.uniform(key_obs, shape=(num_obstacles, 1), minval=5.0, maxval=20.0)
        obstacles = jnp.concatenate([obs_xy, obs_r], axis=1)
        
        state = EnvState(
            usv_state=usv_state,
            goal_pos=goal_pos,
            obstacles=obstacles,
            step_count=0,
            time=0.0
        )
        
        obs = self.get_obs(state, params)
        return obs, state

    @jax.jit
    def step(self, key: jax.random.PRNGKey, state: EnvState, action: jnp.ndarray, params: EnvParams) -> Tuple[jnp.ndarray, EnvState, float, bool, Dict]:
        """Steps the environment dynamics forward."""
        
        # Action space: [throttle, steering]
        throttle = jnp.clip(action[0], -1.0, 1.0)
        steering = jnp.clip(action[1], -1.0, 1.0)
        
        # Decode action to Fossen forces
        tau_u = throttle * 100.0  # Surge force
        tau_r = steering * 20.0   # Yaw moment
        tau = jnp.array([tau_u, 0.0, tau_r])
        
        # RK4 Physics Step (Fully JIT compiled)
        new_usv_state = rk4_step(state.usv_state, tau, params.usv_params)
        
        # Update Environment State
        new_state = state.replace(
            usv_state=new_usv_state,
            step_count=state.step_count + 1,
            time=state.time + params.usv_params.dt
        )
        
        # Compute Observation
        obs = self.get_obs(new_state, params)
        
        # Reward & Termination
        pos = new_usv_state.eta[:2]
        dist_to_goal = jnp.linalg.norm(state.goal_pos - pos)
        
        reached_goal = dist_to_goal < params.goal_radius
        timeout = new_state.step_count >= params.max_steps
        
        # Check collision (distance to closest obstacle < USV radius)
        # Using LiDAR distances as a proxy for collision logic (if min lidar < threshold)
        lidar_dists = jax_synthetic_lidar(pos, new_usv_state.eta[2], state.obstacles, params.lidar_range, params.num_lidar_beams)
        min_dist = jnp.min(lidar_dists)
        collision = min_dist < 2.0
        
        done = reached_goal | timeout | collision
        
        # PBRS Dense Reward
        prev_dist = jnp.linalg.norm(state.goal_pos - state.usv_state.eta[:2])
        reward_shaping = (prev_dist - dist_to_goal) * 1.0
        
        reward = reward_shaping
        reward = jnp.where(reached_goal, 100.0, reward)
        reward = jnp.where(collision, -100.0, reward)
        
        info = {
            "reached_goal": reached_goal,
            "collision": collision,
            "timeout": timeout,
            "dist_to_goal": dist_to_goal
        }
        
        return obs, new_state, reward, done, info

    @jax.jit
    def get_obs(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Constructs the observation vector."""
        pos = state.usv_state.eta[:2]
        yaw = state.usv_state.eta[2]
        nu = state.usv_state.nu
        
        # Ego State
        sin_yaw = jnp.sin(yaw)
        cos_yaw = jnp.cos(yaw)
        
        # Goal relative
        rel_goal = state.goal_pos - pos
        dist_to_goal = jnp.linalg.norm(rel_goal)
        angle_to_goal = jnp.arctan2(rel_goal[1], rel_goal[0]) - yaw
        
        ego_feats = jnp.array([
            sin_yaw, cos_yaw,
            nu[0], nu[1], nu[2],
            dist_to_goal / params.map_size,
            jnp.sin(angle_to_goal), jnp.cos(angle_to_goal)
        ])
        
        # LiDAR
        lidar_dists = jax_synthetic_lidar(pos, yaw, state.obstacles, params.lidar_range, params.num_lidar_beams)
        lidar_norm = lidar_dists / params.lidar_range
        
        return jnp.concatenate([ego_feats, lidar_norm])
