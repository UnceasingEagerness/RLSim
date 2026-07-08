import jax
import jax.numpy as jnp
import functools
from flax import struct
from typing import Tuple

@struct.dataclass
class ReplayBufferState:
    """State of the GPU-resident Replay Buffer."""
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    count: int
    pos: int
    max_size: int

class JaxReplayBuffer:
    """
    A replay buffer where all data lives permanently on the GPU as a JAX PyTree.
    This entirely eliminates CPU-GPU data transfer during sampling.
    """
    def __init__(self, max_size: int, obs_dim: int, action_dim: int):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def init_state(self) -> ReplayBufferState:
        """Initializes the empty buffers on the GPU."""
        return ReplayBufferState(
            obs=jnp.zeros((self.max_size, self.obs_dim), dtype=jnp.float32),
            actions=jnp.zeros((self.max_size, self.action_dim), dtype=jnp.float32),
            rewards=jnp.zeros((self.max_size,), dtype=jnp.float32),
            next_obs=jnp.zeros((self.max_size, self.obs_dim), dtype=jnp.float32),
            dones=jnp.zeros((self.max_size,), dtype=jnp.bool_),
            count=jnp.array(0, dtype=jnp.int32),
            pos=jnp.array(0, dtype=jnp.int32),
            max_size=jnp.array(self.max_size, dtype=jnp.int32)
        )

    @staticmethod
    @jax.jit
    def add(state: ReplayBufferState, obs: jnp.ndarray, action: jnp.ndarray, reward: float, next_obs: jnp.ndarray, done: bool) -> ReplayBufferState:
        """Adds a single transition to the buffer. Can be vmapped for batched envs."""
        pos = state.pos
        
        new_obs = state.obs.at[pos].set(obs)
        new_actions = state.actions.at[pos].set(action)
        new_rewards = state.rewards.at[pos].set(reward)
        new_next_obs = state.next_obs.at[pos].set(next_obs)
        new_dones = state.dones.at[pos].set(done)
        
        new_pos = (pos + 1) % state.max_size
        new_count = jnp.minimum(state.count + 1, state.max_size)
        
        return state.replace(
            obs=new_obs,
            actions=new_actions,
            rewards=new_rewards,
            next_obs=new_next_obs,
            dones=new_dones,
            count=new_count,
            pos=new_pos
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=['batch_size'])
    def add_batch(state: ReplayBufferState, obs: jnp.ndarray, action: jnp.ndarray, reward: jnp.ndarray, next_obs: jnp.ndarray, done: jnp.ndarray, batch_size: int) -> ReplayBufferState:
        """Adds a batch of E transitions to the buffer."""
        # Calculate the indices for the batch
        indices = (state.pos + jnp.arange(batch_size)) % state.max_size
        
        new_obs = state.obs.at[indices].set(obs)
        new_actions = state.actions.at[indices].set(action)
        new_rewards = state.rewards.at[indices].set(reward)
        new_next_obs = state.next_obs.at[indices].set(next_obs)
        new_dones = state.dones.at[indices].set(done)
        
        new_pos = (state.pos + batch_size) % state.max_size
        new_count = jnp.minimum(state.count + batch_size, state.max_size)
        
        return state.replace(
            obs=new_obs,
            actions=new_actions,
            rewards=new_rewards,
            next_obs=new_next_obs,
            dones=new_dones,
            pos=new_pos,
            count=new_count
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=['batch_size'])
    def sample(state: ReplayBufferState, key: jax.random.PRNGKey, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Samples a batch of transitions from the buffer."""
        idxs = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=state.count)
        
        return (
            state.obs[idxs],
            state.actions[idxs],
            state.rewards[idxs],
            state.next_obs[idxs],
            state.dones[idxs]
        )
