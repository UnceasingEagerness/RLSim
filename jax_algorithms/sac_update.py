import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Any

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray

def update_critic(critic_state: TrainState, target_critic_params: Any, actor_state: TrainState, log_alpha: jnp.ndarray, transitions: Transition, gamma: float, key: jax.random.PRNGKey):
    """Computes the loss and gradients for the SAC Critic."""
    obs = transitions.obs
    action = transitions.action
    reward = transitions.reward
    next_obs = transitions.next_obs
    done = transitions.done
    
    # 1. Get next actions from current policy
    next_action, next_log_prob = actor_state.apply_fn({"params": actor_state.params}, next_obs, key, method="get_action")
    
    # 2. Get target Q values
    q1_target = critic_state.apply_fn({"params": target_critic_params}, next_obs, next_action)
    # If we had 2 critics we would take the minimum, for simplicity in this port we assume 1 or we duplicate it
    # We will assume a single critic for brevity in this snippet
    
    alpha = jnp.exp(log_alpha)
    target_q = reward + (1.0 - done) * gamma * (q1_target - alpha * next_log_prob)
    
    def critic_loss_fn(params):
        q1 = critic_state.apply_fn({"params": params}, obs, action)
        loss = jnp.mean((q1 - target_q)**2)
        return loss
    
    loss, grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=grads)
    return new_critic_state, loss

def update_actor(actor_state: TrainState, critic_state: TrainState, log_alpha: jnp.ndarray, obs: jnp.ndarray, key: jax.random.PRNGKey):
    """Computes the loss and gradients for the SAC Actor."""
    
    def actor_loss_fn(params):
        action, log_prob = actor_state.apply_fn({"params": params}, obs, key, method="get_action")
        q_value = critic_state.apply_fn({"params": critic_state.params}, obs, action)
        alpha = jnp.exp(log_alpha)
        # Minimize alpha * log_prob - Q
        loss = jnp.mean(alpha * log_prob - q_value)
        return loss, log_prob
        
    (loss, log_prob), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=grads)
    return new_actor_state, loss, log_prob

def update_alpha(log_alpha: jnp.ndarray, opt_state: Any, log_prob: jnp.ndarray, target_entropy: float, optimizer: optax.GradientTransformation):
    """Updates the temperature parameter alpha."""
    
    def alpha_loss_fn(log_alpha):
        alpha = jnp.exp(log_alpha)
        loss = -jnp.mean(alpha * (log_prob + target_entropy))
        return loss
        
    loss, grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    updates, new_opt_state = optimizer.update(grads, opt_state, log_alpha)
    new_log_alpha = optax.apply_updates(log_alpha, updates)
    return new_log_alpha, new_opt_state, loss
