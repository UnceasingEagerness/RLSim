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
    """Computes the loss and gradients for the CTDE Centralized Critic."""
    # Joint observations: [B, N, ...]
    obs      = transitions.obs
    action   = transitions.action
    reward   = transitions.reward
    next_obs = transitions.next_obs
    done     = transitions.done
    
    B, N, _ = obs.shape
    
    # Global reward is sum of individual rewards
    global_reward = jnp.sum(reward, axis=1) # [B]
    # Global done if ANY agent is done
    global_done = jnp.any(done, axis=1).astype(jnp.float32) # [B]

    # 1. Get next actions + log probs from decentralized current policy
    # Reshape to [B*N, obs_dim] for the Actor
    next_obs_flat = next_obs.reshape((B * N, -1))
    next_action_flat, next_log_prob_flat = actor_state.apply_fn(
        {"params": actor_state.params}, next_obs_flat, key, method="get_action"
    )
    
    next_action = next_action_flat.reshape((B, N, -1))
    next_log_prob = next_log_prob_flat.reshape((B, N))
    
    # Global entropy is the sum of individual entropies
    global_log_prob = jnp.sum(next_log_prob, axis=1) # [B]

    # 2. Compute target Q using Centralized Target Critic
    # CentralizedSoftQNetwork takes [B, N, obs_dim] and [B, N, act_dim]
    q_target = critic_state.apply_fn(
        {"params": target_critic_params}, next_obs, next_action
    ).squeeze(-1)  # [B]

    alpha = jnp.exp(log_alpha)
    # Bellman backup for joint state
    target_q = global_reward + (1.0 - global_done) * gamma * (
        q_target - alpha * global_log_prob
    )  # [B]

    def critic_loss_fn(params):
        q1 = critic_state.apply_fn({"params": params}, obs, action).squeeze(-1)  # [B]
        loss = jnp.mean((q1 - target_q) ** 2)
        return loss

    loss, grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
    new_critic_state = critic_state.apply_gradients(grads=grads)
    return new_critic_state, loss

def update_actor(actor_state: TrainState, critic_state: TrainState, log_alpha: jnp.ndarray, obs: jnp.ndarray, key: jax.random.PRNGKey):
    """Computes the loss and gradients for the CTDE Actor."""
    # obs: [B, N, obs_dim]
    B, N, _ = obs.shape

    def actor_loss_fn(params):
        obs_flat = obs.reshape((B * N, -1))
        action_flat, log_prob_flat = actor_state.apply_fn(
            {"params": params}, obs_flat, key, method="get_action"
        )
        
        action = action_flat.reshape((B, N, -1))
        log_prob = log_prob_flat.reshape((B, N))
        
        # Centralized Critic evaluates the JOINT action
        q_value = critic_state.apply_fn(
            {"params": critic_state.params}, obs, action
        ).squeeze(-1)  # [B]
        
        global_log_prob = jnp.sum(log_prob, axis=1) # [B]
        alpha = jnp.exp(log_alpha)
        
        # SAC actor maximises Global Q while maintaining Global entropy
        loss = jnp.mean(alpha * global_log_prob - q_value)
        
        return loss, log_prob_flat

    (loss, log_prob_flat), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
    new_actor_state = actor_state.apply_gradients(grads=grads)
    return new_actor_state, loss, log_prob_flat

def update_alpha(log_alpha: jnp.ndarray, opt_state: Any, log_prob: jnp.ndarray, target_entropy: float, optimizer: optax.GradientTransformation):
    """Updates the temperature parameter alpha."""
    # log_prob here is flattened from actor [B*N]. target_entropy is per-agent.
    def alpha_loss_fn(log_alpha):
        alpha = jnp.exp(log_alpha)
        loss = -jnp.mean(alpha * (log_prob + target_entropy))
        return loss
        
    loss, grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
    updates, new_opt_state = optimizer.update(grads, opt_state, log_alpha)
    new_log_alpha = optax.apply_updates(log_alpha, updates)
    return new_log_alpha, new_opt_state, loss
