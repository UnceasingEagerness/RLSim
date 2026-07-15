import os
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training.train_state import TrainState

# Import custom JAX modules
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor, SoftQNetwork
from algorithms.jax_buffer import JaxReplayBuffer, ReplayBufferState
from algorithms.sac_update import update_critic, update_actor, update_alpha, Transition

def main():
    print(f"Running on JAX devices: {jax.devices()}")
    
    # ── Configurations ────────────────────────────────────────────────────────
    total_timesteps = 40_000
    learning_starts = 1_000
    batch_size = 256
    buffer_size = 100_000
    gamma = 0.99
    tau_target = 0.005
    policy_lr = 3e-4
    q_lr = 3e-4
    target_entropy = -2.0  # -dim(A)
    
    # We will run a simplified observation layout for this test
    # Obs = 8 kin/goal + 64 LiDAR = 72
    obs_dim = 72
    action_dim = 2
    
    # The layout dictionary expected by ActorBackbone/CriticBackbone
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, # Reusing kin_feat for now in simple layout
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    rng = jax.random.PRNGKey(42)
    
    # ── Initialize Environment ────────────────────────────────────────────────
    env = JaxUSVEnv()
    env_params = env.default_params
    
    rng, key_reset = jax.random.split(rng)
    # Jit compile reset and step
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    
    # ── Initialize Networks ───────────────────────────────────────────────────
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    critic = SoftQNetwork(layout=layout)
    
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, action_dim))
    
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    actor_params = actor.init(actor_key, dummy_obs)["params"]
    critic_params = critic.init(critic_key, dummy_obs, dummy_act)["params"]
    
    actor_state = TrainState.create(
        apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate=policy_lr)
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply, params=critic_params, tx=optax.adam(learning_rate=q_lr)
    )
    target_critic_params = critic_params
    
    # Alpha (Temperature)
    log_alpha = jnp.array(0.0)
    alpha_optimizer = optax.adam(learning_rate=policy_lr)
    alpha_opt_state = alpha_optimizer.init(log_alpha)
    
    # ── Initialize Replay Buffer ──────────────────────────────────────────────
    buffer = JaxReplayBuffer(buffer_size, obs_dim, action_dim)
    buffer_state = buffer.init_state()
    
    # ── JIT Compiled Update Step ──────────────────────────────────────────────
    @jax.jit
    def train_step(actor_state, critic_state, target_critic_params, log_alpha, alpha_opt_state, batch, key):
        key_critic, key_actor = jax.random.split(key)
        
        # 1. Update Critic
        critic_state, q_loss = update_critic(critic_state, target_critic_params, actor_state, log_alpha, batch, gamma, key_critic)
        
        # 2. Update Actor
        actor_state, a_loss, log_prob = update_actor(actor_state, critic_state, log_alpha, batch.obs, key_actor)
        
        # 3. Update Alpha
        log_alpha, alpha_opt_state, alpha_loss = update_alpha(log_alpha, alpha_opt_state, log_prob, target_entropy, alpha_optimizer)
        
        # 4. Soft Update Target Critic
        target_critic_params = jax.tree_util.tree_map(
            lambda target, current: tau_target * current + (1 - tau_target) * target,
            target_critic_params, critic_state.params
        )
        
        return actor_state, critic_state, target_critic_params, log_alpha, alpha_opt_state, q_loss, a_loss
    
    # ── Training Loop ─────────────────────────────────────────────────────────
    print("Starting Training Loop (JIT Compiled Steps)")
    start_time = time.time()
    
    obs, env_state = reset_fn(key_reset, env_params)
    episode_reward = 0.0
    episodes = 0
    
    for step in range(total_timesteps):
        rng, action_key, step_key = jax.random.split(rng, 3)
        
        if step < learning_starts:
            action = jax.random.uniform(action_key, shape=(action_dim,), minval=-1.0, maxval=1.0)
        else:
            action, _ = actor.apply({"params": actor_state.params}, jnp.expand_dims(obs, 0), action_key, method=actor.get_action)
            action = action[0] # remove batch dim
            
        next_obs, next_env_state, reward, done, info = step_fn(step_key, env_state, action, env_params)
        episode_reward += reward
        
        # Add to GPU buffer
        buffer_state = buffer.add(buffer_state, obs, action, reward, next_obs, done)
        
        obs = next_obs
        env_state = next_env_state
        
        if done:
            episodes += 1
            print(f"Global Step: {step} | Ep {episodes} Reward: {episode_reward:.2f} | Reached Goal: {info['reached_goal']} | Time: {time.time()-start_time:.1f}s")
            episode_reward = 0.0
            rng, reset_key = jax.random.split(rng)
            obs, env_state = reset_fn(reset_key, env_params)
            
        # Optimization
        if step > learning_starts:
            rng, sample_key, update_key = jax.random.split(rng, 3)
            b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(buffer_state, sample_key, batch_size)
            batch = Transition(obs=b_obs, action=b_act, reward=b_rew, next_obs=b_next_obs, done=b_done)
            
            actor_state, critic_state, target_critic_params, log_alpha, alpha_opt_state, q_loss, a_loss = train_step(
                actor_state, critic_state, target_critic_params, log_alpha, alpha_opt_state, batch, update_key
            )
            
            if step % 5000 == 0:
                print(f"[{step}/{total_timesteps}] Q-Loss: {q_loss:.4f}, A-Loss: {a_loss:.4f}, Alpha: {jnp.exp(log_alpha):.4f}")

if __name__ == "__main__":
    main()
