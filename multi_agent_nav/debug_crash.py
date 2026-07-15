import jax
import jax.numpy as jnp
from train_pure_jax import Actor, CentralizedSoftQNetwork, JaxReplayBuffer
from env.jax_usv_env import JaxUSVEnv

print("1. Initializing Environment...")
env = JaxUSVEnv()
env_params = env.default_params.replace(num_agents=5)

print("2. Jitting reset...")
vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
rng = jax.random.PRNGKey(42)
reset_keys = jax.random.split(rng, 128)

print("3. Executing vmap_reset...")
init_obs, init_env_state = vmap_reset(reset_keys, env_params)
print("vmap_reset complete.")

print("4. Initializing Actor...")
layout = {
    "ego": {"start": 0, "dim": 8},
    "goal": {"start": 0, "dim": 8}, 
    "lidar": {"start": 8, "dim": 64},
    "auv_entities": {"start": 72, "dim": 4 * 5, "count": 4, "feature_dim": 5},
    "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
}
actor = Actor(layout=layout, action_dim=2, action_scale=jnp.ones(2), action_bias=jnp.zeros(2))
dummy_obs_actor = jnp.zeros((1, 920))
actor_key = jax.random.PRNGKey(0)
actor_params = actor.init(actor_key, dummy_obs_actor)["params"]
print("Actor complete.")

print("5. Initializing Critic...")
critic = CentralizedSoftQNetwork()
dummy_obs_critic = jnp.zeros((1, 5, 920))
dummy_act_critic = jnp.zeros((1, 5, 2))
critic_key = jax.random.PRNGKey(1)
critic_params = critic.init(critic_key, dummy_obs_critic, dummy_act_critic)["params"]
print("Critic complete.")
