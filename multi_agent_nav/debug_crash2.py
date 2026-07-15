import jax
import jax.numpy as jnp
from train_pure_jax import Actor, CentralizedSoftQNetwork, JaxReplayBuffer
from env.jax_usv_env import JaxUSVEnv

print("1. Initializing Environment...")
env = JaxUSVEnv()
env_params = env.default_params.replace(num_agents=5)

print("2. Jitting map gen...")
rng = jax.random.PRNGKey(42)
rng, map_key = jax.random.split(rng)
jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
goals_bank, obstacles_bank = jitted_map_gen(map_key, 5, 200, 3000.0, 1000)
jax.block_until_ready(goals_bank)
env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank)
print("Map Bank successfully loaded!")

print("3. Executing vmap_reset...")
vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
rng, _rng = jax.random.split(rng)
reset_keys = jax.random.split(_rng, 1)
init_obs, init_env_state = vmap_reset(reset_keys, env_params)
jax.block_until_ready(init_obs)
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
