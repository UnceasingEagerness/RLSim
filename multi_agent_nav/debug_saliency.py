import os
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

env_num_agents = 5  
nn_num_agents = 5   
obs_dim_env = (72 + (env_num_agents - 1) * 5) * 10
obs_dim_nn = (72 + (nn_num_agents - 1) * 5) * 10
layout = {
    "ego": {"start": 0, "dim": 8},
    "goal": {"start": 0, "dim": 8}, 
    "lidar": {"start": 8, "dim": 64},
    "auv_entities": {"start": 72, "dim": (nn_num_agents - 1) * 5, "count": nn_num_agents - 1, "feature_dim": 5},
    "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
}

env = JaxUSVEnv()
env_params = env.default_params.replace(num_agents=env_num_agents, map_size=2000.0, num_obstacles=150, max_steps=1500)
rng = jax.random.PRNGKey(42)
goals_bank, obstacles_bank = env.generate_map_bank(rng, int(env_params.num_agents), int(env_params.num_obstacles), float(env_params.map_size), 10)
env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank)

actor = Actor(layout=layout, action_dim=2, action_scale=jnp.ones(2), action_bias=jnp.zeros(2))
ckpt = PyTreeCheckpointer()
dummy_obs = jnp.zeros((1, obs_dim_nn))
init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
raw_params = ckpt.restore("checkpoints/sac_actor_final", item=init_params, restore_args=jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params))
actor_params = jax.tree_util.tree_map(jnp.array, raw_params)

def get_saliency(params, obs_single):
    def batched_action(o):
        o_b = jnp.expand_dims(o, 0)
        mean, _ = actor.apply({"params": params}, o_b)
        return jnp.tanh(mean)[0]
    J = jax.jacobian(batched_action)(obs_single) 
    return jnp.mean(jnp.abs(J), axis=0) 

jit_saliency = jax.jit(get_saliency)
vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
obs_batch, state_batch = vmap_reset(jax.random.split(rng, 1), env_params)

flat_obs = obs_batch.reshape(env_num_agents, obs_dim_env)
sal = jit_saliency(actor_params, flat_obs[0])
sal_frames = sal.reshape((10, 92))
sal_spatial = np.sum(sal_frames, axis=0)

k = np.sum(sal_spatial[:8])
l = np.sum(sal_spatial[8:72])
n = np.sum(sal_spatial[72:92])
total = k + l + n
print(f"Kin: {k:.5f} ({k/total*100:.2f}%)")
print(f"Lidar: {l:.5f} ({l/total*100:.2f}%)")
print(f"Neighbors: {n:.5f} ({n/total*100:.2f}%)")
