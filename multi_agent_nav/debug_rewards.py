import jax
import jax.numpy as jnp
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor
import os

def main():
    env = JaxUSVEnv()
    params = env.default_params.replace(encircle_mode=True, encircle_radius=80.0, num_agents=5)
    
    rng = jax.random.PRNGKey(42)
    rng, map_key, reset_key = jax.random.split(rng, 3)
    
    goals, obs = env.generate_map_bank(map_key, 5, 200, 3000.0, 10)
    params = params.replace(goals_bank=goals, obstacles_bank=obs)
    
    obs_arr, state = env.reset(reset_key, params)
    
    # Load Actor
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": 4 * 5, "count": 4, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    actor = Actor(layout=layout, action_dim=2, action_scale=jnp.ones(2), action_bias=jnp.zeros(2))
    dummy_obs = jnp.zeros((1, 920))
    
    rng, actor_key = jax.random.split(rng)
    actor_params = actor.init(actor_key, dummy_obs)["params"]
    
    jitted_step = jax.jit(env.step)
    
    print("Stepping environment for 50 steps...")
    for i in range(50):
        rng, act_key = jax.random.split(rng)
        action, _ = actor.apply({"params": actor_params}, obs_arr, act_key, method=actor.get_action)
        
        obs_arr, state, reward, done, info = jitted_step(rng, state, action, params)
        
        print(f"Step {i+1} | Reward: {reward[0]:.4f} | Approach: {info.get('r_enc_approach', [0])[0]:.4f} | Dense: {info.get('r_enc_dense', [0])[0]:.4f}")
        if jnp.any(done):
            print(f"Episode done at step {i+1}. Collision: {info.get('collision')}")
            break

if __name__ == "__main__":
    main()
