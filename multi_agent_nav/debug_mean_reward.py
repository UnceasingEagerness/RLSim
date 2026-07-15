import jax
import jax.numpy as jnp
from env.jax_usv_env import JaxUSVEnv, EnvParams

def main():
    env = JaxUSVEnv()
    params = env.default_params.replace(encircle_mode=True, encircle_radius=80.0, num_agents=5)
    
    rng = jax.random.PRNGKey(42)
    rng, map_key, reset_key = jax.random.split(rng, 3)
    
    goals, obs = env.generate_map_bank(map_key, 5, 200, 3000.0, 10)
    params = params.replace(goals_bank=goals, obstacles_bank=obs)
    
    obs_arr, state = env.reset(reset_key, params)
    
    jitted_step = jax.jit(env.step)
    
    # Simulate 1000 steps of random actions across 128 envs
    # Wait, just 1 env for 1000 steps to see the exact mean
    
    total_reward = 0.0
    print("Stepping environment for 1000 steps with random actions...")
    for i in range(1000):
        rng, act_key = jax.random.split(rng)
        action = jax.random.uniform(act_key, shape=(5, 2), minval=-1.0, maxval=1.0)
        
        obs_arr, state, reward, done, info = jitted_step(rng, state, action, params)
        total_reward += jnp.mean(reward)
        
        if jnp.any(done):
            rng, reset_key = jax.random.split(rng)
            obs_arr, state = env.reset(reset_key, params)
            
    print(f"Exact Mean Reward over 1000 steps: {total_reward / 1000}")

if __name__ == "__main__":
    main()
