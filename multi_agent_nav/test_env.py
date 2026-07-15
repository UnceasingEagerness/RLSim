import jax
import jax.numpy as jnp
from env.jax_usv_env import JaxUSVEnv, EnvParams

def main():
    env = JaxUSVEnv()
    params = env.default_params.replace(
        encircle_mode=True, 
        encircle_radius=80.0,
        num_agents=5
    )
    
    rng = jax.random.PRNGKey(0)
    rng_reset, rng_step = jax.random.split(rng)
    
    print("Generating map bank...")
    goals, obs = env.generate_map_bank(rng_reset, 5, 200, 3000.0, 10)
    params = params.replace(goals_bank=goals, obstacles_bank=obs)
    
    print("Testing Reset...")
    obs, state = env.reset(rng_reset, params)
    print("Reset OK. Obs shape:", obs.shape)
    
    print("JIT Compiling Step function...")
    @jax.jit
    def jitted_step(key, st, act):
        return env.step(key, st, act, params)
    
    dummy_action = jnp.zeros((5, 2))
    
    print("Testing Step...")
    next_obs, next_state, reward, done, info = jitted_step(rng_step, state, dummy_action)
    
    print("Step OK!")
    print("Reward shape:", reward.shape)
    print("Reward values:", reward)
    print("Done shape:", done.shape)
    print("Info keys:", info.keys())
    print("Everything compiled and executed flawlessly.")

if __name__ == "__main__":
    main()
