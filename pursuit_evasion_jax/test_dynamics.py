import jax
import jax.numpy as jnp
from env.jax_pe_env import JaxPursuitEvasionEnv, PEEnvParams

def test_dynamics_long():
    env = JaxPursuitEvasionEnv()
    params = PEEnvParams()
    
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, params)
    
    # Random aggressive actions
    rng = jax.random.PRNGKey(100)
    
    @jax.jit
    def step_loop(val):
        state, key = val
        
        def body_fn(i, carry):
            s, k = carry
            k, act_k = jax.random.split(k)
            # Random throttle and steering
            acts = jax.random.uniform(act_k, (4, 2), minval=-1.0, maxval=1.0)
            _, next_s, _, _, _ = env.step(act_k, s, acts, params)
            return next_s, k
            
        return jax.lax.fori_loop(0, 1000, body_fn, (state, key))
        
    final_state, _ = step_loop((state, rng))
    
    print("Final ETA:", final_state.eta)
    print("Has NaNs:", jnp.any(jnp.isnan(final_state.eta)))

if __name__ == "__main__":
    test_dynamics_long()
