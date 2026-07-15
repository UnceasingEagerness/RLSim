import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
from flax.training import checkpoints
from algorithms.pe_flax_sac import PursuerActor, EvaderActor
from env.jax_pe_env import JaxPursuitEvasionEnv, PEEnvParams

def test_actions():
    p_actor = PursuerActor()
    e_actor = EvaderActor()
    
    dummy_obs = jnp.zeros((1, 10, 77))
    rng = jax.random.PRNGKey(0)
    p_params = p_actor.init(rng, dummy_obs)["params"]
    e_params = e_actor.init(rng, dummy_obs)["params"]
    
    from flax.training import train_state
    import optax
    p_state = train_state.TrainState.create(apply_fn=p_actor.apply, params=p_params, tx=optax.adam(1e-4))
    e_state = train_state.TrainState.create(apply_fn=e_actor.apply, params=e_params, tx=optax.adam(1e-4))
    
    p_state = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath("checkpoints_pe/pursuer_actor"), target=p_state)
    e_state = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath("checkpoints_pe/evader_actor"), target=e_state)
    
    env = JaxPursuitEvasionEnv()
    params = PEEnvParams()
    
    obs, state = env.reset(jax.random.PRNGKey(42), params)
    
    print("Initial Obs (Agent 0):", obs[0, :5]) # Print some obs features
    
    # Expand to history buffer
    obs_history = jnp.repeat(jnp.expand_dims(obs, axis=1), 10, axis=1)
    
    # Test Pursuer Action
    p_means, p_logstds = p_actor.apply({"params": p_state.params}, jnp.expand_dims(obs_history[0], 0))
    p_action = jnp.tanh(p_means)[0]
    
    # Test Evader Action
    e_means, e_logstds = e_actor.apply({"params": e_state.params}, jnp.expand_dims(obs_history[3], 0))
    e_action = jnp.tanh(e_means)[0]
    
    print("Pursuer 0 Mean:", p_means[0])
    print("Pursuer 0 Action (tanh):", p_action)
    print("Evader Mean:", e_means[0])
    print("Evader Action (tanh):", e_action)
    
    # Try multiple random observations
    for i in range(5):
        random_obs = jax.random.normal(jax.random.PRNGKey(100 + i), (1, 10, 77))
        m, _ = p_actor.apply({"params": p_state.params}, random_obs)
        print(f"Random Obs {i} Action:", jnp.tanh(m)[0])

if __name__ == "__main__":
    test_actions()
