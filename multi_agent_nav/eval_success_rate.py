import os
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from rich.console import Console

from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

console = Console()

def main():
    console.print("[bold cyan]RLSim V2: Automated Evaluation Suite (100 Episodes)[/bold cyan]")
    
    num_envs = 100
    num_agents = 5
    seq_len = 10
    base_obs_dim = 72 + (num_agents - 1) * 5
    obs_dim = base_obs_dim * seq_len
    action_dim = 2
    
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (num_agents - 1) * 5, "count": num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    env = JaxUSVEnv()
    # Map size 1000m forces goals to spawn 300-500m away
    env_params = env.default_params.replace(num_agents=num_agents, map_size=1000.0, num_obstacles=100, max_steps=2000)
    
    rng = jax.random.PRNGKey(42)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank = jitted_map_gen(
        rng, 
        int(env_params.num_agents), 
        int(env_params.num_obstacles), 
        float(env_params.map_size), 
        int(env_params.map_bank_size)
    )
    env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank)
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    ckpt_dir = os.path.abspath("checkpoint_cnn/sac_actor_final")
    ckpt = PyTreeCheckpointer()
    dummy_obs = jnp.zeros((1, obs_dim))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    def get_action(params, obs):
        mean, _ = actor.apply({"params": params}, obs)
        return jnp.tanh(mean)
        
    jit_action = jax.jit(get_action)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, num_envs)
    obs_batch, state_batch = vmap_reset(reset_keys, env_params)
    
    console.print(f"Running {num_envs} massive parallel simulations...")
    
    # Track results
    env_success = np.zeros(num_envs, dtype=bool)
    env_collision = np.zeros(num_envs, dtype=bool)
    env_collision_obs = np.zeros(num_envs, dtype=bool)
    env_collision_agent = np.zeros(num_envs, dtype=bool)
    env_timeout = np.zeros(num_envs, dtype=bool)
    env_finished = np.zeros(num_envs, dtype=bool)
    
    for step in range(env_params.max_steps):
        if np.all(env_finished):
            break
            
        flat_obs = obs_batch.reshape(num_envs * num_agents, obs_dim)
        actions = jit_action(actor_params, flat_obs)
        batched_actions = actions.reshape(num_envs, num_agents, 2)
        
        step_keys = jax.random.split(jax.random.PRNGKey(step), num_envs)
        obs_batch, state_batch, reward, done, info = vmap_step(step_keys, state_batch, batched_actions, env_params)
        
        # Determine episode end
        env_done = np.array(jnp.any(done, axis=1)) # [100]
        
        # Only process envs that just finished this step
        just_finished = env_done & ~env_finished
        
        if np.any(just_finished):
            # Evaluate why they finished
            reached_goal = np.array(jnp.any(info["reached_goal"], axis=1))
            collision = np.array(jnp.any(info["collision"], axis=1))
            col_obs = np.array(jnp.any(info["collision_obs"], axis=1))
            col_agent = np.array(jnp.any(info["collision_agent"], axis=1))
            timeout = np.array(jnp.any(info["timeout"], axis=1))
            
            # An env is a success if at least one agent reached the goal and NO agent collided
            # (Because they share fate, if any agent collides it's a mission failure)
            success_mask = reached_goal & ~collision
            
            env_success[just_finished & success_mask] = True
            env_collision[just_finished & collision] = True
            env_collision_obs[just_finished & col_obs] = True
            env_collision_agent[just_finished & col_agent] = True
            env_timeout[just_finished & timeout & ~reached_goal & ~collision] = True
            
            env_finished[just_finished] = True
            
    # Calculate stats
    total_success = np.sum(env_success)
    total_collision = np.sum(env_collision)
    total_col_obs = np.sum(env_collision_obs)
    total_col_agent = np.sum(env_collision_agent)
    total_timeout = np.sum(env_timeout)
    
    console.print(f"\n[bold green]Evaluation Complete![/bold green]")
    console.print(f"Total Episodes     : {num_envs}")
    console.print(f"Success Rate       : [bold green]{total_success}%[/bold green]")
    console.print(f"Total Collisions   : [bold red]{total_collision}%[/bold red]")
    console.print(f"  ├─ Hitting Obstacle : [bold yellow]{total_col_obs}%[/bold yellow]")
    console.print(f"  └─ Hitting Teammate : [bold yellow]{total_col_agent}%[/bold yellow]")
    console.print(f"Timeout Rate       : [bold magenta]{total_timeout}%[/bold magenta]")

if __name__ == "__main__":
    main()
