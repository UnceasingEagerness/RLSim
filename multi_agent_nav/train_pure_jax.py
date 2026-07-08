import os
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
from flax.training.train_state import TrainState
from flax import struct
from typing import Any, Tuple, Dict
from orbax.checkpoint import PyTreeCheckpointer

# Rich for beautiful logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import print as rprint

# Import custom JAX modules
from env.jax_usv_env import JaxUSVEnv, EnvParams, EnvState
from algorithms.flax_sac import Actor, SoftQNetwork
from algorithms.jax_buffer import JaxReplayBuffer, ReplayBufferState
from algorithms.sac_update import update_critic, update_actor, update_alpha, Transition

console = Console()

@struct.dataclass
class RunnerState:
    env_state: EnvState
    obs: jnp.ndarray
    episode_return: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    target_critic_params: Any
    log_alpha: jnp.ndarray
    alpha_opt_state: optax.OptState
    buffer_state: ReplayBufferState
    rng: jax.random.PRNGKey
    step_count: int

def print_rich_config(total_timesteps, num_envs, num_agents, batch_size, obs_dim, action_dim, policy_lr, env_params):
    console.print(Panel.fit("[bold cyan]RLSim V2 (Massive Multi-Agent Swarm)[/bold cyan]", border_style="cyan"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim", width=15)
    table.add_column("Parameter", width=25)
    table.add_column("Value", justify="right", style="green")
    
    # Engine
    table.add_row("Engine", "Hardware Accelerator", str(jax.devices()[0]))
    table.add_row("Engine", "JIT Mode", "jax.lax.scan + vmap (Double)")
    table.add_row("Engine", "Outer Timesteps", f"{total_timesteps:,}")
    table.add_row("Engine", "Parallel Envs (E)", str(num_envs))
    table.add_row("Engine", "Agents per Env (N)", str(num_agents))
    table.add_row("Engine", "Transitions/Step", str(num_envs * num_agents))
    
    # Environment
    table.add_section()
    table.add_row("Environment", "Max Steps per Ep", str(env_params.max_steps))
    table.add_row("Environment", "Map Size", f"{env_params.map_size}m")
    table.add_row("Environment", "Collision Radius", f"{env_params.collision_radius}m")
    
    # Agent
    table.add_section()
    table.add_row("Agent (SAC)", "Observation Dim", str(obs_dim))
    table.add_row("Agent (SAC)", "Action Dim", str(action_dim))
    table.add_row("Agent (SAC)", "Policy LR", str(policy_lr))
    
    console.print(table)
    console.print("\n[bold yellow]Initializing Replay Buffer on GPU (Zero-Copy)...[/bold yellow]")

def main():
    # Configurations
    num_envs = 128             
    num_agents = 5             # 5 Agents interacting simultaneously per environment!
    total_timesteps = 400_000
    learning_starts = 10_000   
    batch_size = 256
    buffer_size = 2_000_000    # 2 Million capacity for massive swarms
    gamma = 0.99
    tau_target = 0.005
    policy_lr = 3e-4
    q_lr = 3e-4
    target_entropy = -2.0 
    
    # 8 Kinematic + 64 LiDAR + (5-1)*5 Neighbor DeepSets
    obs_dim = 72 + (num_agents - 1) * 5
    action_dim = 2
    total_insertions_per_step = num_envs * num_agents
    
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (num_agents - 1) * 5, "count": num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    env = JaxUSVEnv()
    # Use defaults from EnvParams (max_steps=2000, map_size=300) — do NOT override here
    env_params = env.default_params.replace(num_agents=num_agents)
    
    print_rich_config(total_timesteps, num_envs, num_agents, batch_size, obs_dim, action_dim, policy_lr, env_params)
    
    rng = jax.random.PRNGKey(42)
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    critic = SoftQNetwork(layout=layout)
    buffer = JaxReplayBuffer(buffer_size, obs_dim, action_dim)
    
    # Vectorized Environment Methods (Vectorize over Envs, the Env itself is already multi-agent)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    # ── Initialize State ──────────────────────────────────────────────────────
    rng, _rng = jax.random.split(rng)
    reset_keys = jax.random.split(_rng, num_envs)
    init_obs, init_env_state = vmap_reset(reset_keys, env_params) # [E, N, Obs_Dim]
    
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, action_dim))
    
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    actor_params = actor.init(actor_key, dummy_obs)["params"]
    critic_params = critic.init(critic_key, dummy_obs, dummy_act)["params"]
    
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate=policy_lr))
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(learning_rate=q_lr))
    
    log_alpha = jnp.array(0.5)  # Start with higher entropy = more exploration
    alpha_optimizer = optax.adam(learning_rate=policy_lr)
    alpha_opt_state = alpha_optimizer.init(log_alpha)
    
    runner_state = RunnerState(
        env_state=init_env_state,
        obs=init_obs, 
        episode_return=jnp.zeros((num_envs, num_agents)),
        actor_state=actor_state,
        critic_state=critic_state,
        target_critic_params=critic_params,
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
        buffer_state=buffer.init_state(),
        rng=rng,
        step_count=0
    )
    
    # ── Pure JAX Scan Loop ────────────────────────────────────────────────────
    def _step_fn(runner_state: RunnerState, _):
        rng, action_key, step_key, sample_key, update_key, reset_key = jax.random.split(runner_state.rng, 6)
        
        # Flatten observations for 2D Actor inference
        flat_obs = runner_state.obs.reshape(num_envs * num_agents, obs_dim)
        
        # 1. Action Selection (Batched)
        def explore_fn():
            return jax.random.uniform(action_key, shape=(num_envs * num_agents, action_dim), minval=-1.0, maxval=1.0)
        
        def exploit_fn():
            action, _ = actor.apply({"params": runner_state.actor_state.params}, flat_obs, action_key, method=actor.get_action)
            return action
            
        flat_action = jax.lax.cond(runner_state.step_count < learning_starts, explore_fn, exploit_fn)
        action = flat_action.reshape(num_envs, num_agents, action_dim)
        
        # 2. Vectorized Environment Step
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_state, reward, done, info = vmap_step(step_keys, runner_state.env_state, action, env_params)
        new_episode_return = runner_state.episode_return + reward
        
        # 3. Replay Buffer Insertion
        # CRITICAL: store next_obs BEFORE environment reset.
        # If we stored reset_obs as next_obs for terminal transitions, the critic
        # would learn Q(s', a') using the initial state of the NEXT episode,
        # which is wrong. The done flag already masks out the bootstrap term.
        flat_reward   = reward.flatten()                        # [E*N]
        flat_next_obs = next_obs.reshape(-1, obs_dim)          # [E*N, obs_dim] — true terminal obs
        flat_done     = done.flatten()                         # [E*N]

        new_buffer_state = buffer.add_batch(
            runner_state.buffer_state,
            flat_obs, flat_action, flat_reward, flat_next_obs, flat_done,
            total_insertions_per_step
        )
        
        # 4. Vectorized Environment Reset (If ANY agent in the env is done, we reset that specific ENV)
        # Note: In Multi-Agent CTDE, we typically reset the whole env when max_steps is reached, 
        # but to keep it simple, if ANY agent crashes, we reset the env.
        env_done = jnp.any(done, axis=1) # [E]
        
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_obs, reset_state = vmap_reset(reset_keys, env_params)
        
        # Merge observations
        final_obs = jnp.where(env_done[:, None, None], reset_obs, next_obs)
        final_episode_return = jnp.where(env_done[:, None], 0.0, new_episode_return)
        
        def merge_states(reset_val, next_val):
            shape = (num_envs,) + (1,) * (next_val.ndim - 1)
            return jnp.where(jnp.reshape(env_done, shape), reset_val, next_val)
            
        final_env_state = jax.tree_util.tree_map(merge_states, reset_state, next_env_state)
        
        # 5. Network Updates
        def perform_update():
            b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(new_buffer_state, sample_key, batch_size)
            batch = Transition(obs=b_obs, action=b_act, reward=b_rew, next_obs=b_next_obs, done=b_done)
            
            key_critic, key_actor = jax.random.split(update_key)
            new_critic, q_loss = update_critic(runner_state.critic_state, runner_state.target_critic_params, runner_state.actor_state, runner_state.log_alpha, batch, gamma, key_critic)
            new_actor, a_loss, log_prob = update_actor(runner_state.actor_state, new_critic, runner_state.log_alpha, batch.obs, key_actor)
            new_log_alpha, new_alpha_opt, alpha_loss = update_alpha(runner_state.log_alpha, runner_state.alpha_opt_state, log_prob, target_entropy, alpha_optimizer)
            
            new_target = jax.tree_util.tree_map(
                lambda t, c: tau_target * c + (1 - tau_target) * t,
                runner_state.target_critic_params, new_critic.params
            )
            # Clamp log_alpha so alpha never falls below exp(-2.0)=0.135 — prevents entropy collapse
            new_log_alpha = jnp.maximum(new_log_alpha, -2.0)
            return new_actor, new_critic, new_target, new_log_alpha, new_alpha_opt, q_loss, a_loss
            
        def skip_update():
            return runner_state.actor_state, runner_state.critic_state, runner_state.target_critic_params, runner_state.log_alpha, runner_state.alpha_opt_state, 0.0, 0.0
            
        new_actor, new_critic, new_target, new_log_alpha, new_alpha_opt, q_loss, a_loss = jax.lax.cond(
            runner_state.step_count >= learning_starts, perform_update, skip_update
        )
        
        # Update Runner State
        new_runner_state = runner_state.replace(
            env_state=final_env_state,
            obs=final_obs,
            episode_return=final_episode_return,
            actor_state=new_actor,
            critic_state=new_critic,
            target_critic_params=new_target,
            log_alpha=new_log_alpha,
            alpha_opt_state=new_alpha_opt,
            buffer_state=new_buffer_state,
            rng=rng,  # ← advance the rng each step (was reusing same key every step!)
            step_count=runner_state.step_count + 1
        )
        
        metrics = {
            "reward": jnp.mean(reward), # Mean across all agents in all envs
            "episode_return": jnp.mean(new_episode_return),
            "env_done": jnp.mean(env_done.astype(jnp.float32)),
            "q_loss": q_loss,
            "a_loss": a_loss,
            "alpha": jnp.exp(new_log_alpha)
        }
        
        return new_runner_state, metrics

    # ── JIT Compile the entire scan ───────────────────────────────────────────
    with console.status("[bold cyan]Compiling massive JAX XLA Graph...[/bold cyan]", spinner="dots"):
        @jax.jit
        def run_training(init_runner_state):
            final_state, metrics = jax.lax.scan(_step_fn, init_runner_state, None, length=total_timesteps)
            return final_state, metrics
        
        start_time = time.time()
        final_state, metrics = run_training(runner_state)
        jax.block_until_ready(metrics["reward"])
        end_time = time.time()
        
    duration = end_time - start_time
    total_simulated_steps = total_timesteps * num_envs * num_agents
    sps = total_simulated_steps / duration
    
    console.print(Panel.fit(f"[bold green]Execution Complete![/bold green]\n"
                            f"Simulated {total_simulated_steps:,} transitions in [bold white]{duration:.2f}s[/bold white]\n"
                            f"Speed: [bold magenta]{sps:,.0f} SPS[/bold magenta]", border_style="green"))
    
    # ── Save Checkpoints & Metrics ────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    with console.status("[bold yellow]Saving Model Checkpoint and Metrics to disk...[/bold yellow]"):
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            "step": np.arange(total_timesteps),
            "mean_reward": np.array(metrics["reward"]),
            "mean_episode_return": np.array(metrics["episode_return"]),
            "env_done_rate": np.array(metrics["env_done"]),
            "q_loss": np.array(metrics["q_loss"]),
            "a_loss": np.array(metrics["a_loss"]),
            "alpha": np.array(metrics["alpha"])
        })
        metrics_df.to_csv("logs/metrics.csv", index=False)
        
        # Save Network Weights
        ckpt = PyTreeCheckpointer()
        actor_path = os.path.abspath("checkpoints/sac_actor_final")
        critic_path = os.path.abspath("checkpoints/sac_critic_final")
        ckpt.save(actor_path, final_state.actor_state.params, force=True)
        ckpt.save(critic_path, final_state.critic_state.params, force=True)
        
    console.print("[bold green]✔ Saved metrics to logs/metrics.csv[/bold green]")
    console.print("[bold green]✔ Saved model weights to checkpoints/[/bold green]")
    
    res_table = Table(title="Training Summaries", show_header=True, header_style="bold blue")
    res_table.add_column("Outer Step", justify="right")
    res_table.add_column("Mean Batch Reward", justify="right", style="green")
    res_table.add_column("Actor Loss", justify="right", style="dim")
    
    # Print every 50,000 outer steps
    for step in range(0, total_timesteps, 50_000):
        if step == 0: continue
        res_table.add_row(f"{step:,}", f"{metrics_df['mean_reward'][step]:.4f}", f"{metrics_df['a_loss'][step]:.4f}")
        
    res_table.add_row(f"{total_timesteps-1:,}", f"{metrics_df['mean_reward'].iloc[-1]:.4f}", f"{metrics_df['a_loss'].iloc[-1]:.4f}")
    console.print(res_table)

if __name__ == "__main__":
    main()
