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
import orbax.checkpoint as ocp

# Rich for beautiful logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import print as rprint

# Import custom JAX modules
from env.jax_usv_env import JaxUSVEnv, EnvParams, EnvState
from algorithms.flax_sac import Actor, SoftQNetwork, CentralizedSoftQNetwork
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
    # ── MODE TOGGLE ───────────────────────────────────────────────────
    #   False  →  Standard nav (independent goals, existing behaviour)
    #   True   →  Encirclement task (agents orbit a static target at d_E=80m)
    ENCIRCLE_MODE = True
    # ─────────────────────────────────────────────────────
    # Configurations
    num_envs = 32             
    num_agents = 5             # 5 Agents interacting simultaneously per environment!
    # Fine-tuning run: 400k steps. It will auto-load the 600k checkpoint
    # and train further to improve spreading and orbiting speed.
    total_timesteps = 400_000
    learning_starts = 5_000 if ENCIRCLE_MODE else 10_000   
    actor_freeze_steps = 15_000 if ENCIRCLE_MODE else 0
    batch_size = 256
    buffer_size = 100_000    # 2 Million capacity for massive swarms
    gamma = 0.99
    tau_target = 0.005
    policy_lr = 3e-4
    q_lr = 3e-4
    target_entropy = -2.0 
    
    # 8 Kinematic + 64 LiDAR + (5-1)*5 Neighbor DeepSets
    seq_len = 10
    base_obs_dim = 72 + (num_agents - 1) * 5
    obs_dim = base_obs_dim * seq_len
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

    # ── Encirclement mode overrides ─────────────────────────────────
    # These lines are only active when ENCIRCLE_MODE=True above.
    # All nav training is completely unchanged when ENCIRCLE_MODE=False.
    if ENCIRCLE_MODE:
        env_params = env_params.replace(
            encircle_mode           = True,
            encircle_radius         = 80.0,   # d_E from paper
            orbit_lead_angle        = 0.4,    # kept for record; unused in Option B (paper-faithful)
            max_steps               = 3000,   # INCREASED: Give them more time to reach the target
            map_size                = 800.0,  # DECREASED: 3000m was way too massive for early training!
            formation_reward_scale  = 3.0,    # Increased from 1.0 to force agents to spread out more
            orbit_reward_scale      = 3.0,    # Increased from 2.0 to force faster CCW circling
            goal_radius             = 30.0,   # larger on-ring tolerance (vs 15m for nav)
            num_obstacles           = 30,     # reduced from 200 — Stage C Curriculum Constraint
            # ── Option A: Moving evader (random walk, matches MAPPO paper §IV-A) ──
            moving_target           = True,   # evader moves randomly each step
            target_speed            = 1.0,    # 1 m/s random walk speed (MAPPO paper value)
        )
    # ─────────────────────────────────────────────────────
    
    print_rich_config(total_timesteps, num_envs, num_agents, batch_size, obs_dim, action_dim, policy_lr, env_params)
    
    rng = jax.random.PRNGKey(42)
    
    # ── Pre-compute the O(1) Map Bank ──────────────────────────────────────────
    console.print("[bold yellow]Pre-computing Map Bank (1000 Maps) on GPU...[/bold yellow]")
    rng, map_key = jax.random.split(rng)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank = jitted_map_gen(
        map_key, 
        int(env_params.num_agents), 
        int(env_params.num_obstacles), 
        float(env_params.map_size), 
        int(env_params.map_bank_size)
    )
    jax.block_until_ready(goals_bank)
    
    env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank)
    console.print("[bold green]✔ Map Bank successfully loaded into memory![/bold green]")
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    critic = CentralizedSoftQNetwork()
    buffer = JaxReplayBuffer(buffer_size // num_agents, num_agents, obs_dim, action_dim)
    
    # Vectorized Environment Methods (Vectorize over Envs, the Env itself is already multi-agent)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    # ── Initialize State ──────────────────────────────────────────────────────
    rng, _rng = jax.random.split(rng)
    reset_keys = jax.random.split(_rng, num_envs)
    init_obs, init_env_state = vmap_reset(reset_keys, env_params) # [E, N, Obs_Dim]
    
    dummy_obs_actor = jnp.zeros((1, obs_dim))
    dummy_obs_critic = jnp.zeros((1, num_agents, obs_dim))
    dummy_act_critic = jnp.zeros((1, num_agents, action_dim))
    
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    actor_params = actor.init(actor_key, dummy_obs_actor)["params"]
    critic_params = critic.init(critic_key, dummy_obs_critic, dummy_act_critic)["params"]
    
    # ── Checkpoint & log directories (mode-aware) ────────────────────────────
    if ENCIRCLE_MODE:
        ckpt_dir = "checkpoints_encircle_ctde"
        log_dir  = "logs_encircle_ctde"
    else:
        ckpt_dir = "checkpoints_max"
        log_dir  = "logs_max"

    # --- Auto-Resume / Warm-Start Logic ---
    # Encircle mode:  try checkpoints_encircle (resume) → fall back to checkpoints_max (warm-start)
    # Nav mode:       try checkpoints_max (resume) → start fresh
    actor_path  = os.path.abspath(f"{ckpt_dir}/sac_actor_final")
    critic_path = os.path.abspath(f"{ckpt_dir}/sac_critic_final")
    warmstart_actor  = os.path.abspath("checkpoints_max/sac_actor_final")
    warmstart_critic = os.path.abspath("checkpoints_max/sac_critic_final")
    ckpt = ocp.StandardCheckpointer()
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        console.print(f"[bold yellow]Resuming from {ckpt_dir}...[/bold yellow]")
        actor_params = ckpt.restore(actor_path, target=actor_params)
        critic_params = ckpt.restore(critic_path, target=critic_params)
    elif ENCIRCLE_MODE and os.path.exists(warmstart_actor) and os.path.exists(warmstart_critic):
        console.print("[bold cyan]No encircle checkpoint found — warm-starting from checkpoints_max/[/bold cyan]")
        actor_params  = ckpt.restore(warmstart_actor, target=actor_params)
        console.print("[bold yellow]NOTE: CTDE enabled. Skipping Critic warm-start (incompatible shapes). Initializing Critic from scratch![/bold yellow]")
        # We do NOT load warmstart_critic because checkpoints_max contains an ISAC critic, not a CTDE critic!
    else:
        console.print("[bold yellow]No checkpoint found — training from scratch.[/bold yellow]")
    
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=policy_lr)))
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=q_lr)))
    
    log_alpha = jnp.array(0.5)  # Start with higher entropy = more exploration
    alpha_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=policy_lr))
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
            
        if ENCIRCLE_MODE:
            # Warm-started actor knows how to drive — use its stochastic policy for rollout!
            flat_action = exploit_fn()
        else:
            flat_action = jax.lax.cond(runner_state.step_count < learning_starts, explore_fn, exploit_fn)
            
        action = flat_action.reshape(num_envs, num_agents, action_dim)
        
        # Compute dynamic Curriculum Progress (freeze at 250,000 steps)
        p = jnp.clip(runner_state.step_count / 250000.0, 0.0, 1.0)
        current_env_params = env_params.replace(global_progress=p)
        
        # 2. Vectorized Environment Step
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_state, reward, done, info = vmap_step(step_keys, runner_state.env_state, action, current_env_params)
        new_episode_return = runner_state.episode_return + reward
        
        # 3. Replay Buffer Insertion (CTDE stores JOINT transitions)
        # We do NOT flatten the N dimension anymore.
        # obs is [E, N, obs_dim], action is [E, N, action_dim], etc.
        new_buffer_state = buffer.add_batch(
            runner_state.buffer_state,
            runner_state.obs, action, reward, next_obs, done,
            num_envs
        )
        
        # 4. Vectorized Environment Reset (If ANY agent in the env is done, we reset that specific ENV)
        # Note: In Multi-Agent CTDE, we typically reset the whole env when max_steps is reached, 
        # but to keep it simple, if ANY agent crashes, we reset the env.
        env_done = jnp.any(done, axis=1) # [E]
        
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_obs, reset_state = vmap_reset(reset_keys, current_env_params)
        
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
            
            # [Fix 7] Freeze Actor to let Critic catch up. We compute the actor update anyway to get log_prob for alpha, 
            # but throw away the new_actor parameters if we are in the freeze window.
            new_actor_cand, a_loss, log_prob = update_actor(runner_state.actor_state, new_critic, runner_state.log_alpha, batch.obs, key_actor)
            
            new_actor = jax.lax.cond(
                runner_state.step_count >= actor_freeze_steps,
                lambda _: new_actor_cand,
                lambda _: runner_state.actor_state,
                operand=None
            )
            
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
            "alpha": jnp.exp(new_log_alpha),
            "k_progress": jnp.mean(info.get("k_progress", jnp.zeros_like(reward))),
            "k_form": jnp.mean(info.get("k_form", jnp.zeros_like(reward))),
            "k_cap": jnp.mean(info.get("k_cap", jnp.zeros_like(reward))),
            "p": p,
            "capture_rate": jnp.mean(info.get("encircled", jnp.zeros_like(reward)).astype(jnp.float32)),
            "encircle_ratio": jnp.mean(info.get("encircle_ratio", jnp.zeros_like(reward)).astype(jnp.float32)),
            "collision_rate": jnp.mean(info.get("collision", jnp.zeros_like(reward)).astype(jnp.float32)),
            "gate_active": jnp.mean(info.get("gate_active", jnp.zeros_like(reward)).astype(jnp.float32)),
            "agent_dist_mean": jnp.mean(info.get("min_agent_dist", jnp.zeros_like(reward))),
            "agent_dist_min": jnp.min(info.get("min_agent_dist", jnp.zeros_like(reward))),
            "target_dist": jnp.mean(info.get("dist_to_goal", jnp.zeros_like(reward))),
            "max_gap": jnp.mean(info.get("max_escape_gap", jnp.zeros_like(reward))),
            "radius_mean": jnp.mean(info.get("radius_mean", jnp.zeros_like(reward))),
            "radius_std": jnp.mean(info.get("radius_std", jnp.zeros_like(reward)))
        }
        
        return new_runner_state, metrics

    # ── JIT Compile the inner scan ───────────────────────────────────────────
    steps_per_epoch = 10_000
    num_epochs = total_timesteps // steps_per_epoch

    with console.status("[bold cyan]Compiling massive JAX XLA Graph...[/bold cyan]", spinner="dots"):
        @jax.jit
        def run_epoch(runner_state):
            final_state, epoch_metrics = jax.lax.scan(_step_fn, runner_state, None, length=steps_per_epoch)
            return final_state, epoch_metrics
        
    start_time = time.time()
    all_metrics = []
    
    console.print("\n[bold green]Starting Training Loop...[/bold green]")
    try:
        for epoch in range(num_epochs):
            current_step = (epoch + 1) * steps_per_epoch
            
            # Checkpoint saving every 1 epoch (10,000 steps)
            checkpointer = ocp.StandardCheckpointer()
            actor_save  = os.path.abspath(f"{ckpt_dir}/step_{current_step}/sac_actor")
            critic_save = os.path.abspath(f"{ckpt_dir}/step_{current_step}/sac_critic")
            checkpointer.save(actor_save,  runner_state.actor_state.params,  force=True)
            checkpointer.save(critic_save, runner_state.critic_state.params, force=True)
            
            epoch_start = time.time()
            runner_state, epoch_metrics = run_epoch(runner_state)
            jax.block_until_ready(epoch_metrics["reward"])
            epoch_end = time.time()
            
            mean_reward = np.mean(epoch_metrics["reward"])
            mean_a_loss = np.mean(epoch_metrics["a_loss"])
            
            # Calculate speed for this epoch
            sps = (steps_per_epoch * num_envs * num_agents) / (epoch_end - epoch_start)
            
            console.print(f"Epoch {epoch+1:02d}/{num_epochs} | Step {current_step:>7,} | "
                          f"Reward: {mean_reward:>7.2f} | Actor Loss: {mean_a_loss:>7.4f} | "
                          f"Speed: {sps:,.0f} SPS\n"
                          f"          ↳ p: {np.mean(epoch_metrics['p']):.2f} | "
                          f"k(P:{np.mean(epoch_metrics['k_progress']):.1f}, F:{np.mean(epoch_metrics['k_form']):.1f}, C:{np.mean(epoch_metrics['k_cap']):.1f}) | "
                          f"Gate: {np.mean(epoch_metrics['gate_active'])*100:.1f}% | "
                          f"Enc: {np.mean(epoch_metrics['encircle_ratio'])*100:.1f}% | "
                          f"Cap: {np.mean(epoch_metrics['capture_rate'])*100:.1f}% | "
                          f"Coll: {np.mean(epoch_metrics['collision_rate'])*100:.1f}% | "
                          f"Gap: {np.mean(epoch_metrics['max_gap']):.2f} (Idl:1.26) | "
                          f"RadStd: {np.mean(epoch_metrics['radius_std']):.1f}m | "
                          f"MinDist: {np.min(epoch_metrics['agent_dist_min']):.1f}m")
                          
            all_metrics.append(epoch_metrics)
    except KeyboardInterrupt:
        console.print("\n[bold red]Ctrl+C Detected! Gracefully saving checkpoint before exiting...[/bold red]")
        
    end_time = time.time()
    final_state = runner_state
    
    # Concatenate metrics across all epochs for final saving
    metrics = {k: jnp.concatenate([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        
    duration = end_time - start_time
    total_simulated_steps = total_timesteps * num_envs * num_agents
    avg_sps = total_simulated_steps / duration
    
    console.print(Panel.fit(f"[bold green]Execution Complete![/bold green]\n"
                            f"Simulated {total_simulated_steps:,} transitions in [bold white]{duration:.2f}s[/bold white]\n"
                            f"Avg Speed: [bold magenta]{avg_sps:,.0f} SPS[/bold magenta]", border_style="green"))
    
    # ── Save Checkpoints & Metrics ────────────────────────────────────────────
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    with console.status("[bold yellow]Saving Model Checkpoint and Metrics to disk...[/bold yellow]"):
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            "step": np.arange(len(metrics["reward"])),
            "mean_reward": np.array(metrics["reward"]),
            "mean_episode_return": np.array(metrics["episode_return"]),
            "env_done_rate": np.array(metrics["env_done"]),
            "q_loss": np.array(metrics["q_loss"]),
            "a_loss": np.array(metrics["a_loss"]),
            "alpha": np.array(metrics["alpha"])
        })
        metrics_df.to_csv(f"{log_dir}/metrics.csv", index=False)
        
        # Save Network Weights
        ckpt = ocp.StandardCheckpointer()
        actor_save  = os.path.abspath(f"{ckpt_dir}/sac_actor_final")
        critic_save = os.path.abspath(f"{ckpt_dir}/sac_critic_final")
        ckpt.save(actor_save,  final_state.actor_state.params,  force=True)
        ckpt.save(critic_save, final_state.critic_state.params, force=True)
        
    console.print(f"[bold green]✔ Saved metrics to {log_dir}/metrics.csv[/bold green]")
    console.print(f"[bold green]✔ Saved model weights to {ckpt_dir}/[/bold green]")
    
    res_table = Table(title="Training Summaries", show_header=True, header_style="bold blue")
    res_table.add_column("Outer Step", justify="right")
    res_table.add_column("Mean Batch Reward", justify="right", style="green")
    res_table.add_column("Actor Loss", justify="right", style="dim")
    
    # Print every 50,000 outer steps
    for step in range(0, len(metrics_df), 50_000):
        if step == 0: continue
        res_table.add_row(f"{step:,}", f"{metrics_df['mean_reward'][step]:.4f}", f"{metrics_df['a_loss'][step]:.4f}")
        
    res_table.add_row(f"{len(metrics_df)-1:,}", f"{metrics_df['mean_reward'].iloc[-1]:.4f}", f"{metrics_df['a_loss'].iloc[-1]:.4f}")
    console.print(res_table)

if __name__ == "__main__":
    main()
