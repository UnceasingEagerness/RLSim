import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs

# Rich for beautiful logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import custom JAX modules
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

console = Console()

def get_feature_name(idx, num_agents):
    """Maps the 92-dimensional observation index to a human-readable feature name."""
    if idx == 0: return "Ego: sin(yaw)"
    if idx == 1: return "Ego: cos(yaw)"
    if idx == 2: return "Ego: u (surge velocity)"
    if idx == 3: return "Ego: v (sway velocity)"
    if idx == 4: return "Ego: r (yaw rate)"
    if idx == 5: return "Goal: Distance"
    if idx == 6: return "Goal: sin(relative angle)"
    if idx == 7: return "Goal: cos(relative angle)"
    
    if 8 <= idx < 72:
        ray_idx = idx - 8
        # Ray 0 is directly behind, Ray 32 is directly ahead
        angle_deg = (ray_idx / 64.0) * 360.0 - 180.0
        return f"LiDAR Ray {ray_idx} ({angle_deg:.1f} deg)"
        
    if idx >= 72:
        neighbor_idx = (idx - 72) // 5
        feat_idx = (idx - 72) % 5
        feats = ["Active", "Relative X", "Relative Y", "Relative Vel X", "Relative Vel Y"]
        return f"Neighbor {neighbor_idx+1}: {feats[feat_idx]}"
        
    return f"Unknown Feature {idx}"

def main():
    console.print(Panel.fit("[bold cyan]RLSim V2 Saliency Evaluator[/bold cyan]\n[italic]Analyzing Neural Network Decision Making via Jacobians[/italic]", border_style="cyan"))
    
    # ── Configurations ────────────────────────────────────────────────────────
    num_agents = 5
    obs_dim = 72 + (num_agents - 1) * 5
    action_dim = 2
    
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (num_agents - 1) * 5, "count": num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    env = JaxUSVEnv()
    env_params = env.default_params.replace(num_agents=num_agents)
    
    map_rng = jax.random.PRNGKey(42)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank = jitted_map_gen(
        map_rng, 
        int(env_params.num_agents), 
        int(env_params.num_obstacles), 
        float(env_params.map_size), 
        int(env_params.map_bank_size)
    )
    env_params = env_params.replace(goals_bank=goals_bank, obstacles_bank=obstacles_bank)
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # ── Load Model Weights ────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath("checkpoints/sac_actor_final")
    if not os.path.exists(ckpt_dir):
        console.print(f"[bold red]Error: Checkpoint '{ckpt_dir}' not found![/bold red]")
        console.print("Please run `python3 train_pure_jax.py` first to generate the trained model.")
        return
        
    console.print("[yellow]Loading weights from Orbax checkpoint...[/yellow]")
    ckpt = PyTreeCheckpointer()
    # To load the params, we just need a dummy structure of the same shape
    dummy_obs = jnp.zeros((1, obs_dim))
    rng = jax.random.PRNGKey(0)
    init_params = actor.init(rng, dummy_obs)["params"]
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    console.print("[bold green]Weights loaded successfully![/bold green]")
    
    # ── Integrated Gradients Setup ────────────────────────────────────────────
    def get_mean_action(params, obs):
        mean, _ = actor.apply({"params": params}, jnp.expand_dims(obs, 0))
        return mean[0]
        
    compute_gradients = jax.jacobian(get_mean_action, argnums=1)
    
    steps = 50
    alphas = jnp.linspace(0.0, 1.0, steps)
    
    def compute_integrated_gradients(params, obs):
        baseline = jnp.zeros_like(obs)
        diff = obs - baseline
        
        def step_fn(alpha):
            interp_obs = baseline + alpha * diff
            return compute_gradients(params, interp_obs)
            
        # [steps, 2, 92]
        grads = jax.vmap(step_fn)(alphas)
        avg_grads = jnp.mean(grads, axis=0)
        
        # Element-wise multiply by diff [2, 92]
        ig = avg_grads * diff
        return ig
        
    compute_saliency = jax.jit(compute_integrated_gradients)
    
    # ── Run Evaluation Episode ────────────────────────────────────────────────
    rng, reset_key = jax.random.split(rng)
    # We will just evaluate a single environment
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    # Initialize 1 Environment with N agents
    reset_keys = jax.random.split(reset_key, 1)
    obs_batch, env_state = vmap_reset(reset_keys, env_params)
    
    max_eval_steps = 200
    saliency_log = []
    
    console.print(f"[bold magenta]Starting Simulation ({max_eval_steps} steps)...[/bold magenta]")
    
    for step in range(max_eval_steps):
        # We will analyze Ego Agent 0
        ego_obs = obs_batch[0, 0, :] # [Obs_Dim]
        
        # 1. Compute Action
        ego_action = get_mean_action(actor_params, ego_obs)
        
        # 2. Compute Saliency (dAction / dObs)
        # Returns a matrix of shape [2, 92]
        jacobian_matrix = compute_saliency(actor_params, ego_obs)
        
        throttle_grads = np.abs(jacobian_matrix[0, :])
        steering_grads = np.abs(jacobian_matrix[1, :])
        
        # Find the top 3 most influential features for this step
        top_throttle_idx = np.argsort(throttle_grads)[-3:][::-1]
        top_steering_idx = np.argsort(steering_grads)[-3:][::-1]
        
        # Log the step
        step_data = {
            "Step": step,
            "Throttle Action": float(ego_action[0]),
            "Steering Action": float(ego_action[1]),
            "Top Throttle Factor 1": get_feature_name(top_throttle_idx[0], num_agents),
            "Top Throttle Factor 2": get_feature_name(top_throttle_idx[1], num_agents),
            "Top Throttle Factor 3": get_feature_name(top_throttle_idx[2], num_agents),
            "Top Steering Factor 1": get_feature_name(top_steering_idx[0], num_agents),
            "Top Steering Factor 2": get_feature_name(top_steering_idx[1], num_agents),
            "Top Steering Factor 3": get_feature_name(top_steering_idx[2], num_agents)
        }
        saliency_log.append(step_data)
        
        # Step the environment (we just duplicate the action for all agents for the test)
        # In reality, we would infer actions for all agents
        batched_actions = jnp.repeat(jnp.expand_dims(ego_action, axis=(0, 1)), num_agents, axis=1)
        step_keys = jax.random.split(jax.random.PRNGKey(step), 1)
        obs_batch, env_state, _, done, _ = vmap_step(step_keys, env_state, batched_actions, env_params)
        
        if jnp.any(done):
            console.print(f"[yellow]Collision or Goal reached at step {step}![/yellow]")
            break
            
    # ── Save Results ──────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    df = pd.DataFrame(saliency_log)
    df.to_csv("logs/saliency_analysis.csv", index=False)
    
    console.print("\n[bold green]Saliency Evaluation Complete![/bold green]")
    console.print("Saved detailed breakdown to [bold]logs/saliency_analysis.csv[/bold]")
    
    # Print a sample of the first few steps
    sample_table = Table(title="Saliency Snapshot (First 5 Steps)", show_header=True)
    sample_table.add_column("Step")
    sample_table.add_column("Primary Driver (Throttle)", style="cyan")
    sample_table.add_column("Primary Driver (Steering)", style="magenta")
    
    for i in range(min(5, len(saliency_log))):
        sample_table.add_row(
            str(saliency_log[i]["Step"]),
            saliency_log[i]["Top Throttle Factor 1"],
            saliency_log[i]["Top Steering Factor 1"]
        )
    console.print(sample_table)
    
if __name__ == "__main__":
    main()
