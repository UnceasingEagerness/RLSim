import os
import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs

# Import our custom modules
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

console = Console()

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    console.print("[bold cyan]RLSim V2: Deep Neuron Activation Mapper[/bold cyan]")
    
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
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # ── Load Model ────────────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath("checkpoints/sac_actor_final")
    if not os.path.exists(ckpt_dir):
        console.print(f"[red]Error: Checkpoint {ckpt_dir} not found![/red]")
        return
        
    ckpt = PyTreeCheckpointer()
    dummy_obs = jnp.zeros((1, obs_dim))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    console.print("[green]✔ Neural Network loaded.[/green]")
    
    # ── Get a Sample Observation ──────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    obs_batch, _ = env.reset(rng, env_params)
    
    # We will analyze the very first agent's brain
    single_obs = obs_batch[0:1] # [1, Obs_Dim]
    
    # ── Capture Intermediates ─────────────────────────────────────────────────
    console.print("Running forward pass and capturing all hidden neuron states...")
    
    # By passing capture_intermediates=True, Flax will record the output of every single module
    _, state = actor.apply({"params": actor_params}, single_obs, capture_intermediates=True)
    
    intermediates = state.get('intermediates', {})
    flat_intermediates = flatten_dict(intermediates)
    
    # ── Log and Print Analysis ───────────────────────────────────────────────
    table = Table(title="Neural Network Hidden Layer Activations (Snapshot)")
    table.add_column("Layer Name", style="cyan")
    table.add_column("Shape", style="magenta")
    table.add_column("Mean Act", justify="right")
    table.add_column("Max Act", justify="right")
    table.add_column("Top Firing Neuron Index", justify="right", style="yellow")
    
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/activation_analysis.txt"
    
    with open(log_path, "w") as f:
        f.write("=== RLSim V2 Deep Neuron Activation Log ===\n")
        f.write("Analyzed Agent 0 at Step 0\n\n")
        
        for layer_name, output in flat_intermediates.items():
            # Outputs might be tuples if a module returned multiple things. Usually a tuple of length 1.
            if isinstance(output, tuple):
                output = output[0]
                
            val = np.array(output)
            
            # We only care about 2D activations [Batch, Features]
            if val.ndim == 2:
                features = val[0] # Drop batch dim
                mean_val = np.mean(features)
                max_val = np.max(features)
                top_idx = np.argmax(features)
                
                table.add_row(
                    layer_name,
                    str(features.shape),
                    f"{mean_val:.3f}",
                    f"{max_val:.3f}",
                    f"#{top_idx}"
                )
                
                f.write(f"Layer: {layer_name}\n")
                f.write(f"Shape: {features.shape}\n")
                f.write(f"Mean: {mean_val:.3f} | Max: {max_val:.3f} | Min: {np.min(features):.3f}\n")
                
                # Print the top 5 most highly activated neurons in this layer
                top_5_indices = np.argsort(features)[-5:][::-1]
                f.write(f"Top 5 Firing Neurons:\n")
                for idx in top_5_indices:
                    f.write(f"  Neuron {idx}: {features[idx]:.3f}\n")
                f.write("-" * 50 + "\n")
                
    console.print(table)
    console.print(f"[bold green]✔ Full detailed dump saved to {log_path}[/bold green]")

if __name__ == "__main__":
    main()
