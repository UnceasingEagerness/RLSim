import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
import argparse
from rich.console import Console

from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

console = Console()

def load_agent(ckpt_dir, layout, action_dim):
    """Loads the trained actor from an Orbax checkpoint."""
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # Initialize dummy variables to get shapes
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, 920))
    
    actor_params = actor.init(key, dummy_obs)["params"]
    
    # Load from Orbax
    actor_path = os.path.abspath(f"{ckpt_dir}/sac_actor_final")
    if not os.path.exists(actor_path):
        console.print(f"[bold red]Checkpoint not found at {actor_path}[/bold red]")
        return None, None
        
    checkpointer = ocp.StandardCheckpointer()
    actor_params = checkpointer.restore(actor_path, target=actor_params)
    console.print(f"[bold green]✔ Successfully loaded Actor from {actor_path}[/bold green]")
    
    return actor, actor_params

def main():
    parser = argparse.ArgumentParser(description="Offline XAI Framework for Swarm RL")
    parser.add_argument("--ckpt", type=str, default="checkpoints_max", help="Directory containing sac_actor_final")
    parser.add_argument("--mode", type=str, choices=["heat", "lidar", "attention"], default="heat", help="Which XAI module to run")
    args = parser.parse_args()
    
    console.print("[bold cyan]Initializing XAI Visualizer (CPU Mode)...[/bold cyan]")
    
    num_agents = 5
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (num_agents - 1) * 5, "count": num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    action_dim = 2
    
    # 2. Load the trained actor
    actor, actor_params = load_agent(args.ckpt, layout, action_dim)
    if actor is None:
        return
        
    console.print(f"[bold magenta]Starting XAI Mode: {args.mode.upper()}[/bold magenta]")
    
    # 3. Setup Environment for XAI Evaluation
    from env.jax_usv_env import JaxUSVEnv
    env = JaxUSVEnv()
    
    # Generate map bank
    key_bank = jax.random.PRNGKey(99)
    map_bank_size = 10
    goals_bank, obs_bank = JaxUSVEnv.generate_map_bank(key_bank, num_agents, num_obstacles=30, map_size=800.0, map_bank_size=map_bank_size)
    
    env_params = env.default_params.replace(
        num_agents=num_agents,
        encircle_mode=True,
        encircle_radius=80.0,
        map_size=800.0,
        num_obstacles=30,
        goals_bank=goals_bank,
        obstacles_bank=obs_bank,
        map_bank_size=map_bank_size
    )
    
    # 4. Generate a Test Episode
    console.print("[bold cyan]Running Environment Episode for XAI...[/bold cyan]")
    key = jax.random.PRNGKey(42)
    key_reset, key = jax.random.split(key)
    obs, state = env.reset(key_reset, env_params)
    
    history_obs = []
    history_state = []
    history_attention = []
    
    @jax.jit
    def step_fn(key_step, obs, state):
        # Extract Attention
        action, actor_state = actor.apply({"params": actor_params}, obs, mutable=['intermediates'])
        attn_weights = actor_state['intermediates']['STAE_Max_ActorBackbone_0']['FlaxSpatioTemporalAttentionEncoder_0']['MultiHeadDotProductAttention_0']['attention_weights'][0]
        # attn_weights shape: [B, 4, 1, 4]
        
        # We need to process action to ensure it matches environment shape
        # Action is raw from network, usually we sample, but for offline XAI we just take mean
        # However actor.apply returns mean, log_std in default mode without get_action
        # Wait, the default __call__ returns (mean, log_std). Let's use mean.
        mean, _ = action
        action_env = jnp.tanh(mean) * actor.action_scale + actor.action_bias
        
        # Step env
        next_obs, next_state, reward, done, info = env.step(key_step, state, action_env, env_params)
        return next_obs, next_state, attn_weights
        
    for step in range(500):
        key_step, key = jax.random.split(key)
        
        # Store for rendering
        history_state.append(state)
        
        obs, state, attn_weights = step_fn(key_step, obs, state)
        history_attention.append(attn_weights) # [B, 4, 1, 4]
        
    console.print("[bold green]✔ Episode Complete. Rendering Attention Maps...[/bold green]")
    
    # 5. Render Attention GIF
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#0f172a') # Deep modern blue/black
    fig.patch.set_facecolor('#0f172a')
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
        ax.set_aspect('equal')
        ax.set_title(f"STAE Cross-Attention | Step {frame}", color="white", fontsize=16)
        
        s = history_state[frame]
        attn = history_attention[frame][0] # [4, 1, 4] (Heads, Q, K)
        # Average across the 4 attention heads
        mean_attn = np.mean(attn[:, 0, :], axis=0) # [4]
        
        target = s.target_pos
        ax.add_patch(Circle(target, 80.0, color='#3b82f6', fill=False, linestyle='--', alpha=0.5))
        ax.scatter(target[0], target[1], color='#3b82f6', marker='x', s=100)
        
        positions = s.usv_state.eta[:, :2]
        # Plot attention lines from Ego (Agent 0) to Teammates (Agents 1-4)
        ego_pos = positions[0]
        for j in range(1, 5):
            teammate_pos = positions[j]
            weight = mean_attn[j-1]
            
            # Map weight [0, 1] to line thickness and alpha
            if weight > 0.05: # Threshold to reduce noise
                ax.plot([ego_pos[0], teammate_pos[0]], 
                        [ego_pos[1], teammate_pos[1]], 
                        color='#10b981', # Neon Green
                        linewidth=weight * 15, 
                        alpha=float(weight))
                        
        # Draw Agents
        for i in range(num_agents):
            color = '#f43f5e' if i == 0 else '#e2e8f0' # Red for Ego, White for Teammates
            ax.scatter(positions[i, 0], positions[i, 1], color=color, s=150, zorder=5)
            
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#334155')
            
    ani = animation.FuncAnimation(fig, update, frames=len(history_state), interval=100)
    save_path = os.path.abspath("stae_attention.gif")
    ani.save(save_path, writer='pillow', fps=10)
    console.print(f"[bold cyan]✔ Attention visualization saved to {save_path}[/bold cyan]")

if __name__ == "__main__":
    main()
