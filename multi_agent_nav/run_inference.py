import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from rich.console import Console

# Import our custom modules
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

console = Console()

def create_boat_polygon(x, y, yaw, length=6.0, width=3.0):
    """Creates a triangular boat shape."""
    # Local coordinates of a simple boat
    pts = np.array([
        [length/2, 0],         # bow
        [-length/2, width/2],  # port stern
        [-length/2, -width/2]  # starboard stern
    ])
    # Rotate by yaw
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rot_pts = np.dot(pts, R.T)
    # Translate
    return rot_pts + np.array([x, y])

def main():
    console.print("[bold cyan]RLSim V2: Multi-Agent Inference Engine[/bold cyan]")
    
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
    # 2000m map means goals spawn up to 1000m (1km) away
    env_params = env.default_params.replace(num_agents=num_agents, map_size=2000.0, max_steps=2000)
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # ── Load Model ────────────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath("checkpoints/sac_actor_final")
    if not os.path.exists(ckpt_dir):
        console.print(f"[red]Error: Checkpoint {ckpt_dir} not found![/red]")
        return
        
    ckpt = PyTreeCheckpointer()
    
    # Load as raw numpy arrays to avoid GPU-to-CPU sharding mismatch
    dummy_obs = jnp.zeros((1, obs_dim))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    
    # Convert raw arrays to JAX device arrays
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    console.print("[green]✔ Neural Network loaded.[/green]")
    
    # ── JIT Inference Functions ───────────────────────────────────────────────
    def get_action(params, obs):
        # We exploit the mean action for inference!
        # CRITICAL: SAC outputs an unbounded mean. We must squash it with tanh!
        mean, _ = actor.apply({"params": params}, obs)
        return jnp.tanh(mean)
        
    jit_action = jax.jit(get_action)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    # ── Run Simulation ────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    reset_keys = jax.random.split(rng, 1)
    
    obs_batch, state_batch = vmap_reset(reset_keys, env_params)
    
    history_pos = []
    history_yaw = []
    
    obstacles = state_batch.obstacles[0] # [10, 3]
    goals = state_batch.goal_pos[0]      # [N, 2]
    
    max_steps = 1500
    console.print(f"Simulating {max_steps} steps (saving every 2nd frame for smooth playback)...")
    
    for step in range(max_steps):
        # obs_batch is [1, N, 92]
        flat_obs = obs_batch.reshape(num_agents, obs_dim)
        
        # Get actions for all agents
        actions = jit_action(actor_params, flat_obs)
        batched_actions = jnp.expand_dims(actions, 0) # [1, N, 2]
        
        # Step env
        step_keys = jax.random.split(jax.random.PRNGKey(step), 1)
        obs_batch, state_batch, reward, done, _ = vmap_step(step_keys, state_batch, batched_actions, env_params)
        
        # Save state for rendering every 2nd step
        if step % 2 == 0:
            pos = state_batch.usv_state.eta[0, :, :2] # [N, 2]
            yaw = state_batch.usv_state.eta[0, :, 2]  # [N]
            history_pos.append(np.array(pos))
            history_yaw.append(np.array(yaw))
        
        if jnp.any(done):
            # We hit a goal or a collision! But we'll keep simulating for the video
            pass
            
    # ── Rendering ─────────────────────────────────────────────────────────────
    console.print("[yellow]Rendering MP4/GIF Animation... (This may take a minute)[/yellow]")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Setup map bounds for a 2000m map
    ax.set_xlim(-1200, 1200)
    ax.set_ylim(-1200, 1200)
    
    ax.set_title("SAC Navigation Agent - Swarm Inference", color='black', fontsize=16)
    ax.grid(True, color='grey', linestyle='-', alpha=0.5)
    ax.tick_params(colors='black')
    
    # Draw Obstacles (Small grey squares)
    obs_arr = np.array(obstacles)
    ax.scatter(obs_arr[:, 0], obs_arr[:, 1], marker='s', color='grey', s=10, label='Obstacle Field')
        
    # Draw Goals (Large green stars)
    goal_arr = np.array(goals)
    ax.scatter(goal_arr[:, 0], goal_arr[:, 1], marker='*', color='green', s=300, label='Goal')
    
    # Draw Start Points (Large blue circles)
    start_pos = history_pos[0]
    ax.scatter(start_pos[:, 0], start_pos[:, 1], marker='o', color='blue', s=150, label='Start')
        
    # Initialize AUV trajectory lines and boat markers
    traj_lines = []
    boat_markers = []
    colors = ['red', 'orange', 'purple', 'magenta', 'brown']
    
    for i in range(num_agents):
        # The trajectory trailing line
        line, = ax.plot([], [], color=colors[i%len(colors)], linewidth=2, label='Trajectory' if i==0 else "")
        traj_lines.append(line)
        
        # The current AUV position dot
        marker, = ax.plot([], [], 'o', color=colors[i%len(colors)], markersize=8)
        boat_markers.append(marker)
        
    ax.legend(loc='center')
        
    def animate(frame):
        # We plot the trajectory up to the current frame
        past_pos = np.array(history_pos[:frame+1]) # [T, N, 2]
        
        updated_artists = []
        for i in range(num_agents):
            traj_lines[i].set_data(past_pos[:, i, 0], past_pos[:, i, 1])
            boat_markers[i].set_data([past_pos[-1, i, 0]], [past_pos[-1, i, 1]])
            updated_artists.extend([traj_lines[i], boat_markers[i]])
            
        return updated_artists
        
    anim = animation.FuncAnimation(fig, animate, frames=len(history_pos), interval=50, blit=True)
    
    # Export using Pillow (No ffmpeg required!)
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/swarm_simulation.gif"
    writer = animation.PillowWriter(fps=20)
    anim.save(out_path, writer=writer)
    
    console.print(f"[bold green]✔ Saved gorgeous swarm simulation to {out_path}[/bold green]")
    plt.close()

if __name__ == "__main__":
    main()
