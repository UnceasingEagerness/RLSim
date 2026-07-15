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

def main():
    console.print("[bold cyan]RLSim V2: Advanced XAI Visualization Engine[/bold cyan]")
    
    env_num_agents = 2  # The physical simulation only has 2 agents
    nn_num_agents = 5   # The Neural Network was trained to expect 5 agents (92 dims)
    obs_dim_env = 72 + (env_num_agents - 1) * 5 # 77
    obs_dim_nn = 92
    action_dim = 2
    
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (nn_num_agents - 1) * 5, "count": nn_num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    env = JaxUSVEnv()
    # 2 Agents, 150 Obstacles
    env_params = env.default_params.replace(num_agents=env_num_agents, map_size=1000.0, num_obstacles=150, max_steps=1500)
    
    rng = jax.random.PRNGKey(42)
    jitted_map_gen = jax.jit(env.generate_map_bank, static_argnums=(1, 2, 3, 4))
    goals_bank, obstacles_bank = jitted_map_gen(
        rng, 
        int(env_params.num_agents), 
        int(env_params.num_obstacles), 
        float(env_params.map_size), 
        int(env_params.map_bank_size)
    )
    
    # --- TOGGLE: SHARED VS DIFFERENT GOALS ---
    shared_goal = goals_bank[:, 0:1, :]
    goals_bank_shared = jnp.repeat(shared_goal, env_num_agents, axis=1)
    env_params = env_params.replace(goals_bank=goals_bank_shared, obstacles_bank=obstacles_bank)
    
    actor = Actor(layout=layout, action_dim=action_dim, action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
    
    # ── Load Model ────────────────────────────────────────────────────────────
    ckpt_dir = os.path.abspath("checkpoints/sac_actor_final")
    if not os.path.exists(ckpt_dir):
        console.print(f"[red]Error: Checkpoint {ckpt_dir} not found![/red]")
        return
        
    ckpt = PyTreeCheckpointer()
    dummy_obs = jnp.zeros((1, obs_dim_nn))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    # ── JIT Inference Functions ───────────────────────────────────────────────
    def get_action(params, obs):
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
    history_obs = []
    history_act = []
    history_collisions = []
    
    obstacles = state_batch.obstacles[0] # [150, 3]
    goals = state_batch.goal_pos[0]      # [N, 2]
    
    max_steps = 1500
    console.print(f"Simulating {max_steps} steps (saving every 2nd frame)...")
    
    for step in range(max_steps):
        # Pad obs from 77 to 92 for the neural network
        flat_obs_env = obs_batch.reshape(env_num_agents, obs_dim_env)
        flat_obs_nn = jnp.zeros((env_num_agents, obs_dim_nn))
        flat_obs_nn = flat_obs_nn.at[:, :obs_dim_env].set(flat_obs_env)
        
        actions = jit_action(actor_params, flat_obs_nn)
        batched_actions = jnp.expand_dims(actions, 0)
        
        step_keys = jax.random.split(jax.random.PRNGKey(step), 1)
        obs_batch, state_batch, reward, done, _ = vmap_step(step_keys, state_batch, batched_actions, env_params)
        
        if step % 2 == 0:
            pos = state_batch.usv_state.eta[0, :, :2]
            
            # Mathematical Collision Detection (Distance < 4.0)
            col_list = []
            for i in range(env_num_agents):
                # Check obstacle collision
                obs_dist = jnp.linalg.norm(obstacles[:, :2] - pos[i], axis=1)
                if jnp.any(obs_dist < 4.0):
                    col_list.append(i)
                    continue
                # Check agent collision
                for j in range(env_num_agents):
                    if i != j and jnp.linalg.norm(pos[j] - pos[i]) < 4.0:
                        col_list.append(i)
                        break
            
            history_pos.append(np.array(pos))
            history_obs.append(np.array(flat_obs_nn[0])) # Track Ego Agent 0's brain
            history_act.append(np.array(actions[0]))
            history_collisions.append(col_list)
            
    # ── Rendering XAI Split-Screen ────────────────────────────────────────────
    console.print("[yellow]Rendering XAI Animation...[/yellow]")
    
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#1e1e1e') # Dark mode for XAI
    
    # Left Plot: Physical Environment
    ax_env = fig.add_subplot(1, 2, 1)
    ax_env.set_facecolor('#121212')
    ax_env.set_xlim(-800, 800) # Guarantee goals are in frame
    ax_env.set_ylim(-800, 800)
    ax_env.set_title("Physical Swarm Environment (150 Obstacles)", color='white', fontsize=16)
    ax_env.tick_params(colors='white')
    ax_env.grid(True, color='#333333', linestyle='-', alpha=0.5)
    
    # Right Plot: Neural Activation Data Flow
    ax_brain = fig.add_subplot(1, 2, 2)
    ax_brain.set_facecolor('#121212')
    ax_brain.set_xlim(-1, 1)
    ax_brain.set_ylim(0, 92)
    ax_brain.set_title("Ego Agent 0: Live Neural Data Flow", color='cyan', fontsize=16)
    ax_brain.tick_params(colors='white')
    ax_brain.set_yticks([4, 40, 82])
    ax_brain.set_yticklabels(['Kinematics', 'LiDAR Rays', 'Neighbor Entities'], color='cyan')
    
    # Draw Obstacles
    obs_arr = np.array(obstacles)
    ax_env.scatter(obs_arr[:, 0], obs_arr[:, 1], marker='s', color='#ff5555', s=20, label='Obstacle', alpha=0.7)
        
    # Draw Goals
    goal_arr = np.array(goals)
    ax_env.scatter(goal_arr[:, 0], goal_arr[:, 1], marker='*', color='#00ff00', s=300, label='Goal')
    
    traj_lines = []
    boat_markers = []
    crash_markers = []
    colors = ['#00ffff', '#ff00ff', '#ffff00', '#ff8800', '#00ff88']
    
    for i in range(env_num_agents):
        line, = ax_env.plot([], [], color=colors[i], linewidth=2, alpha=0.5)
        traj_lines.append(line)
        marker, = ax_env.plot([], [], 'o', color=colors[i], markersize=10, markeredgecolor='white')
        boat_markers.append(marker)
        crash, = ax_env.plot([], [], marker='x', color='red', markersize=20, markeredgewidth=4)
        crash_markers.append(crash)
        
    # Neural Bars
    bar_y = np.arange(92)
    bars = ax_brain.barh(bar_y, np.zeros(92), color='cyan', alpha=0.8)
    
    action_text = ax_brain.text(-0.9, 88, "", color='#00ff00', fontsize=18, fontweight='bold')
    col_text = ax_env.text(-750, 700, "", color='red', fontsize=24, fontweight='bold')
        
    def animate(frame):
        past_pos = np.array(history_pos[:frame+1])
        updated_artists = []
        
        # 1. Update Environment
        active_crashes = history_collisions[frame]
        
        for i in range(env_num_agents):
            traj_lines[i].set_data(past_pos[:, i, 0], past_pos[:, i, 1])
            boat_markers[i].set_data([past_pos[-1, i, 0]], [past_pos[-1, i, 1]])
            
            if i in active_crashes:
                crash_markers[i].set_data([past_pos[-1, i, 0]], [past_pos[-1, i, 1]])
            else:
                crash_markers[i].set_data([], [])
                
            updated_artists.extend([traj_lines[i], boat_markers[i], crash_markers[i]])
            
        if len(active_crashes) > 0:
            col_text.set_text("💥 COLLISION DETECTED! 💥")
        else:
            col_text.set_text("")
        updated_artists.append(col_text)
            
        # 2. Update Neural Brain (Ego Agent 0)
        obs = history_obs[frame]
        act = history_act[frame]
        
        for idx, bar in enumerate(bars):
            val = obs[idx]
            # Color code different zones
            if idx < 8:
                bar.set_color('#ff00ff') # Kinematics
            elif idx < 72:
                bar.set_color('#00ffff') # LiDAR
            else:
                bar.set_color('#ffff00') # Neighbors
            bar.set_width(val)
            
        action_text.set_text(f"Throttle: {act[0]:.2f} | Steering: {act[1]:.2f}")
        
        updated_artists.extend(bars)
        updated_artists.append(action_text)
            
        return updated_artists
        
    anim = animation.FuncAnimation(fig, animate, frames=len(history_pos), interval=50, blit=True)
    
    os.makedirs("logs", exist_ok=True)
    out_path = 'logs/xai_simulation_2_agents_dense.gif'
    anim.save(out_path, writer='pillow', fps=20)
    console.print(f"[bold green]✔ Saved XAI visualization to {out_path}[/bold green]")
    plt.close()

if __name__ == "__main__":
    main()
