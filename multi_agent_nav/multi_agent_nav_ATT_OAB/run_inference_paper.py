import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs
from rich.console import Console

# Import our custom modules
from env.jax_usv_env import JaxUSVEnv, EnvParams
from algorithms.flax_sac import Actor

console = Console()

def main():
    console.print("[bold cyan]RLSim V2: Research Paper Visualization Engine[/bold cyan]")
    
    env_num_agents = 5  
    nn_num_agents = 5   
    obs_dim_env = (72 + (env_num_agents - 1) * 5) * 10
    obs_dim_nn = (72 + (nn_num_agents - 1) * 5) * 10
    action_dim = 2
    
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (nn_num_agents - 1) * 5, "count": nn_num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    
    env = JaxUSVEnv()
    # Scenario A: 5 Agents, 150 Obstacles, Different Goals
    env_params = env.default_params.replace(num_agents=env_num_agents, map_size=2000.0, num_obstacles=150, max_steps=1500)
    
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
        
    def get_saliency(params, obs_single):
        # Calculate Jacobian of the action wrt the observation input
        # Wrap it to handle the batch dimension required by Actor
        def batched_action(o):
            o_b = jnp.expand_dims(o, 0)
            mean, _ = actor.apply({"params": params}, o_b)
            return jnp.tanh(mean)[0]
        J = jax.jacobian(batched_action)(obs_single) # Shape: (2, 92)
        return jnp.mean(jnp.abs(J), axis=0) # Average sensitivity across both Throttle & Steering. Shape: (92,)
        
    jit_action = jax.jit(get_action)
    jit_saliency = jax.jit(get_saliency)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    # ── Run Simulation ────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    reset_keys = jax.random.split(rng, 1)
    obs_batch, state_batch = vmap_reset(reset_keys, env_params)
    
    history_pos = []
    history_saliency = []
    history_act = []
    history_collisions = []
    
    obstacles = state_batch.obstacles[0] 
    goals = state_batch.goal_pos[0]      
    
    max_steps = 1500
    console.print(f"Simulating {max_steps} steps (saving every 2nd frame)...")
    
    for step in range(max_steps):
        flat_obs_env = obs_batch.reshape(env_num_agents, obs_dim_env)
        flat_obs_nn = jnp.zeros((env_num_agents, obs_dim_nn))
        flat_obs_nn = flat_obs_nn.at[:, :obs_dim_env].set(flat_obs_env)
        
        actions = jit_action(actor_params, flat_obs_nn)
        batched_actions = jnp.expand_dims(actions, 0)
        
        step_keys = jax.random.split(jax.random.PRNGKey(step), 1)
        obs_batch, state_batch, reward, done, _ = vmap_step(step_keys, state_batch, batched_actions, env_params)
        
        if step % 2 == 0:
            pos = state_batch.usv_state.eta[0, :, :2]
            
            col_list = []
            for i in range(env_num_agents):
                obs_dist = jnp.linalg.norm(obstacles[:, :2] - pos[i], axis=1)
                if jnp.any(obs_dist < 4.0):
                    col_list.append(i)
                    continue
                for j in range(env_num_agents):
                    if i != j and jnp.linalg.norm(pos[j] - pos[i]) < 4.0:
                        col_list.append(i)
                        break
            
            history_pos.append(np.array(pos))
            # Track Ego Agent 0's brain mathematically using True Saliency Gradients
            sal = jit_saliency(actor_params, flat_obs_nn[0])
            history_saliency.append(np.array(sal)) 
            history_act.append(np.array(actions[0]))
            history_collisions.append(col_list)
            
    # ── Rendering Academic Split-Screen ────────────────────────────────────────────
    console.print("[yellow]Rendering Academic Animation...[/yellow]")
    
    # White background for paper
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white') 
    
    # Left Plot: Physical Environment
    ax_env = fig.add_subplot(1, 2, 1)
    ax_env.set_facecolor('white')
    ax_env.set_xlim(-1100, 1100) 
    ax_env.set_ylim(-1100, 1100)
    ax_env.set_title("Multi-Agent Navigation (Dense Environment)", color='black', fontsize=14, fontweight='bold')
    ax_env.tick_params(colors='black')
    ax_env.grid(True, color='#dddddd', linestyle='--', alpha=0.8)
    
    # Right Plot: Neural Information Processing
    ax_brain = fig.add_subplot(1, 2, 2)
    ax_brain.set_facecolor('white')
    ax_brain.set_xlim(0, 1.0)
    ax_brain.set_ylim(-0.5, 2.5)
    ax_brain.set_title("Agent 0: Neural Attention (Saliency Gradients)", color='black', fontsize=14, fontweight='bold')
    ax_brain.tick_params(colors='black')
    ax_brain.set_yticks([0, 1, 2])
    ax_brain.set_yticklabels(['Teammate Sensors', 'Obstacle Sensors (LiDAR)', 'Self-Kinematics'], color='black', fontsize=12)
    ax_brain.grid(True, color='#dddddd', linestyle='--', alpha=0.8)
    ax_brain.set_xlabel("Gradient Relevance / Sensitivity (Normalized)", color='black', fontsize=12)
    
    # Draw Obstacles (Simple black dots/squares)
    obs_arr = np.array(obstacles)
    ax_env.scatter(obs_arr[:, 0], obs_arr[:, 1], marker='s', color='black', s=10, label='Obstacle', alpha=0.8)
        
    # Draw Goals (Red stars)
    goal_arr = np.array(goals)
    ax_env.scatter(goal_arr[:, 0], goal_arr[:, 1], marker='*', color='#d62728', s=200, label='Goal')
    
    traj_lines = []
    boat_markers = []
    crash_markers = []
    # Academic color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    
    for i in range(env_num_agents):
        line, = ax_env.plot([], [], color=colors[i], linewidth=1.5, alpha=0.7)
        traj_lines.append(line)
        marker, = ax_env.plot([], [], 'o', color=colors[i], markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        boat_markers.append(marker)
        crash, = ax_env.plot([], [], marker='x', color='red', markersize=15, markeredgewidth=2)
        crash_markers.append(crash)
        
    # Simple 3-bar chart
    bar_y = np.array([2, 1, 0]) # Kinematics, LiDAR, Neighbors
    bars = ax_brain.barh(bar_y, np.zeros(3), color=['#7f7f7f', '#1f77b4', '#ff7f0e'], height=0.6, alpha=0.8, edgecolor='black')
    
    col_text = ax_env.text(-1000, 950, "", color='red', fontsize=16, fontweight='bold')
        
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
            col_text.set_text("COLLISION DETECTED")
        else:
            col_text.set_text("")
        updated_artists.append(col_text)
            
        # 2. Update Neural Brain (Ego Agent 0) - True Saliency Gradient
        sal = history_saliency[frame]
        
        # Reshape the 920-dim saliency into (10, 92) and average across the 10 time frames
        sal_frames = sal.reshape((10, 92))
        sal_spatial = np.mean(sal_frames, axis=0)
        
        kinematics_val = np.sum(sal_spatial[:8])
        lidar_val = np.sum(sal_spatial[8:72])
        neighbor_val = np.sum(sal_spatial[72:92])
        
        # Normalize so the largest is 1.0, or sum is 1.0, to show relative focus
        total = kinematics_val + lidar_val + neighbor_val + 1e-6
        kin_norm = kinematics_val / total
        lidar_norm = lidar_val / total
        neigh_norm = neighbor_val / total
        
        bars[0].set_width(kin_norm)
        bars[1].set_width(lidar_norm)
        bars[2].set_width(neigh_norm)
        
        updated_artists.extend(bars)
            
        return updated_artists
        
    anim = animation.FuncAnimation(fig, animate, frames=len(history_pos), interval=50, blit=True)
    
    os.makedirs("logs", exist_ok=True)
    out_path = 'logs/paper_simulation_mean_pool.gif'
    anim.save(out_path, writer='pillow', fps=20)
    console.print(f"[bold green]✔ Saved Academic visualization to {out_path}[/bold green]")
    plt.close()

if __name__ == "__main__":
    main()
