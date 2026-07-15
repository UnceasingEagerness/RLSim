import jax
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import os

from env.jax_pe_env import JaxPursuitEvasionEnv, PEEnvParams, PEEnvState
from algorithms.pe_flax_sac import PursuerActor, EvaderActor

def main():
    print("Loading Environment and Checkpoints...")
    env = JaxPursuitEvasionEnv()
    params = PEEnvParams()
    
    # Initialize empty states for restoration
    p_actor = PursuerActor()
    e_actor = EvaderActor()
    
    dummy_obs = jnp.zeros((1, 10, 77))
    rng = jax.random.PRNGKey(0)
    p_params = p_actor.init(rng, dummy_obs)["params"]
    e_params = e_actor.init(rng, dummy_obs)["params"]
    
    p_state = train_state.TrainState.create(apply_fn=p_actor.apply, params=p_params, tx=optax.adam(1e-4))
    e_state = train_state.TrainState.create(apply_fn=e_actor.apply, params=e_params, tx=optax.adam(1e-4))
    
    # Restore using flax.training.checkpoints which is more robust for TrainState
    from flax.training import checkpoints
    p_state = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath("checkpoints_pe/pursuer_actor"), target=p_state)
    e_state = checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath("checkpoints_pe/evader_actor"), target=e_state)
    print("Models restored successfully!")

    # ---------------------------------------------------------
    # Saliency (Gradient) Function
    # ---------------------------------------------------------
    # We compute the gradient of the magnitude of the action (throttle + steering)
    # with respect to the input observation history.
    def get_action_and_saliency(params, obs_hist):
        def action_magnitude(obs):
            # obs is [1, 10, 77]
            means, _ = p_actor.apply({"params": params}, obs)
            action = jnp.tanh(means)[0] # [2]
            return jnp.sum(jnp.abs(action)) # Sum of absolute throttle and steering
        
        # Saliency map is the derivative of the action magnitude w.r.t the observation
        saliency_grad = jax.grad(action_magnitude)(obs_hist)
        
        # Get actual action for the step
        means, _ = p_actor.apply({"params": params}, obs_hist)
        return jnp.tanh(means)[0], saliency_grad[0, -1] # Return latest timestep saliency

    jit_saliency = jax.jit(get_action_and_saliency)
    
    @jax.jit
    def get_e_action(params, obs_hist):
        means, _ = e_actor.apply({"params": params}, obs_hist)
        return jnp.tanh(means)[0]

    # ---------------------------------------------------------
    # Rollout Simulation
    # ---------------------------------------------------------
    key = jax.random.PRNGKey(42) # Fixed seed for reproducible visualization
    reset_key, sim_key = jax.random.split(key)
    
    obs, state = env.reset(reset_key, params)
    
    # History buffer for the single episode
    obs_history = jnp.repeat(jnp.expand_dims(obs, axis=1), 10, axis=1) # [4, 10, 77]
    
    frames = []
    print("Simulating Rollout and Extracting Saliency...")
    for step in range(500): # 50 seconds max
        p_actions = []
        p_saliencies = []
        
        # Get Pursuer Actions & Saliency
        for i in range(3):
            act, sal = jit_saliency(p_state.params, jnp.expand_dims(obs_history[i], axis=0))
            p_actions.append(act)
            p_saliencies.append(sal)
            
        p_actions = jnp.stack(p_actions)
        p_saliencies = jnp.stack(p_saliencies) # [3, 77]
        
        # Get Evader Action
        e_act = get_e_action(e_state.params, jnp.expand_dims(obs_history[3], axis=0))
        
        actions = jnp.concatenate([p_actions, jnp.expand_dims(e_act, 0)], axis=0)
        
        # Step environment
        step_key, sim_key = jax.random.split(sim_key)
        next_obs, next_state, _, done, info = env.step(step_key, state, actions, params)
        
        # Save frame data
        frames.append({
            "eta": np.array(state.eta),
            "obstacles": np.array(state.static_obstacles),
            "saliency": np.array(p_saliencies)
        })
        
        # Update history
        obs_history = jnp.concatenate([obs_history[:, 1:, :], jnp.expand_dims(next_obs, axis=1)], axis=1)
        state = next_state
        
        if jnp.any(done):
            print(f"Episode finished at step {step} due to Capture/Collision.")
            break

    # ---------------------------------------------------------
    # Rendering GIF
    # ---------------------------------------------------------
    print("Rendering Saliency Map GIF...")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    def update(frame_idx):
        ax.clear()
        ax.set_facecolor('#0f172a')
        ax.set_xlim(0, params.map_size)
        ax.set_ylim(0, params.map_size)
        ax.grid(color='#334155', linestyle='--', alpha=0.3)
        ax.set_title(f"Swarm Transformer Saliency Analysis | Step {frame_idx}", color="white", pad=20)
        
        frame = frames[frame_idx]
        eta = frame["eta"]
        obs = frame["obstacles"]
        saliencies = frame["saliency"]
        
        # Plot Obstacles
        for o in obs:
            circle = patches.Circle((o[0], o[1]), o[2], color='#475569', alpha=0.8, ec='#94a3b8', lw=2)
            ax.add_patch(circle)
            
        # Plot Evader (Index 3)
        evader = patches.Circle((eta[3, 0], eta[3, 1]), 1.5, color='#ef4444', zorder=5) # Red
        ax.add_patch(evader)
        
        # Plot Pursuers and their Saliency Beams
        for i in range(3):
            px, py, pyaw = eta[i]
            sal = saliencies[i]
            
            # 1. Target Saliency (O_out: indices 0,1,2)
            target_sal = np.sum(np.abs(sal[:3]))
            
            # 2. Teammate Saliency (O_in: indices 3 to 12)
            team_sal = np.sum(np.abs(sal[3:13]))
            
            # 3. Obstacle Saliency (LiDAR: indices 13 to 76)
            lidar_sal = np.abs(sal[13:77])
            
            # Draw Pursuer
            pursuer = patches.Polygon([
                [px + 2*np.cos(pyaw), py + 2*np.sin(pyaw)],
                [px + 1.5*np.cos(pyaw + 2.5), py + 1.5*np.sin(pyaw + 2.5)],
                [px + 1.5*np.cos(pyaw - 2.5), py + 1.5*np.sin(pyaw - 2.5)]
            ], color='#3b82f6', zorder=6) # Blue
            ax.add_patch(pursuer)
            
            # --- Draw Attention Beams ---
            # Saliency Line to Evader
            alpha_e = np.clip(target_sal * 5.0, 0, 1)
            ax.plot([px, eta[3,0]], [py, eta[3,1]], color='#ef4444', lw=2, alpha=alpha_e, ls='--')
            
            # Saliency Line to Teammates
            alpha_t = np.clip(team_sal * 5.0, 0, 1)
            for j in range(3):
                if i != j:
                    ax.plot([px, eta[j,0]], [py, eta[j,1]], color='#3b82f6', lw=1, alpha=alpha_t, ls=':')
            
            # Saliency LiDAR Beams
            angles = np.linspace(-np.pi/3, np.pi/3, 64)
            for ray_idx, angle in enumerate(angles):
                ray_sal = lidar_sal[ray_idx]
                ray_alpha = np.clip(ray_sal * 20.0, 0, 1) # Amplify small gradients
                if ray_alpha > 0.1:
                    ray_angle = pyaw + angle
                    # Draw a short ray indicating attention to that direction
                    rx = px + 5.0 * np.cos(ray_angle)
                    ry = py + 5.0 * np.sin(ray_angle)
                    ax.plot([px, rx], [py, ry], color='#10b981', lw=1.5, alpha=ray_alpha)
                    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
    ani.save('saliency_pursuit_evasion.gif', writer='pillow')
    print("Saved saliency_pursuit_evasion.gif!")

if __name__ == "__main__":
    main()
