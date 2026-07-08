import os
import math
import jax
import jax.numpy as jnp
import numpy as np
import holoocean
from rich.console import Console
from orbax.checkpoint import PyTreeCheckpointer, RestoreArgs

# Import JAX network
from algorithms.flax_sac import Actor

console = Console()

# ── 1. HOLOOCEAN SCENARIO SETUP ───────────────────────────────────────────────
def generate_swarm_agents(num_agents=4):
    agents = []
    spacing = 15.0
    for i in range(num_agents):
        size = max(1, int(num_agents**0.5))
        row = i // size
        col = i % size
        num_rows = math.ceil(num_agents / size)
        
        spawn_location = [
            float((col - (size - 1) / 2.0) * spacing), 
            float((row - (num_rows - 1) / 2.0) * spacing), 
            0.5
        ]

        agents.append({
            "agent_name": f"vessel{i}",
            "agent_type": "SurfaceVessel",
            "sensors": [
                {"sensor_type": "DynamicsSensor", "sensor_name": "DynamicsSensor"},
                {
                    "sensor_type": "RaycastLidar",
                    "sensor_name": "Lidar",
                    "Hz": 10,
                    "location": [0.0, 0.0, 1.5],
                    "configuration": {
                        "Channels": 1,
                        "Range": 70.0,
                        "HorizontalFov": 360.0,
                        "RotationFrequency": 10,
                        "PointsPerSecond": 640,
                        "UpperFovLimit": 0.0,
                        "LowerFovLimit": 0.0,
                        "ShowDebugPoints": False,
                        "NoiseStdDev": 0.0
                    }
                }
            ],
            "control_scheme": 0,
            "location": spawn_location,
            "rotation": [0, 0, 0]
        })
    return agents

def make_scenario(num_agents: int) -> dict:
    return {
        "name": "SurfaceVesselSwarmNav",
        "package_name": "Ocean",
        "world": "PierHarbor",
        "main_agent": "vessel0",
        "show_viewport": True,
        "agents": generate_swarm_agents(num_agents=num_agents),
        "ticks_per_sec": 200,
        "frames_per_sec": True,
    }

# ── 2. DATA TRANSLATION HELPERS ───────────────────────────────────────────────
def global_to_body_vel(vx, vy, yaw):
    """Converts global UE4 velocities to the AUV body frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    u = vx * c + vy * s
    v = -vx * s + vy * c
    return u, v

def global_to_body_pos(x, y, ego_x, ego_y, ego_yaw):
    """Converts global UE4 coordinates to the AUV relative body frame."""
    rel_x = x - ego_x
    rel_y = y - ego_y
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    body_x = rel_x * c + rel_y * s
    body_y = -rel_x * s + rel_y * c
    return body_x, body_y

# ── 3. MAIN INFERENCE LOOP ────────────────────────────────────────────────────
def main():
    num_agents = 5
    obs_dim = 92
    
    # Define Goals manually for the 5 agents (spreading out in PierHarbor)
    goals = np.array([
        [100.0, 100.0],
        [100.0, -100.0],
        [-100.0, 100.0],
        [-100.0, -100.0],
        [0.0, 150.0]
    ])
    
    # Load Neural Network
    console.print("[bold cyan]Loading JAX Neural Network...[/bold cyan]")
    layout = {
        "ego": {"start": 0, "dim": 8},
        "goal": {"start": 0, "dim": 8}, 
        "lidar": {"start": 8, "dim": 64},
        "auv_entities": {"start": 72, "dim": (num_agents - 1) * 5, "count": num_agents - 1, "feature_dim": 5},
        "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5}
    }
    actor = Actor(layout=layout, action_dim=2, action_scale=jnp.ones(2), action_bias=jnp.zeros(2))
    
    ckpt_dir = os.path.abspath("checkpoints/sac_actor_final")
    ckpt = PyTreeCheckpointer()
    dummy_obs = jnp.zeros((1, obs_dim))
    init_params = actor.init(jax.random.PRNGKey(0), dummy_obs)["params"]
    restore_args = jax.tree_util.tree_map(lambda _: RestoreArgs(restore_type=np.ndarray), init_params)
    raw_params = ckpt.restore(ckpt_dir, item=init_params, restore_args=restore_args)
    actor_params = jax.tree_util.tree_map(jnp.array, raw_params)
    
    def get_action(params, obs):
        # obs shape: (1, obs_dim) — called per-agent to guarantee unique output
        mean, _ = actor.apply({"params": params}, obs)
        return jnp.tanh(mean[0])  # squeeze to (action_dim,)
    jit_action = jax.jit(get_action)
    
    # Initialize Holoocean
    scenario = make_scenario(num_agents)
    console.print("[bold green]Booting Unreal Engine via Holoocean...[/bold green]")
    env = holoocean.make(scenario_cfg=scenario)
    
    # Start Simulation Loop
    states = env.tick()
    
    # ── Draw Goals ────────────────────────────────────────────────────────────
    for g in goals:
        try:
            # Holoocean draw point: Location, Color (R,G,B), Thickness, Lifetime (0=infinite)
            env.draw_point([g[0], g[1], 1.0], color=[0, 255, 0], thickness=50.0, lifetime=0.0)
        except Exception:
            pass # Fails gracefully if draw_point is not supported in this Holoocean version
            
    # Track previous positions for trajectory drawing
    prev_pos = [None] * num_agents
    
    try:
        while True:
            # 1. Parse all agent states
            agent_data = []
            for i in range(num_agents):
                name = f"vessel{i}"
                state = states[name]
                
                # Dynamics: [accel(3), vel(3), pos(3), ang_accel(3), ang_vel(3), rpy(3)]
                dyn = state["DynamicsSensor"]
                pos = dyn[6:8]
                pos_3d = [pos[0], pos[1], 0.0]
                vel = dyn[3:5]
                r = np.radians(dyn[14])
                yaw = np.radians(dyn[17])
                
                # Draw Trajectory
                if prev_pos[i] is not None:
                    try:
                        env.draw_line([prev_pos[i][0], prev_pos[i][1], 0.5], 
                                      [pos_3d[0], pos_3d[1], 0.5], 
                                      color=[255, 0, 0], thickness=10.0, lifetime=0.0)
                    except Exception:
                        pass
                prev_pos[i] = pos_3d
                
                u, v = global_to_body_vel(vel[0], vel[1], yaw)
                
                # Lidar point cloud (N, 3) or (N, 4)
                lidar_pts = state["Lidar"]
                if lidar_pts.shape[0] == 0:
                    lidar_dists = np.full(64, 70.0)
                else:
                    lidar_dists = np.linalg.norm(lidar_pts[:, :3], axis=1)
                    if len(lidar_dists) > 64:
                        lidar_dists = lidar_dists[:64]
                    elif len(lidar_dists) < 64:
                        lidar_dists = np.pad(lidar_dists, (0, 64 - len(lidar_dists)), constant_values=70.0)
                        
                lidar_norm = lidar_dists / 70.0
                
                agent_data.append({
                    "pos": pos,
                    "yaw": yaw,
                    "vel": vel,
                    "u": u, "v": v, "r": r,
                    "lidar": lidar_norm
                })
                
            # 2. Build Observation Tensor [5, 92]
            obs_batch = []
            for i in range(num_agents):
                data = agent_data[i]
                
                # Ego
                rel_goal = goals[i] - data["pos"]
                dist = np.linalg.norm(rel_goal)
                ang_goal = np.arctan2(rel_goal[1], rel_goal[0]) - data["yaw"]
                ego_feats = [
                    np.sin(data["yaw"]), np.cos(data["yaw"]),
                    data["u"], data["v"], data["r"],
                    dist / 300.0,  # Normalize by Holoocean scale (goals ~100-200m), not JAX 1500m scale
                    np.sin(ang_goal), np.cos(ang_goal)
                ]
                
                # Neighbors (Deep Sets)
                neighbor_feats = []
                for j in range(num_agents):
                    if i == j: continue
                    other = agent_data[j]
                    
                    rel_x, rel_y = global_to_body_pos(other["pos"][0], other["pos"][1], data["pos"][0], data["pos"][1], data["yaw"])
                    rel_u, rel_v = global_to_body_vel(other["vel"][0]-data["vel"][0], other["vel"][1]-data["vel"][1], data["yaw"])
                    
                    active = 1.0
                    neighbor_feats.extend([active, rel_x, rel_y, rel_u, rel_v])
                    
                full_obs = np.concatenate([ego_feats, data["lidar"], neighbor_feats])
                obs_batch.append(full_obs)
                
            # 3. Get Neural Actions — call per-agent to guarantee unique output
            actions = []
            for i, obs in enumerate(obs_batch):
                obs_1d = jnp.array(obs[None, :])  # (1, 92)
                act = jit_action(actor_params, obs_1d)  # (2,)
                actions.append(np.array(act))
            actions = np.array(actions)  # (5, 2)
            
            # DEBUG: Print first frame to see if obs/actions differ per agent
            import sys
            if not hasattr(main, '_debug_printed'):
                main._debug_printed = True
                print("\n===== DEBUG: Agent Observations & Actions =====")
                for i in range(num_agents):
                    data = agent_data[i]
                    rel_goal = goals[i] - data["pos"]
                    ang = np.degrees(np.arctan2(rel_goal[1], rel_goal[0]) - data["yaw"])
                    dist = np.linalg.norm(rel_goal)
                    print(f"  Agent {i}: pos={data['pos'].round(1)}, goal={goals[i]}, dist={dist:.1f}m, ang_to_goal={ang:.1f}deg | action=[{actions[i,0]:.3f}, {actions[i,1]:.3f}]")
                print("===============================================\n")
                sys.stdout.flush()
            
            # 4. Apply Actions to Holoocean
            for i in range(num_agents):
                throttle = actions[i, 0]
                steering = actions[i, 1]
                
                left = (throttle + steering) * 5000.0
                right = (throttle - steering) * 5000.0
                
                env.act(f"vessel{i}", [left, right])
                
            # Tick physics with a frame skip
            for _ in range(10):
                states = env.tick()
            
    except KeyboardInterrupt:
        console.print("[yellow]Shutting down Holoocean Bridge...[/yellow]")
        env.__del__()

if __name__ == "__main__":
    main()
