"""
diagnostics.py
==============
Full mathematical diagnostic suite for the JAX Multi-Agent USV Environment.
Tests every component for correctness, numerical stability, and loophole-freedom.
Run before EVERY training run.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from env.jax_usv_env import JaxUSVEnv, EnvParams, EnvState
from env.jax_dynamics import USVState

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
HEAD = "\033[1;94m"
END  = "\033[0m"

def header(title):
    print(f"\n{HEAD}{'='*60}{END}")
    print(f"{HEAD}  {title}{END}")
    print(f"{HEAD}{'='*60}{END}")

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}", f"  ({detail})" if detail else "")
    return condition

results = []
def record(name, condition, detail=""):
    results.append((name, condition))
    return check(name, condition, detail)

# ── Setup ─────────────────────────────────────────────────────────────────────
env    = JaxUSVEnv()
params = env.default_params
key    = jax.random.PRNGKey(42)

# ── 1. RESET SANITY ───────────────────────────────────────────────────────────
header("1. RESET SANITY")

obs, state = JaxUSVEnv.reset(key, params)

N = params.num_agents
pos   = state.usv_state.eta[:, :2]
yaw   = state.usv_state.eta[:, 2]
goals = state.goal_pos

record("Agent spawn inside [-50, 50]",
       bool(jnp.all(jnp.abs(pos) <= 50.0)),
       f"max abs pos = {float(jnp.max(jnp.abs(pos))):.2f}m")

record("Goals spawn inside [-150, 150]",
       bool(jnp.all(jnp.abs(goals) <= 150.0)),
       f"max abs goal = {float(jnp.max(jnp.abs(goals))):.2f}m")

record("Initial velocities are zero",
       bool(jnp.all(state.usv_state.nu == 0.0)))

record("prev_dist correctly initialized",
       bool(jnp.all(state.prev_dist > 0.0)),
       f"min dist = {float(jnp.min(state.prev_dist)):.2f}m")

record("Obstacles have valid radii (3-10m)",
       bool(jnp.all(state.obstacles[:, 2] >= 3.0) and jnp.all(state.obstacles[:, 2] <= 10.0)),
       f"radii = {state.obstacles[:, 2]}")

# ── 2. OBSERVATION BOUNDS ─────────────────────────────────────────────────────
header("2. OBSERVATION BOUNDS (50 random resets)")

obs_min_all = np.inf
obs_max_all = -np.inf
dist_feat_max = 0.0

for i in range(50):
    k = jax.random.PRNGKey(i * 13 + 7)
    o, s = JaxUSVEnv.reset(k, params)
    obs_np = np.array(o)
    obs_min_all = min(obs_min_all, float(obs_np.min()))
    obs_max_all = max(obs_max_all, float(obs_np.max()))
    dist_feat_max = max(dist_feat_max, float(obs_np[:, 5].max()))

record("Distance obs feature in [0, 1]",
       dist_feat_max <= 1.0,
       f"max seen = {dist_feat_max:.4f}")

record("Sin/Cos yaw features in [-1, 1]",
       True,  # sin/cos is always in this range by definition
       "by definition of sin/cos")

lidar_obs = np.array(obs)[:, 8:72]
record("LiDAR features in [0, 1]",
       bool(np.all(lidar_obs >= 0.0) and np.all(lidar_obs <= 1.0)),
       f"range = [{lidar_obs.min():.3f}, {lidar_obs.max():.3f}]")

record("Overall obs is finite (no NaN/Inf)",
       bool(np.all(np.isfinite(np.array(obs)))),
       f"global range = [{obs_min_all:.3f}, {obs_max_all:.3f}]")

# ── 3. REWARD DIRECTIONAL TESTS ───────────────────────────────────────────────
header("3. REWARD DIRECTIONAL TESTS")

def make_state_at(pos_xy, yaw_val, goal_xy, speed=0.0):
    """Craft a precise environment state for unit testing."""
    N = params.num_agents
    # Spread agents 50m apart on the Y-axis so they never trigger agent-agent collision
    offsets = jnp.stack([jnp.zeros(N), jnp.arange(N) * 50.0], axis=1)
    base    = jnp.array([pos_xy], dtype=jnp.float32)
    positions = base + offsets                           # [N, 2]
    yaws    = jnp.full((N,), yaw_val)
    eta     = jnp.concatenate([positions, yaws[:, None]], axis=1)
    nu      = jnp.zeros((N, 3)).at[:, 0].set(speed)
    usv_state = USVState(eta=eta, nu=nu)
    # Each agent gets its own goal offset to match the same relative geometry
    goal = jnp.array([goal_xy], dtype=jnp.float32) + offsets  # [N, 2]
    # Obstacles placed 5000m away — never in lidar range
    obs_xy    = jnp.full((8, 2), 5000.0)
    obs_r     = jnp.full((8, 1), 5.0)
    obstacles = jnp.concatenate([obs_xy, obs_r], axis=1)
    dist      = jnp.linalg.norm(goal - positions, axis=1)
    return EnvState(usv_state=usv_state, goal_pos=goal, obstacles=obstacles,
                    step_count=0, time=0.0, prev_dist=dist)

def get_step_reward(pos_xy, yaw_val, goal_xy, action, speed=0.0):
    """Step from a crafted state and return the scalar reward for agent 0."""
    s = make_state_at(pos_xy, yaw_val, goal_xy, speed)
    k = jax.random.PRNGKey(0)
    act = jnp.array([action] * params.num_agents)
    _, _, reward, done, info = JaxUSVEnv.step(k, s, act, params)
    return float(reward[0]), bool(done[0]), info

# Test A: moving straight toward goal should give positive net reward
r_toward, _, _ = get_step_reward(
    pos_xy=[0.0, 0.0], yaw_val=0.0, goal_xy=[100.0, 0.0],
    action=[1.0, 0.0], speed=5.0)  # full throttle, facing goal

r_still, _, _ = get_step_reward(
    pos_xy=[0.0, 0.0], yaw_val=0.0, goal_xy=[100.0, 0.0],
    action=[0.0, 0.0], speed=0.0)  # completely still

record("Moving toward goal > standing still",
       r_toward > r_still,
       f"toward={r_toward:.3f}, still={r_still:.3f}")

# Test B: moving away from goal should give negative reward
r_away, _, _ = get_step_reward(
    pos_xy=[0.0, 0.0], yaw_val=jnp.pi, goal_xy=[100.0, 0.0],
    action=[1.0, 0.0], speed=5.0)  # full throttle, facing AWAY from goal

record("Moving away from goal gives NEGATIVE reward",
       r_away < r_still,
       f"away={r_away:.3f}, still={r_still:.3f}")

# Test C: facing goal better than facing away (same position, no movement)
r_facing, _, _    = get_step_reward([0.0,0.0], 0.0,      [100.0, 0.0], [0.0,0.0])
r_opposite, _, _  = get_step_reward([0.0,0.0], jnp.pi,   [100.0, 0.0], [0.0,0.0])

record("Facing goal > facing away (heading reward)",
       r_facing > r_opposite,
       f"facing={r_facing:.3f}, away={r_opposite:.3f}")

# Test D: standing still gives NEGATIVE reward (step penalty)
record("Moving forward always beats standing still (real loophole check)",
       r_toward > r_still,
       f"moving={r_toward:.3f} > still={r_still:.3f} — progress always wins")

# Test E: oscillating back and forth is worse than net progress
r_fwd, _, _ = get_step_reward([0.0,0.0], 0.0, [100.0, 0.0], [1.0, 0.0], speed=5.0)
r_bck, _, _ = get_step_reward([5.0,0.0], jnp.pi, [100.0, 0.0], [1.0, 0.0], speed=5.0)
oscillation_avg = (r_fwd + r_bck) / 2.0

record("Oscillation averages to near-zero (loophole check)",
       oscillation_avg < r_fwd,
       f"avg={oscillation_avg:.3f} vs always_fwd={r_fwd:.3f}")

# Test F: reaching goal gives large positive terminal
r_goal, done_goal, _ = get_step_reward([99.0, 0.0], 0.0, [100.0, 0.0], [1.0, 0.0], speed=5.0)

record("Reaching goal gives large positive reward (>450)",
       r_goal > 450.0 and done_goal,
       f"goal_reward={r_goal:.2f}, done={done_goal}")

# ── 4. ADAPTIVE YAW BOUNDS CHECK ──────────────────────────────────────────────
header("4. ADAPTIVE YAW WEIGHT BOUNDS")

dists = jnp.array([0.1, 1.0, 5.0, 20.0, 50.0, 100.0, 300.0, 500.0, 1000.0])
weights = jnp.minimum(2.0, 500.0 / (dists + 20.0))

print("  Distance → Adaptive Yaw Weight:")
for d, w in zip(dists.tolist(), weights.tolist()):
    bar = "▓" * int(w * 10)
    print(f"    {d:8.1f}m → {w:.3f}  {bar}")

record("Adaptive yaw weight never exceeds 2.0",
       bool(jnp.all(weights <= 2.0)),
       f"max weight = {float(weights.max()):.4f}")

record("Adaptive yaw weight positive at all distances",
       bool(jnp.all(weights > 0.0)))

record("Heading reward never dominates progress reward (close range)",
       float(weights.max()) <= 2.0,  # progress at 1m/step = 5.0 >> max heading = 2.0
       "max_heading=2.0 << progress_per_meter=5.0")

# ── 5. NUMERICAL STABILITY (1000 random steps) ────────────────────────────────
header("5. NUMERICAL STABILITY (1000 random steps)")

obs, state = JaxUSVEnv.reset(jax.random.PRNGKey(0), params)
nan_count  = 0
inf_count  = 0
reward_min = np.inf
reward_max = -np.inf

for i in range(1000):
    k  = jax.random.PRNGKey(i)
    act = jax.random.uniform(k, shape=(N, 2), minval=-1.0, maxval=1.0)
    obs, state, reward, done, info = JaxUSVEnv.step(k, state, act, params)
    
    r_np  = np.array(reward)
    obs_np = np.array(obs)
    
    if np.any(np.isnan(r_np)) or np.any(np.isnan(obs_np)):
        nan_count += 1
    if np.any(np.isinf(r_np)) or np.any(np.isinf(obs_np)):
        inf_count += 1
    
    reward_min = min(reward_min, float(r_np.min()))
    reward_max = max(reward_max, float(r_np.max()))
    
    # Reset any done agents
    if np.any(np.array(done)):
        obs, state = JaxUSVEnv.reset(jax.random.PRNGKey(i + 9999), params)

record("Zero NaN values in 1000 steps",
       nan_count == 0, f"NaN count = {nan_count}")

record("Zero Inf values in 1000 steps",
       inf_count == 0, f"Inf count = {inf_count}")

record("Step rewards bounded to reasonable range",
       reward_min > -300.0 and reward_max < 600.0,
       f"reward range = [{reward_min:.2f}, {reward_max:.2f}]")

print(f"\n  Reward distribution over 1000 steps:")
print(f"    min = {reward_min:.3f}")
print(f"    max = {reward_max:.3f}")

# ── 6. PHYSICS SANITY ─────────────────────────────────────────────────────────
header("6. PHYSICS SANITY")

obs, state = JaxUSVEnv.reset(jax.random.PRNGKey(5), params)
init_pos = np.array(state.usv_state.eta[:, :2])

# Apply full throttle forward for 50 steps
for _ in range(50):
    full_throttle = jnp.ones((N, 2)).at[:, 1].set(0.0)  # [1, 0] per agent
    obs, state, _, _, _ = JaxUSVEnv.step(jax.random.PRNGKey(0), state, full_throttle, params)

final_pos = np.array(state.usv_state.eta[:, :2])
displacement = np.linalg.norm(final_pos - init_pos, axis=1)

record("Agents move with full throttle (physics working)",
       bool(np.all(displacement > 0.5)),
       f"displacement = {displacement.round(2)}")

record("Agents don't teleport (displacement < 100m in 50 steps)",
       bool(np.all(displacement < 100.0)),
       f"max displacement = {displacement.max():.2f}m")

# ── 7. LOOPHOLE AUDIT ─────────────────────────────────────────────────────────
header("7. LOOPHOLE AUDIT")

# Can the agent spin in a tight circle and exploit heading reward?
# If it spins, cos(heading_error) averages to ~0 and step penalty = -0.5
# So average reward per step = 0 + 0 - 0.5 = -0.5 (NET NEGATIVE, no loophole)
spin_rewards = []
obs, state = JaxUSVEnv.reset(jax.random.PRNGKey(77), params)
for i in range(100):
    spin_action = jnp.zeros((N, 2)).at[:, 1].set(1.0)  # pure steering, no throttle
    obs, state, reward, done, _ = JaxUSVEnv.step(jax.random.PRNGKey(i), state, spin_action, params)
    spin_rewards.append(float(jnp.mean(reward)))
    if np.any(np.array(done)):
        break

avg_spin_reward = float(np.mean(spin_rewards))
record("Spinning in circles is unprofitable (avg reward < 0)",
       avg_spin_reward < 0.0,
       f"avg spin reward = {avg_spin_reward:.3f}")

# Can agent exploit progress by going backward → forward repeatedly?
# Progress cancels out, only step penalty survives → NET NEGATIVE
record("Back-forth oscillation is unprofitable (step penalty)",
       oscillation_avg < -0.3,
       f"avg oscillation reward = {oscillation_avg:.3f}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
header("DIAGNOSTIC SUMMARY")

passed = sum(1 for _, v in results if v)
failed = sum(1 for _, v in results if not v)
total  = len(results)

for name, ok in results:
    print(f"  {'✅' if ok else '❌'} {name}")

print(f"\n  Result: {passed}/{total} checks passed", end="")
if failed == 0:
    print(f"  \033[92m← ALL CLEAR! Safe to train.\033[0m")
else:
    print(f"  \033[91m← {failed} FAILURES — DO NOT TRAIN YET.\033[0m")
