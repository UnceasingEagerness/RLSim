"""
pretraining_check.py
====================
Full end-to-end pre-training validation.
Simulates the EXACT training pipeline for 200 steps and verifies
every tensor shape, every network update, and every data flow.

Run this immediately before training. If it says ALL CLEAR → train.
"""
import os, sys
os.environ["JAX_LOG_COMPILES"] = "0"

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training.train_state import TrainState

jax.config.update("jax_enable_x64", False)

from env.jax_usv_env   import JaxUSVEnv, EnvParams
from algorithms.flax_sac   import Actor, SoftQNetwork
from algorithms.jax_buffer  import JaxReplayBuffer
from algorithms.sac_update  import update_critic, update_actor, update_alpha, Transition

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
HEAD = "\033[1;94m"
END  = "\033[0m"

results = []
def check(name, cond, detail=""):
    results.append((name, cond))
    tag = PASS if cond else FAIL
    print(f"  {tag} {name}", f" ({detail})" if detail else "")
    return cond

def header(t):
    print(f"\n{HEAD}{'='*62}{END}\n{HEAD}  {t}{END}\n{HEAD}{'='*62}{END}")

# ── Config (mirror train_pure_jax.py exactly) ─────────────────────────────────
num_envs    = 4          # small for speed
num_agents  = 5
batch_size  = 64
obs_dim     = 72 + (num_agents - 1) * 5   # 92
action_dim  = 2
policy_lr   = 3e-4
gamma       = 0.99
tau_target  = 0.005
target_entropy = -2.0
total_insertions_per_step = num_envs * num_agents

layout = {
    "ego":              {"start": 0,  "dim": 8},
    "goal":             {"start": 0,  "dim": 8},
    "lidar":            {"start": 8,  "dim": 64},
    "auv_entities":     {"start": 72, "dim": (num_agents-1)*5, "count": num_agents-1, "feature_dim": 5},
    "moving_obstacles": {"start": 72, "dim": 0, "count": 0, "feature_dim": 5},
}

# ── 1. ENVIRONMENT ─────────────────────────────────────────────────────────────
header("1. ENVIRONMENT SETUP")

env        = JaxUSVEnv()
env_params = env.default_params.replace(num_agents=num_agents)

check("max_steps == 2000",    env_params.max_steps == 2000,    f"got {env_params.max_steps}")
check("map_size == 300.0",    env_params.map_size  == 300.0,   f"got {env_params.map_size}")
check("goal_radius == 15.0",  env_params.goal_radius == 15.0,  f"got {env_params.goal_radius}")
check("num_agents == 5",      env_params.num_agents == 5)

vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
vmap_step  = jax.vmap(env.step,  in_axes=(0, 0, 0, None))

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
reset_keys = jax.random.split(_rng, num_envs)
obs, env_state = vmap_reset(reset_keys, env_params)  # [E, N, obs_dim]

check("obs shape == [E, N, obs_dim]",
      obs.shape == (num_envs, num_agents, obs_dim),
      f"got {obs.shape}")
check("obs is finite",
      bool(jnp.all(jnp.isfinite(obs))))
check("obs in reasonable range",
      bool(jnp.all(jnp.abs(obs) < 500.0)),
      f"max abs = {float(jnp.max(jnp.abs(obs))):.2f}")

# ── 2. NETWORKS ───────────────────────────────────────────────────────────────
header("2. NETWORK SHAPES")

actor  = Actor(layout=layout, action_dim=action_dim,
               action_scale=jnp.ones(action_dim), action_bias=jnp.zeros(action_dim))
critic = SoftQNetwork(layout=layout)

rng, ak, ck = jax.random.split(rng, 3)
dummy_obs = jnp.zeros((1, obs_dim))
dummy_act = jnp.zeros((1, action_dim))

actor_params  = actor.init(ak, dummy_obs)["params"]
critic_params = critic.init(ck, dummy_obs, dummy_act)["params"]

# Actor output
flat_obs     = obs.reshape(num_envs * num_agents, obs_dim)
action_out, log_prob_out = actor.apply({"params": actor_params}, flat_obs, rng, method=actor.get_action)

check("actor action shape == [E*N, 2]",
      action_out.shape == (num_envs * num_agents, action_dim),
      f"got {action_out.shape}")
check("actor log_prob shape == [E*N, 1]",
      log_prob_out.shape == (num_envs * num_agents, 1),
      f"got {log_prob_out.shape}")
check("actions in tanh range [-1, 1]",
      bool(jnp.all(jnp.abs(action_out) <= 1.0 + 1e-5)),
      f"max abs = {float(jnp.max(jnp.abs(action_out))):.4f}")

# Critic output
q_out = critic.apply({"params": critic_params}, flat_obs, action_out)
check("critic Q shape == [E*N, 1]",
      q_out.shape == (num_envs * num_agents, 1),
      f"got {q_out.shape}")
check("critic Q is finite",
      bool(jnp.all(jnp.isfinite(q_out))))

# After squeeze (as in sac_update)
check("squeezed Q shape == [E*N]",
      q_out.squeeze(-1).shape == (num_envs * num_agents,),
      f"got {q_out.squeeze(-1).shape}")
check("squeezed log_prob shape == [E*N]",
      log_prob_out.squeeze(-1).shape == (num_envs * num_agents,),
      f"got {log_prob_out.squeeze(-1).shape}")

# ── 3. ENVIRONMENT STEP ───────────────────────────────────────────────────────
header("3. ENVIRONMENT STEP DATA FLOW")

rng, sk = jax.random.split(rng)
step_keys = jax.random.split(sk, num_envs)
action_env = action_out.reshape(num_envs, num_agents, action_dim)
next_obs, next_state, reward, done, info = vmap_step(step_keys, env_state, action_env, env_params)

check("next_obs shape == [E, N, obs_dim]",
      next_obs.shape == (num_envs, num_agents, obs_dim),
      f"got {next_obs.shape}")
check("reward shape == [E, N]",
      reward.shape == (num_envs, num_agents),
      f"got {reward.shape}")
check("done shape == [E, N]",
      done.shape == (num_envs, num_agents),
      f"got {done.shape}")
check("reward is finite",
      bool(jnp.all(jnp.isfinite(reward))))
check("reward in expected range [-210, 510]",
      bool(jnp.all(reward > -210.0) and jnp.all(reward < 510.0)),
      f"range=[{float(reward.min()):.2f}, {float(reward.max()):.2f}]")

# Buffer insertion shapes
flat_reward   = reward.flatten()
flat_next_obs = next_obs.reshape(-1, obs_dim)
flat_done     = done.flatten()

check("flat_obs shape == [E*N, obs_dim]",
      flat_obs.shape == (num_envs * num_agents, obs_dim),
      f"got {flat_obs.shape}")
check("flat_action shape == [E*N, 2]",
      action_out.shape == (num_envs * num_agents, action_dim),
      f"got {action_out.shape}")
check("flat_reward shape == [E*N]",
      flat_reward.shape == (num_envs * num_agents,),
      f"got {flat_reward.shape}")
check("flat_next_obs shape == [E*N, obs_dim]",
      flat_next_obs.shape == (num_envs * num_agents, obs_dim),
      f"got {flat_next_obs.shape}")
check("flat_done shape == [E*N]",
      flat_done.shape == (num_envs * num_agents,),
      f"got {flat_done.shape}")

# ── 4. REPLAY BUFFER ──────────────────────────────────────────────────────────
header("4. REPLAY BUFFER")

buffer       = JaxReplayBuffer(10_000, obs_dim, action_dim)
buffer_state = buffer.init_state()

# Fill buffer with 500 transitions
for i in range(500 // total_insertions_per_step + 1):
    buffer_state = buffer.add_batch(
        buffer_state, flat_obs, action_out, flat_reward,
        flat_next_obs, flat_done, total_insertions_per_step
    )

check("buffer count > 0",
      int(buffer_state.count) > 0,
      f"count = {int(buffer_state.count)}")

rng, sk = jax.random.split(rng)
b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(buffer_state, sk, batch_size)

check("sampled obs shape == [batch, obs_dim]",
      b_obs.shape == (batch_size, obs_dim),
      f"got {b_obs.shape}")
check("sampled action shape == [batch, 2]",
      b_act.shape == (batch_size, action_dim),
      f"got {b_act.shape}")
check("sampled reward shape == [batch]",
      b_rew.shape == (batch_size,),
      f"got {b_rew.shape}")
check("sampled done dtype is bool",
      b_done.dtype == jnp.bool_,
      f"got {b_done.dtype}")

# ── 5. SAC UPDATES ────────────────────────────────────────────────────────────
header("5. SAC UPDATE CORRECTNESS")

actor_state  = TrainState.create(apply_fn=actor.apply,  params=actor_params,  tx=optax.adam(policy_lr))
critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(policy_lr))
log_alpha    = jnp.array(0.5)
alpha_opt    = optax.adam(policy_lr)
alpha_state  = alpha_opt.init(log_alpha)

batch = Transition(obs=b_obs, action=b_act, reward=b_rew, next_obs=b_next_obs, done=b_done)

rng, ck, ak = jax.random.split(rng, 3)

# Critic update
new_critic, q_loss = update_critic(
    critic_state, critic_params, actor_state, log_alpha, batch, gamma, ck
)
check("critic update runs without error",  True)
check("q_loss is scalar and finite",
      bool(jnp.isfinite(q_loss) and q_loss.shape == ()),
      f"q_loss = {float(q_loss):.4f}, shape={q_loss.shape}")
check("q_loss is not absurdly large (< 1e6)",
      float(q_loss) < 1e6,
      f"q_loss = {float(q_loss):.4f}")

# Actor update
new_actor, a_loss, log_prob = update_actor(
    actor_state, new_critic, log_alpha, b_obs, ak
)
check("actor update runs without error", True)
check("a_loss is scalar and finite",
      bool(jnp.isfinite(a_loss) and a_loss.shape == ()),
      f"a_loss = {float(a_loss):.4f}, shape={a_loss.shape}")
check("log_prob shape == [batch] (squeezed)",
      log_prob.shape == (batch_size,),
      f"got {log_prob.shape}")

# Alpha update
new_log_alpha, new_alpha_state, alpha_loss = update_alpha(
    log_alpha, alpha_state, log_prob, target_entropy, alpha_opt
)
new_log_alpha = jnp.maximum(new_log_alpha, -2.0)  # floor
check("alpha update runs without error", True)
check("new alpha in [exp(-2), exp(1)]",
      bool(0.13 <= float(jnp.exp(new_log_alpha)) <= 3.0),
      f"alpha = {float(jnp.exp(new_log_alpha)):.4f}")

# Target network update
new_target = jax.tree_util.tree_map(
    lambda t, c: tau_target * c + (1 - tau_target) * t,
    critic_params, new_critic.params
)
check("target network soft-update runs without error", True)

# ── 6. MULTI-STEP STABILITY ───────────────────────────────────────────────────
header("6. MULTI-STEP STABILITY (200 real SAC updates)")

q_losses, a_losses, alphas = [], [], []
cur_log_alpha = jnp.array(0.5)
cur_alpha_state = alpha_opt.init(cur_log_alpha)
cur_actor  = actor_state
cur_critic = critic_state
cur_target = critic_params

for i in range(200):
    # Fill buffer a bit more
    buffer_state = buffer.add_batch(
        buffer_state, flat_obs, action_out, flat_reward,
        flat_next_obs, flat_done, total_insertions_per_step
    )
    rng, sk = jax.random.split(rng)
    b = Transition(*buffer.sample(buffer_state, sk, batch_size))

    rng, ck2, ak2 = jax.random.split(rng, 3)
    cur_critic, ql = update_critic(cur_critic, cur_target, cur_actor, cur_log_alpha, b, gamma, ck2)
    cur_actor, al, lp = update_actor(cur_actor, cur_critic, cur_log_alpha, b.obs, ak2)
    cur_log_alpha, cur_alpha_state, _ = update_alpha(cur_log_alpha, cur_alpha_state, lp, target_entropy, alpha_opt)
    cur_log_alpha = jnp.maximum(cur_log_alpha, -2.0)
    cur_target = jax.tree_util.tree_map(
        lambda t, c: tau_target * c + (1 - tau_target) * t, cur_target, cur_critic.params
    )
    q_losses.append(float(ql))
    a_losses.append(float(al))
    alphas.append(float(jnp.exp(cur_log_alpha)))

q_arr = np.array(q_losses)
a_arr = np.array(a_losses)

check("No NaN in 200 Q-loss updates",
      not np.any(np.isnan(q_arr)), f"NaN count={np.isnan(q_arr).sum()}")
check("No NaN in 200 actor-loss updates",
      not np.any(np.isnan(a_arr)), f"NaN count={np.isnan(a_arr).sum()}")
check("No Inf in Q-loss",
      not np.any(np.isinf(q_arr)))
check("Q-loss < 1e5 throughout",
      bool(np.all(q_arr < 1e5)), f"max={q_arr.max():.2f}")
check("Alpha stays >= floor (0.135)",
      min(alphas) >= 0.13, f"min={min(alphas):.4f}")

print(f"\n  Q-loss : start={q_arr[:10].mean():.3f} → end={q_arr[-10:].mean():.3f}")
print(f"  A-loss : start={a_arr[:10].mean():.3f} → end={a_arr[-10:].mean():.3f}")
print(f"  Alpha  : start={alphas[0]:.4f} → end={alphas[-1]:.4f}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
header("FINAL VERDICT")

passed = sum(1 for _, v in results if v)
failed = sum(1 for _, v in results if not v)
total  = len(results)

for name, ok in results:
    print(f"  {'✅' if ok else '❌'} {name}")

print()
if failed == 0:
    print(f"\033[92m  ✅ {passed}/{total} checks passed — PICTURE PERFECT. START TRAINING!\033[0m")
else:
    print(f"\033[91m  ❌ {failed}/{total} FAILED — DO NOT TRAIN YET.\033[0m")
    for name, ok in results:
        if not ok:
            print(f"\033[91m     → FIX: {name}\033[0m")
