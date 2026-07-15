import os
import time
import jax
import jax.numpy as jnp
import optax
import flax
import numpy as np
import pandas as pd
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

from env.jax_pe_env import JaxPursuitEvasionEnv
from algorithms.pe_flax_sac import PursuerActor, PursuerCritic, EvaderActor, EvaderCritic
from algorithms.jax_buffer import JaxReplayBuffer

# -----------------------------------------------------------------------------
# JAX Structs
# -----------------------------------------------------------------------------
class Transition(flax.struct.PyTreeNode):
    local_obs: jnp.ndarray
    global_obs: jnp.ndarray
    action: jnp.ndarray
    joint_actions: jnp.ndarray
    reward: jnp.ndarray
    next_local_obs: jnp.ndarray
    next_global_obs: jnp.ndarray
    done: jnp.ndarray

@flax.struct.dataclass
class PE_RunnerState:
    env_state: Any
    obs_history: jnp.ndarray # [E, 4, 10, 77]
    episode_return: jnp.ndarray # [E, 4]
    
    p_actor_state: TrainState
    p_critic_state: TrainState
    p_target_critic_params: Any
    p_log_alpha: jnp.ndarray
    p_alpha_opt_state: optax.OptState
    p_buffer_state: Any
    
    e_actor_state: TrainState
    e_critic_state: TrainState
    e_target_critic_params: Any
    e_log_alpha: jnp.ndarray
    e_alpha_opt_state: optax.OptState
    e_buffer_state: Any
    
    rng: jax.random.PRNGKey
    step_count: int

# -----------------------------------------------------------------------------
# SAC Updates
# -----------------------------------------------------------------------------
def get_pursuer_updates(p_actor, p_critic, e_actor, tau_target=0.005, gamma=0.99, target_entropy=-2.0):
    def update_critic(critic_state, target_critic_params, p_actor_state, e_actor_state, log_alpha, batch, key):
        # batch has local_obs [B, 10, 77], global_obs [B, 10, 308], joint_actions [B, 8]
        # We need to sample actions from ALL agents for the Target Q
        p1_obs = batch.next_global_obs[:, :, 0:77]
        p2_obs = batch.next_global_obs[:, :, 77:154]
        p3_obs = batch.next_global_obs[:, :, 154:231]
        e_obs = batch.next_global_obs[:, :, 231:308]
        
        # Pursuer Next Actions
        def get_act(obs, actor_cls, state, rng):
            m, ls = actor_cls.apply({"params": state.params}, obs)
            s = jnp.exp(ls)
            n = jax.random.normal(rng, m.shape)
            a = jnp.tanh(m + n * s)
            lp = -0.5 * jnp.sum(jnp.square(n) + 2.0*ls + jnp.log(2.0*jnp.pi), axis=-1, keepdims=True)
            lp -= jnp.sum(2.0 * (jnp.log(2.0) - a - jax.nn.softplus(-2.0*a)), axis=-1, keepdims=True)
            return a, lp
            
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p1_a, p1_lp = get_act(p1_obs, p_actor, p_actor_state, k1)
        p2_a, p2_lp = get_act(p2_obs, p_actor, p_actor_state, k2)
        p3_a, p3_lp = get_act(p3_obs, p_actor, p_actor_state, k3)
        e_a, _ = get_act(e_obs, e_actor, e_actor_state, k4)
        
        next_joint_actions = jnp.concatenate([p1_a, p2_a, p3_a, e_a], axis=-1)
        
        # We only subtract entropy for the ego agent (since this is decentralized execution)
        # But wait, local_obs could be p1, p2, or p3 because the buffer contains all transitions.
        # We can just extract the ego log_prob by recalculating it for batch.next_local_obs
        ego_a, ego_lp = get_act(batch.next_local_obs, p_actor, p_actor_state, k1)
        
        q1_target, q2_target = p_critic.apply({"params": target_critic_params}, batch.next_global_obs, next_joint_actions)
        q_target_min = jnp.minimum(q1_target, q2_target)
        
        alpha = jnp.exp(log_alpha)
        target_q = batch.reward[:, None] + gamma * (1.0 - batch.done[:, None]) * (q_target_min - alpha * ego_lp)
        
        def critic_loss_fn(params):
            q1, q2 = p_critic.apply({"params": params}, batch.global_obs, batch.joint_actions)
            loss = jnp.mean(jnp.square(q1 - target_q) + jnp.square(q2 - target_q))
            return loss, (q1, q2)
            
        (loss, _), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        return new_critic_state, loss

    def update_actor(actor_state, critic_state, log_alpha, batch, key):
        def actor_loss_fn(params):
            # To properly evaluate the Centralized Critic for a Shared Policy, we must evaluate the actions of ALL 3 pursuers simultaneously.
            p1_obs = batch.global_obs[:, :, 0:77]
            p2_obs = batch.global_obs[:, :, 77:154]
            p3_obs = batch.global_obs[:, :, 154:231]
            
            def get_act_and_lp(obs, rng):
                m, ls = p_actor.apply({"params": params}, obs)
                s = jnp.exp(ls)
                n = jax.random.normal(rng, m.shape)
                a = jnp.tanh(m + n * s)
                lp = -0.5 * jnp.sum(jnp.square(n) + 2.0*ls + jnp.log(2.0*jnp.pi), axis=-1, keepdims=True)
                lp -= jnp.sum(2.0 * (jnp.log(2.0) - a - jax.nn.softplus(-2.0*a)), axis=-1, keepdims=True)
                return a, lp
                
            k1, k2, k3 = jax.random.split(key, 3)
            a1, lp1 = get_act_and_lp(p1_obs, k1)
            a2, lp2 = get_act_and_lp(p2_obs, k2)
            a3, lp3 = get_act_and_lp(p3_obs, k3)
            
            # Construct true Joint Action using the live policy for all 3 pursuers, while keeping the Evader's buffer action
            joint_actions = jnp.concatenate([a1, a2, a3, batch.joint_actions[:, 6:8]], axis=-1)
            
            q1, q2 = p_critic.apply({"params": critic_state.params}, batch.global_obs, joint_actions)
            q_min = jnp.minimum(q1, q2)
            
            # Mean log prob across the 3 pursuers
            mean_lp = (lp1 + lp2 + lp3) / 3.0
            
            alpha = jnp.exp(log_alpha)
            # Maximize Q while maximizing entropy
            loss = jnp.mean(alpha * mean_lp - q_min)
            return loss, jnp.mean(mean_lp)
            
        (loss, log_prob_mean), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=grads)
        return new_actor_state, loss, log_prob_mean

    def update_alpha_fn(log_alpha, alpha_opt_state, log_prob_mean, optimizer):
        def alpha_loss_fn(log_alpha_param):
            return jnp.mean(-jnp.exp(log_alpha_param) * (log_prob_mean + target_entropy))
        loss, grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        updates, new_alpha_opt_state = optimizer.update(grads, alpha_opt_state, log_alpha)
        return optax.apply_updates(log_alpha, updates), new_alpha_opt_state, loss
        
    return update_critic, update_actor, update_alpha_fn

def get_evader_updates(e_actor, e_critic, p_actor, tau_target=0.005, gamma=0.99, target_entropy=-2.0):
    def update_critic(critic_state, target_critic_params, e_actor_state, p_actor_state, log_alpha, batch, key):
        p1_obs = batch.next_global_obs[:, :, 0:77]
        p2_obs = batch.next_global_obs[:, :, 77:154]
        p3_obs = batch.next_global_obs[:, :, 154:231]
        e_obs = batch.next_global_obs[:, :, 231:308]
        
        def get_act(obs, actor_cls, state, rng):
            m, ls = actor_cls.apply({"params": state.params}, obs)
            a = jnp.tanh(m + jax.random.normal(rng, m.shape) * jnp.exp(ls))
            return a
            
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p1_a = get_act(p1_obs, p_actor, p_actor_state, k1)
        p2_a = get_act(p2_obs, p_actor, p_actor_state, k2)
        p3_a = get_act(p3_obs, p_actor, p_actor_state, k3)
        
        # Ego Evader Action
        m, ls = e_actor.apply({"params": e_actor_state.params}, batch.next_local_obs)
        n = jax.random.normal(k4, m.shape)
        e_a = jnp.tanh(m + n * jnp.exp(ls))
        lp = -0.5 * jnp.sum(jnp.square(n) + 2.0*ls + jnp.log(2.0*jnp.pi), axis=-1, keepdims=True)
        lp -= jnp.sum(2.0 * (jnp.log(2.0) - e_a - jax.nn.softplus(-2.0*e_a)), axis=-1, keepdims=True)
        
        next_joint_actions = jnp.concatenate([p1_a, p2_a, p3_a, e_a], axis=-1)
        
        q1_target, q2_target = e_critic.apply({"params": target_critic_params}, batch.next_global_obs, next_joint_actions)
        q_target_min = jnp.minimum(q1_target, q2_target)
        
        alpha = jnp.exp(log_alpha)
        target_q = batch.reward[:, None] + gamma * (1.0 - batch.done[:, None]) * (q_target_min - alpha * lp)
        
        def critic_loss_fn(params):
            q1, q2 = e_critic.apply({"params": params}, batch.global_obs, batch.joint_actions)
            loss = jnp.mean(jnp.square(q1 - target_q) + jnp.square(q2 - target_q))
            return loss, (q1, q2)
            
        (loss, _), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        return new_critic_state, loss

    def update_actor(actor_state, critic_state, log_alpha, batch, key):
        def actor_loss_fn(params):
            m, ls = e_actor.apply({"params": params}, batch.local_obs)
            n = jax.random.normal(key, m.shape)
            a = jnp.tanh(m + n * jnp.exp(ls))
            
            lp = -0.5 * jnp.sum(jnp.square(n) + 2.0*ls + jnp.log(2.0*jnp.pi), axis=-1, keepdims=True)
            lp -= jnp.sum(2.0 * (jnp.log(2.0) - a - jax.nn.softplus(-2.0*a)), axis=-1, keepdims=True)
            
            joint_actions = jnp.concatenate([batch.joint_actions[:, 0:6], a], axis=-1)
            
            q1, q2 = e_critic.apply({"params": critic_state.params}, batch.global_obs, joint_actions)
            q_min = jnp.minimum(q1, q2)
            
            alpha = jnp.exp(log_alpha)
            loss = jnp.mean(alpha * lp - q_min)
            return loss, jnp.mean(lp)
            
        (loss, log_prob_mean), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=grads)
        return new_actor_state, loss, log_prob_mean

    def update_alpha_fn(log_alpha, alpha_opt_state, log_prob_mean, optimizer):
        def alpha_loss_fn(log_alpha_param):
            return jnp.mean(-jnp.exp(log_alpha_param) * (log_prob_mean + target_entropy))
        loss, grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        updates, new_alpha_opt_state = optimizer.update(grads, alpha_opt_state, log_alpha)
        return optax.apply_updates(log_alpha, updates), new_alpha_opt_state, loss
        
    return update_critic, update_actor, update_alpha_fn

def print_rich_config(num_envs, num_pursuers, num_evaders, batch_size, obs_pack_dim, env_params):
    console.print(Panel.fit("[bold cyan]RLSim V2: Massive JAX Adversarial Training (Pursuit-Evasion)[/bold cyan]", border_style="cyan"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim", width=20)
    table.add_column("Parameter", width=25)
    table.add_column("Value", justify="right", style="green")
    
    # Engine
    table.add_row("Engine", "Hardware Accelerator", str(jax.devices()[0]))
    table.add_row("Engine", "JIT Mode", "jax.lax.scan + vmap (Dual)")
    table.add_row("Engine", "Parallel Envs (E)", str(num_envs))
    table.add_row("Engine", "Pursuers per Env", str(num_pursuers))
    table.add_row("Engine", "Evaders per Env", str(num_evaders))
    
    # Environment
    table.add_section()
    table.add_row("Environment", "Max Steps per Ep", str(env_params.max_steps))
    table.add_row("Environment", "Map Size", f"{env_params.map_size}m")
    
    # Agent
    table.add_section()
    table.add_row("Agent (Dual SAC)", "Packed Obs Dim", str(obs_pack_dim))
    table.add_row("Agent (Dual SAC)", "Batch Size", str(batch_size))
    
    console.print(table)
    console.print("\n[bold yellow]Allocating Dual Replay Buffers on GPU...[/bold yellow]")

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    
    num_envs = 64
    num_pursuers = 3
    num_evaders = 1
    num_agents = num_pursuers + num_evaders
    total_timesteps = 100_000
    learning_starts = 2_000
    batch_size = 128
    buffer_size = 100_000
    gamma = 0.99
    tau_target = 0.005
    policy_lr = 3e-4
    q_lr = 3e-4
    
    seq_len = 10
    local_dim = 77
    global_dim = 4 * 77
    action_dim = 2
    joint_action_dim = 4 * 2
    
    # Custom buffer element dimension: local_obs(770) + global_obs(3080) + joint_action(8) = 3858
    obs_pack_dim = (local_dim * seq_len) + (global_dim * seq_len) + joint_action_dim
    
    env = JaxPursuitEvasionEnv()
    env_params = env.default_params.replace(map_size=1000.0, num_obstacles=50)
    
    print_rich_config(num_envs, num_pursuers, num_evaders, batch_size, obs_pack_dim, env_params)
    
    rng = jax.random.PRNGKey(42)
    
    p_actor = PursuerActor(action_dim=2)
    p_critic = PursuerCritic()
    e_actor = EvaderActor(action_dim=2)
    e_critic = EvaderCritic()
    
    rng, key1, key2, key3, key4 = jax.random.split(rng, 5)
    
    p_a_params = p_actor.init(key1, jnp.zeros((1, seq_len, local_dim)))["params"]
    p_c_params = p_critic.init(key2, jnp.zeros((1, seq_len, global_dim)), jnp.zeros((1, joint_action_dim)))["params"]
    
    e_a_params = e_actor.init(key3, jnp.zeros((1, seq_len, local_dim)))["params"]
    e_c_params = e_critic.init(key4, jnp.zeros((1, seq_len, global_dim)), jnp.zeros((1, joint_action_dim)))["params"]
    
    p_actor_state = TrainState.create(apply_fn=p_actor.apply, params=p_a_params, tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(policy_lr)))
    p_critic_state = TrainState.create(apply_fn=p_critic.apply, params=p_c_params, tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(q_lr)))
    e_actor_state = TrainState.create(apply_fn=e_actor.apply, params=e_a_params, tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(policy_lr)))
    e_critic_state = TrainState.create(apply_fn=e_critic.apply, params=e_c_params, tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(q_lr)))
    
    alpha_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(policy_lr))
    
    p_buffer = JaxReplayBuffer(buffer_size, obs_pack_dim, action_dim)
    e_buffer = JaxReplayBuffer(buffer_size, obs_pack_dim, action_dim)
    
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    
    rng, _rng = jax.random.split(rng)
    reset_keys = jax.random.split(_rng, num_envs)
    
    # Init envs
    # In pursuit evasion, get_obs returns [4, 77]
    init_obs, init_env_state = vmap_reset(reset_keys, env_params) # [E, 4, 77]
    # We need a history buffer: [E, 4, 10, 77]
    init_obs_history = jnp.repeat(jnp.expand_dims(init_obs, axis=2), seq_len, axis=2)
    
    runner_state = PE_RunnerState(
        env_state=init_env_state,
        obs_history=init_obs_history,
        episode_return=jnp.zeros((num_envs, 4)),
        
        p_actor_state=p_actor_state,
        p_critic_state=p_critic_state,
        p_target_critic_params=p_c_params,
        p_log_alpha=jnp.array(0.0),
        p_alpha_opt_state=alpha_optimizer.init(jnp.array(0.0)),
        p_buffer_state=p_buffer.init_state(),
        
        e_actor_state=e_actor_state,
        e_critic_state=e_critic_state,
        e_target_critic_params=e_c_params,
        e_log_alpha=jnp.array(0.0),
        e_alpha_opt_state=alpha_optimizer.init(jnp.array(0.0)),
        e_buffer_state=e_buffer.init_state(),
        
        rng=rng,
        step_count=0
    )
    
    # Update functions
    p_update_c, p_update_a, p_update_alpha = get_pursuer_updates(p_actor, p_critic, e_actor, tau_target, gamma, -2.0)
    e_update_c, e_update_a, e_update_alpha = get_evader_updates(e_actor, e_critic, p_actor, tau_target, gamma, -2.0)
    
    def _step_fn(rs: PE_RunnerState, _):
        rng, action_key, step_key, sample_key, up_key = jax.random.split(rs.rng, 5)
        
        # --- 1. Action Selection ---
        def get_actions(actor_cls, actor_state, obs_hist):
            # obs_hist is [B, 10, 77]
            mean, _ = actor_cls.apply({"params": actor_state.params}, obs_hist)
            return jnp.tanh(mean)
            
        def explore_actions():
            return jax.random.uniform(action_key, shape=(num_envs, 4, 2), minval=-1.0, maxval=1.0)
            
        def exploit_actions():
            flat_obs = rs.obs_history.reshape(-1, 10, 77)
            # Separate pursuer vs evader obs
            p_obs = flat_obs.reshape(num_envs, 4, 10, 77)[:, :3].reshape(-1, 10, 77)
            e_obs = flat_obs.reshape(num_envs, 4, 10, 77)[:, 3].reshape(-1, 10, 77)
            
            p_act = get_actions(p_actor, rs.p_actor_state, p_obs).reshape(num_envs, 3, 2)
            e_act = get_actions(e_actor, rs.e_actor_state, e_obs).reshape(num_envs, 1, 2)
            return jnp.concatenate([p_act, e_act], axis=1)
            
        actions = jax.lax.cond(rs.step_count < learning_starts, explore_actions, exploit_actions) # [E, 4, 2]
        
        # --- 2. Environment Step ---
        step_keys = jax.random.split(step_key, num_envs)
        next_obs_t, next_env_state, reward, done, info = vmap_step(step_keys, rs.env_state, actions, env_params)
        
        # Scale reward to stabilize Critic gradients
        scaled_reward = reward * 0.01
        
        # Update history
        next_obs_history = jnp.concatenate([rs.obs_history[:, :, 1:, :], jnp.expand_dims(next_obs_t, axis=2)], axis=2)
        
        # --- 3. Pack data for buffers ---
        # Pack global obs: [E, 10, 4*77] -> [E, 3080]
        global_obs_flat = rs.obs_history.reshape(num_envs, 10, -1).reshape(num_envs, -1)
        next_global_obs_flat = next_obs_history.reshape(num_envs, 10, -1).reshape(num_envs, -1)
        
        joint_actions_flat = actions.reshape(num_envs, -1) # [E, 8]
        
        # Create massive packed obs: local(770) + global(3080) + joint_act(8) = 3858
        def pack_obs(local_hist, global_hist, joint_act):
            return jnp.concatenate([local_hist.reshape(-1), global_hist, joint_act], axis=-1)
            
        # Pursuers (3 * E transitions)
        p_local_hists = rs.obs_history[:, :3].reshape(num_envs * 3, 10, 77)
        p_next_local_hists = next_obs_history[:, :3].reshape(num_envs * 3, 10, 77)
        p_global_repeated = jnp.repeat(global_obs_flat, 3, axis=0)
        p_joint_repeated = jnp.repeat(joint_actions_flat, 3, axis=0)
        
        p_pack = jax.vmap(pack_obs)(p_local_hists, p_global_repeated, p_joint_repeated)
        p_next_pack = jax.vmap(pack_obs)(p_next_local_hists, jnp.repeat(next_global_obs_flat, 3, axis=0), p_joint_repeated)
        
        new_p_buffer = p_buffer.add_batch(rs.p_buffer_state, p_pack, actions[:, :3].reshape(-1, 2), scaled_reward[:, :3].reshape(-1), p_next_pack, done[:, :3].reshape(-1), num_envs * 3)
        
        # Evaders (1 * E transitions)
        e_local_hists = rs.obs_history[:, 3].reshape(num_envs * 1, 10, 77)
        e_next_local_hists = next_obs_history[:, 3].reshape(num_envs * 1, 10, 77)
        
        e_pack = jax.vmap(pack_obs)(e_local_hists, global_obs_flat, joint_actions_flat)
        e_next_pack = jax.vmap(pack_obs)(e_next_local_hists, next_global_obs_flat, joint_actions_flat)
        
        new_e_buffer = e_buffer.add_batch(rs.e_buffer_state, e_pack, actions[:, 3].reshape(-1, 2), scaled_reward[:, 3].reshape(-1), e_next_pack, done[:, 3].reshape(-1), num_envs * 1)
        
        # --- 4. Alternating Optimization Update ---
        def unpack(batch_pack):
            # batch_pack is [B, 3858]
            local = batch_pack[:, :770].reshape(batch_size, 10, 77)
            glob = batch_pack[:, 770:3850].reshape(batch_size, 10, 308)
            j_act = batch_pack[:, 3850:]
            return local, glob, j_act
            
        def perform_updates():
            # Sample buffers
            pk1, pk2, ek1, ek2 = jax.random.split(sample_key, 4)
            
            p_obs, p_act, p_rew, p_next_obs, p_done = p_buffer.sample(new_p_buffer, pk1, batch_size)
            p_l, p_g, p_j = unpack(p_obs)
            pn_l, pn_g, pn_j = unpack(p_next_obs)
            p_batch = Transition(local_obs=p_l, global_obs=p_g, action=p_act, joint_actions=p_j, reward=p_rew, next_local_obs=pn_l, next_global_obs=pn_g, done=p_done)
            
            e_obs, e_act, e_rew, e_next_obs, e_done = e_buffer.sample(new_e_buffer, ek1, batch_size)
            e_l, e_g, e_j = unpack(e_obs)
            en_l, en_g, en_j = unpack(e_next_obs)
            e_batch = Transition(local_obs=e_l, global_obs=e_g, action=e_act, joint_actions=e_j, reward=e_rew, next_local_obs=en_l, next_global_obs=en_g, done=e_done)
            
            # Update Pursuers
            new_pc, pc_loss = p_update_c(rs.p_critic_state, rs.p_target_critic_params, rs.p_actor_state, rs.e_actor_state, rs.p_log_alpha, p_batch, pk2)
            new_pa, pa_loss, pa_log_prob = p_update_a(rs.p_actor_state, new_pc, rs.p_log_alpha, p_batch, pk2)
            new_p_alpha, new_p_opt, _ = p_update_alpha(rs.p_log_alpha, rs.p_alpha_opt_state, pa_log_prob, alpha_optimizer)
            new_ptc = jax.tree_util.tree_map(lambda t, c: tau_target * c + (1 - tau_target) * t, rs.p_target_critic_params, new_pc.params)
            
            # Update Evaders
            new_ec, ec_loss = e_update_c(rs.e_critic_state, rs.e_target_critic_params, rs.e_actor_state, rs.p_actor_state, rs.e_log_alpha, e_batch, ek2)
            new_ea, ea_loss, ea_log_prob = e_update_a(rs.e_actor_state, new_ec, rs.e_log_alpha, e_batch, ek2)
            new_e_alpha, new_e_opt, _ = e_update_alpha(rs.e_log_alpha, rs.e_alpha_opt_state, ea_log_prob, alpha_optimizer)
            new_etc = jax.tree_util.tree_map(lambda t, c: tau_target * c + (1 - tau_target) * t, rs.e_target_critic_params, new_ec.params)
            
            return new_pa, new_pc, new_ptc, jnp.maximum(new_p_alpha, -2.0), new_p_opt, pa_loss, new_ea, new_ec, new_etc, jnp.maximum(new_e_alpha, -2.0), new_e_opt, ea_loss
            
        def skip_updates():
            return rs.p_actor_state, rs.p_critic_state, rs.p_target_critic_params, rs.p_log_alpha, rs.p_alpha_opt_state, 0.0, rs.e_actor_state, rs.e_critic_state, rs.e_target_critic_params, rs.e_log_alpha, rs.e_alpha_opt_state, 0.0
            
        n_pa, n_pc, n_ptc, n_p_alp, n_p_opt, p_a_loss, n_ea, n_ec, n_etc, n_e_alp, n_e_opt, e_a_loss = jax.lax.cond(rs.step_count >= learning_starts, perform_updates, skip_updates)
        
        # Auto-reset envs
        env_dones = jnp.any(done, axis=1) # [E]
        reset_keys = jax.random.split(up_key, num_envs)
        res_obs, res_state = vmap_reset(reset_keys, env_params)
        res_obs_hist = jnp.repeat(jnp.expand_dims(res_obs, axis=2), seq_len, axis=2)
        
        final_obs_history = jnp.where(env_dones[:, None, None, None], res_obs_hist, next_obs_history)
        def merge(a, b):
            s = (num_envs,) + (1,) * (b.ndim - 1)
            return jnp.where(jnp.reshape(env_dones, s), a, b)
        final_state = jax.tree_util.tree_map(merge, res_state, next_env_state)
        
        new_rs = rs.replace(
            env_state=final_state,
            obs_history=final_obs_history,
            p_actor_state=n_pa, p_critic_state=n_pc, p_target_critic_params=n_ptc, p_log_alpha=n_p_alp, p_alpha_opt_state=n_p_opt, p_buffer_state=new_p_buffer,
            e_actor_state=n_ea, e_critic_state=n_ec, e_target_critic_params=n_etc, e_log_alpha=n_e_alp, e_alpha_opt_state=n_e_opt, e_buffer_state=new_e_buffer,
            rng=rng, step_count=rs.step_count+1
        )
        
        metrics = {
            "p_reward": jnp.mean(reward[:, :3]),
            "e_reward": jnp.mean(reward[:, 3]),
            "p_a_loss": p_a_loss,
            "e_a_loss": e_a_loss
        }
        return new_rs, metrics

    # Run Loop
    steps_per_epoch = 1000
    num_epochs = total_timesteps // steps_per_epoch
    
    with console.status("[bold yellow]Compiling massive dual JAX XLA Graph...[/bold yellow]", spinner="dots"):
        @jax.jit
        def run_epoch(rs):
            return jax.lax.scan(_step_fn, rs, None, length=steps_per_epoch)
            
    console.print("\n[bold green]Starting Adversarial Training Loop...[/bold green]")
    all_metrics = []
    
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        runner_state, metrics = run_epoch(runner_state)
        jax.block_until_ready(metrics["p_reward"])
        epoch_end = time.time()
        
        sps = (steps_per_epoch * num_envs * num_agents) / (epoch_end - epoch_start)
        p_r = np.mean(metrics["p_reward"])
        e_r = np.mean(metrics["e_reward"])
        p_loss = np.mean(metrics["p_a_loss"])
        e_loss = np.mean(metrics["e_a_loss"])
        
        console.print(f"Epoch {epoch+1:02d}/{num_epochs} | P Reward: [bold green]{p_r:>6.2f}[/bold green] | E Reward: [bold red]{e_r:>6.2f}[/bold red] | "
                      f"P Loss: {p_loss:>6.3f} | E Loss: {e_loss:>6.3f} | Speed: [bold magenta]{sps:,.0f} SPS[/bold magenta]")
                      
        all_metrics.append(metrics)

    end_time = time.time()
    total_sim_steps = total_timesteps * num_envs * num_agents
    avg_sps = total_sim_steps / (end_time - start_time)
    
    console.print(Panel.fit(f"[bold green]Adversarial Execution Complete![/bold green]\n"
                            f"Simulated {total_sim_steps:,} transitions in {end_time - start_time:.2f}s\n"
                            f"Avg Speed: {avg_sps:,.0f} SPS", border_style="green"))

    # ── Save Checkpoints & Metrics ────────────────────────────────────────────
    os.makedirs("logs_pe", exist_ok=True)
    os.makedirs("checkpoints_pe", exist_ok=True)
    
    with console.status("[bold yellow]Saving Models and Metrics...[/bold yellow]"):
        # Save metrics
        metrics_dict = {k: jnp.concatenate([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        metrics_df = pd.DataFrame({
            "step": np.arange(total_timesteps),
            "p_mean_reward": np.array(metrics_dict["p_reward"]),
            "e_mean_reward": np.array(metrics_dict["e_reward"]),
            "p_a_loss": np.array(metrics_dict["p_a_loss"]),
            "e_a_loss": np.array(metrics_dict["e_a_loss"])
        })
        metrics_df.to_csv("logs_pe/adversarial_metrics.csv", index=False)
        
        # Save checkpionts
        ckpt = PyTreeCheckpointer()
        ckpt.save(os.path.abspath("checkpoints_pe/pursuer_actor"), runner_state.p_actor_state.params, force=True)
        ckpt.save(os.path.abspath("checkpoints_pe/pursuer_critic"), runner_state.p_critic_state.params, force=True)
        ckpt.save(os.path.abspath("checkpoints_pe/evader_actor"), runner_state.e_actor_state.params, force=True)
        ckpt.save(os.path.abspath("checkpoints_pe/evader_critic"), runner_state.e_critic_state.params, force=True)
        
    console.print("[bold green]✔ Saved metrics to logs_pe/adversarial_metrics.csv[/bold green]")
    console.print("[bold green]✔ Saved 4 Neural Network checkpoints to checkpoints_pe/[/bold green]")

if __name__ == "__main__":
    main()
