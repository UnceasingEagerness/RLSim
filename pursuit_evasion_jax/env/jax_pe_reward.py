import jax
import jax.numpy as jnp

def ssa(angle):
    """Smallest signed angle between -pi and pi."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jax.jit
def compute_pursuer_rewards(pos: jnp.ndarray, e_pos: jnp.ndarray, collision_mask: jnp.ndarray, num_pursuers: int = 3, capture_radius: float = 5.0) -> jnp.ndarray:
    """
    Computes hierarchical reward for pursuers:
    pos: [3, 2] pursuer positions
    e_pos: [2] evader position
    collision_mask: [3] boolean array indicating collisions
    """
    # Hyperparameters from the paper
    delta_g = 500.0
    delta_col = -200.0
    k_d1, k_d2 = 0.5, 2.0
    k_a1, k_a2 = 5.0, 5.0
    
    # Distances
    d_p_to_e = jnp.linalg.norm(pos - e_pos, axis=1) # [3]
    d_mean = jnp.mean(d_p_to_e)
    d_std = jnp.std(d_p_to_e) + 1e-6
    
    # 1. Collision Reward
    r_collision = jnp.where(collision_mask, delta_col, 0.0)
    
    # 2. Distance Reward
    norm_dev = jnp.clip((d_p_to_e - d_mean) / d_std, -5.0, 5.0)
    r_distance = - (k_d1 * d_p_to_e + k_d2 * jnp.exp(norm_dev) - 1.0)
    
    # 3. Angle Reward
    # Calculate capture angles between adjacent pursuers
    def calc_angles(i):
        v_ego = pos[i] - e_pos
        v_nbor1 = pos[(i+1)%3] - e_pos
        v_nbor2 = pos[(i+2)%3] - e_pos
        
        angle_ego = jnp.arctan2(v_ego[1], v_ego[0])
        angle_1 = jnp.arctan2(v_nbor1[1], v_nbor1[0])
        angle_2 = jnp.arctan2(v_nbor2[1], v_nbor2[0])
        
        gamma_ab = jnp.abs(ssa(angle_ego - angle_1))
        gamma_ac = jnp.abs(ssa(angle_ego - angle_2))
        
        return gamma_ab, gamma_ac
        
    gammas = jax.vmap(calc_angles)(jnp.arange(3))
    gamma_ab = gammas[0] # [3]
    gamma_ac = gammas[1] # [3]
    
    ideal_angle = 2.0 * jnp.pi / num_pursuers
    r_a1 = jnp.exp(-jnp.abs(gamma_ab - ideal_angle)) + jnp.exp(-jnp.abs(gamma_ac - ideal_angle)) - 2.0
    
    delta_gamma_a = jnp.abs(gamma_ab - gamma_ac)
    r_a2 = jnp.exp(-delta_gamma_a) - 1.0
    
    r_angle = k_a1 * r_a1 + k_a2 * r_a2
    
    # 4. Target Reward (Goal)
    is_captured = jnp.all(d_p_to_e < capture_radius)
    is_encircled = jnp.all(gamma_ab <= jnp.pi) & jnp.all(gamma_ac <= jnp.pi)
    success = is_captured & is_encircled
    
    r_goal = jnp.where(success, delta_g, 0.0)
    
    # Total Pursuer Reward
    r_total = r_goal + r_collision + r_distance + r_angle
    
    return r_total

@jax.jit
def compute_rewards(pos: jnp.ndarray, e_pos: jnp.ndarray, p_collision: jnp.ndarray, e_collision: bool) -> jnp.ndarray:
    """
    Computes global zero-sum rewards for all 4 agents.
    Returns: [4] array of rewards (3 pursuers, 1 evader)
    """
    r_p = compute_pursuer_rewards(pos, e_pos, p_collision)
    
    # Zero-sum game: Evader reward is the negative sum of pursuer rewards
    r_e = -jnp.sum(r_p)
    
    # Add explicit penalty if evader collides (to prevent suicidal evasion)
    r_e = jnp.where(e_collision, -200.0, r_e)
    
    return jnp.concatenate([r_p, jnp.array([r_e])])
