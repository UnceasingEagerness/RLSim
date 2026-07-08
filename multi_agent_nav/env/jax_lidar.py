import jax
import jax.numpy as jnp

@jax.jit
def single_ray_intersect(ex: float, ey: float, dx: float, dy: float, obstacles: jnp.ndarray, max_range: float) -> float:
    """
    Computes the intersection of a single ray with a set of circular obstacles.
    obstacles: (N, 3) array where columns are [cx, cy, radius]
    """
    # Vectorized across obstacles
    fx = obstacles[:, 0] - ex
    fy = obstacles[:, 1] - ey
    r = obstacles[:, 2]
    
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    
    # Only valid if disc >= 0 and roots are > 0.01 (to avoid self-intersection logic if any)
    valid_mask = disc >= 0
    
    sqrt_disc = jnp.where(valid_mask, jnp.sqrt(jnp.maximum(0.0, disc)), 0.0)
    t = (-b - sqrt_disc) / 2.0
    
    # Mask out invalid distances (behind ray or too far)
    valid_t_mask = valid_mask & (t > 0.01) & (t < max_range)
    t_safe = jnp.where(valid_t_mask, t, max_range)
    
    # The closest intersection across all obstacles
    return jnp.min(t_safe)

import functools
@functools.partial(jax.jit, static_argnames=['num_beams'])
def jax_synthetic_lidar(ego_pos: jnp.ndarray, ego_yaw: float, obstacles: jnp.ndarray, max_range: float = 50.0, num_beams: int = 64) -> jnp.ndarray:
    """
    Ray-cast against circular obstacles using JAX vmap.
    """
    ex, ey = ego_pos[0], ego_pos[1]
    
    beam_angles = jnp.arange(num_beams) * (2.0 * jnp.pi / num_beams)
    world_angles = ego_yaw + beam_angles
    
    dxs = jnp.cos(world_angles)
    dys = jnp.sin(world_angles)
    
    # vmap over the beams to compute distances for all rays simultaneously
    vmap_intersect = jax.vmap(single_ray_intersect, in_axes=(None, None, 0, 0, None, None))
    distances = vmap_intersect(ex, ey, dxs, dys, obstacles, max_range)
    
    return distances
