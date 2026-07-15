import jax
import jax.numpy as jnp

@jax.jit
def single_sonar_ray(ex: float, ey: float, dx: float, dy: float, obstacles: jnp.ndarray, max_range: float, map_size: float) -> float:
    """
    Computes intersection of a single sonar ray with circular obstacles and map boundaries.
    """
    # 1. Circle intersections
    fx = obstacles[:, 0] - ex
    fy = obstacles[:, 1] - ey
    r = obstacles[:, 2]
    
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    
    valid_mask = disc >= 0
    sqrt_disc = jnp.where(valid_mask, jnp.sqrt(jnp.maximum(0.0, disc)), 0.0)
    t = (-b - sqrt_disc) / 2.0
    
    valid_t_mask = valid_mask & (t > 0.01) & (t < max_range)
    t_obs = jnp.where(valid_t_mask, t, max_range)
    min_t_obs = jnp.min(t_obs)
    
    # 2. Boundary intersections (x=0, x=map_size, y=0, y=map_size)
    # t = (bound - pos) / d
    
    # X boundaries
    t_x0 = jnp.where(jnp.abs(dx) > 1e-5, (0.0 - ex) / dx, max_range)
    t_x1 = jnp.where(jnp.abs(dx) > 1e-5, (map_size - ex) / dx, max_range)
    
    valid_x0 = (t_x0 > 0.01) & (t_x0 < max_range) & (ey + t_x0 * dy >= 0) & (ey + t_x0 * dy <= map_size)
    valid_x1 = (t_x1 > 0.01) & (t_x1 < max_range) & (ey + t_x1 * dy >= 0) & (ey + t_x1 * dy <= map_size)
    
    min_t_bound_x = jnp.minimum(jnp.where(valid_x0, t_x0, max_range), jnp.where(valid_x1, t_x1, max_range))
    
    # Y boundaries
    t_y0 = jnp.where(jnp.abs(dy) > 1e-5, (0.0 - ey) / dy, max_range)
    t_y1 = jnp.where(jnp.abs(dy) > 1e-5, (map_size - ey) / dy, max_range)
    
    valid_y0 = (t_y0 > 0.01) & (t_y0 < max_range) & (ex + t_y0 * dx >= 0) & (ex + t_y0 * dx <= map_size)
    valid_y1 = (t_y1 > 0.01) & (t_y1 < max_range) & (ex + t_y1 * dx >= 0) & (ex + t_y1 * dx <= map_size)
    
    min_t_bound_y = jnp.minimum(jnp.where(valid_y0, t_y0, max_range), jnp.where(valid_y1, t_y1, max_range))
    
    # Final min
    min_t = jnp.minimum(min_t_obs, jnp.minimum(min_t_bound_x, min_t_bound_y))
    return min_t

import functools
@functools.partial(jax.jit, static_argnames=['num_beams'])
def jax_forward_sonar(ego_pos: jnp.ndarray, ego_yaw: float, obstacles: jnp.ndarray, max_range: float = 20.0, map_size: float = 100.0, num_beams: int = 11) -> jnp.ndarray:
    """
    Ray-cast forward 120 degree sector.
    """
    ex, ey = ego_pos[0], ego_pos[1]
    
    # -60 to +60 degrees
    beam_angles = jnp.linspace(-jnp.pi/3, jnp.pi/3, num_beams)
    world_angles = ego_yaw + beam_angles
    
    dxs = jnp.cos(world_angles)
    dys = jnp.sin(world_angles)
    
    vmap_intersect = jax.vmap(single_sonar_ray, in_axes=(None, None, 0, 0, None, None, None))
    distances = vmap_intersect(ex, ey, dxs, dys, obstacles, max_range, map_size)
    
    return distances
