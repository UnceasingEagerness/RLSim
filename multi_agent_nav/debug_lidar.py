import jax
import jax.numpy as jnp
from env.jax_lidar import jax_synthetic_lidar

print("1. Setup")
pos = jnp.array([0.0, 0.0])
yaw = 0.0
obstacles = jnp.zeros((200, 3))
lidar_range = 50.0
num_beams = 64

print("2. JIT compiling lidar...")
jitted_lidar = jax.jit(jax_synthetic_lidar, static_argnums=(3, 4))
out = jitted_lidar(pos, yaw, obstacles, lidar_range, num_beams)
jax.block_until_ready(out)

print("3. Done. No segfault!")
