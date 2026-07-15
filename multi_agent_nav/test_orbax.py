import orbax.checkpoint as ocp
import jax.numpy as jnp
import os

ckpt = ocp.StandardCheckpointer()
path = os.path.abspath("test_ckpt")
params = {"weight": jnp.ones(5)}
ckpt.save(path, item=params, force=True)
res = ckpt.restore(path, item=params)
print("Success:", res)
