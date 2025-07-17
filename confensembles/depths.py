import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

def tukey1d(x, y):
    f1 = jnp.mean(x < y)
    f2 = jnp.mean(x > y)
    f = jnp.vstack([f1, f2])
    return 2*jnp.min(f)
tukey1d = vmap(tukey1d, (None, 0))
tukey2d = vmap(vmap(tukey1d, (1, 1)), (1, 1))

def tukey_depth(x, y):
    depth = tukey2d(x, y)
    return jnp.mean(depth, axis = (0, 1))
tukey_depth = jit(tukey_depth)