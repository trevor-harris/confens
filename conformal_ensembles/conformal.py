import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

from .depths import *

# Estimates the cutoff value conformal inference on spatiotemporal processes. Assumes tukey depth.
def conf_quantile(res_val, alpha):
    '''
    Parameters
    ----------

    res_val: 3D tensor (n, p1, p2)
        n = sample size
        p1 = spatial dim 1
        p2 = spatial dim 2

        Residual fields from the calibration / validation set, i.e. y - y_hat


    alpha: float
        confidence level between 0 and 1

    Returns
    -------
    Cutoff value for depths to generate an alpha level prediction set.
    '''

    nval = res_val.shape[0]
    adj_alpha = jnp.ceil((1 - alpha) * (nval + 1))/(nval + 1)

    # compute the calibration depths
    depth_val = tukey_depth(res_val, res_val)

    # smoothed quantile estimator
    q_val = jnp.sort(depth_val)[nval-int(jnp.ceil((1 - alpha) * (nval + 1)))]

    return q_val

# Generates the conformal ensemble. Assumes tukey depth.
def conf_ensemble(res_val, alpha):
    '''
    Parameters
    ----------

    res_val: 3D tensor (n, p1, p2)
        n = sample size
        p1 = spatial dim 1
        p2 = spatial dim 2

        Residual fields from the calibration / validation set, i.e. y - y_hat


    alpha: float
        confidence level between 0 and 1

    Returns
    -------
    Full alpha level conformal ensemble. Add these fields onto predictions to generate prediction sets.
    '''
    
    nval = res_val.shape[0]
    adj_alpha = jnp.ceil((1 - alpha) * (nval + 1))/(nval + 1)

    # compute the calibration depths
    depth_val = tukey_depth(res_val, res_val)
    
    # smoothed quantile estimator
    q_val = jnp.sort(depth_val)[nval-int(jnp.ceil((1 - alpha) * (nval + 1)))]

    return res_val[depth_val >= q_val]