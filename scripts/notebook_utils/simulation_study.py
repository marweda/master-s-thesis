import os
from typing import Any, Tuple, List

import numpy as np
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.math import fill_triangular
import ot


def wasserstein_distance(samples_1: jnp.ndarray, samples_2: jnp.ndarray) -> jnp.ndarray:
    n_posterior_samples = len(samples_1)

    weights = jnp.ones(n_posterior_samples) / n_posterior_samples

    M = jnp.array(ot.dist(samples_1, samples_2))

    return jnp.sqrt(ot.emd2(weights, weights, M))


def save_svi_vi_parameters(
    results: dict, save_dir_loc: str, save_dir_chol: str, file_name_prefix: str
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    variational_loc = results["loc_vi_parameters_vec"]
    chol_vec = results["chol_vi_vec"]
    variational_lower_triangle = fill_triangular(chol_vec)

    # Check that all three parameters are provided.
    if not (file_name_prefix and save_dir_loc and save_dir_chol):
        raise ValueError(
            "file_name_prefix, save_dir_loc, and save_dir_chol must all be provided to save the SVI VI parameters."
        )

    os.makedirs(save_dir_loc, exist_ok=True)
    os.makedirs(save_dir_chol, exist_ok=True)

    base, _ = os.path.splitext(file_name_prefix)
    file_name_np_vi_loc = base + "_loc" + ".npy"
    file_name_np_vi_chol = base + "_chol" + ".npy"
    save_path_vi_loc = os.path.join(save_dir_loc, file_name_np_vi_loc)
    save_path_vi_chol = os.path.join(save_dir_chol, file_name_np_vi_chol)

    np.save(save_path_vi_loc, np.array(variational_loc))
    np.save(save_path_vi_chol, np.array(variational_lower_triangle))
    return variational_loc, variational_lower_triangle


def load_vi_loc_and_chol_parameters(
    save_dir_loc: str, save_dir_chol: str
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    files_loc = [
        f
        for f in os.listdir(save_dir_loc)
        if os.path.isfile(os.path.join(save_dir_loc, f))
    ]
    files_chol = [
        f
        for f in os.listdir(save_dir_chol)
        if os.path.isfile(os.path.join(save_dir_chol, f))
    ]
    vi_loc_nps = [np.load(file_loc) for file_loc in files_loc]
    vi_chol_nps = [np.load(file_chol for file_chol in files_chol)]
    vi_loc_jnp = [jnp.array(vi_loc_np) for vi_loc_np in vi_loc_nps]
    vi_chol_jnp = [jnp.array(vi_chol_np) for vi_chol_np in vi_chol_nps]
    return vi_loc_jnp, vi_chol_jnp


def build_vi_posteriors(
    vi_loc: List[jnp.ndarray], vi_chol: List[jnp.ndarray]
) -> tfd.MultivariateNormalTriL:
    """Note:
    Although one tfd.MultivariateNormalTril, this one contains batched MultivariateNormalTrils
    """
    mvn_tril = tfd.MultivariateNormalTriL(vi_loc, vi_chol)
    return mvn_tril
