from functools import partial
import os
from typing import Any, Tuple, List, Dict, Optional, Callable
import warnings

import arviz as az
import numpy as np
import jax.numpy as jnp
import jax
from jax.random import PRNGKey, split
from liesel.distributions.mvn_degen import MultivariateNormalDegenerate
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.math import fill_triangular
import ot

from scripts.vigamlss.utils.transformations import TransformationFunctions


def save_svi_vi_parameters(
    results: dict, save_dir_loc: str, save_dir_chol: str, file_name_prefix: str
) -> None:
    variational_loc = results["loc_vi_parameters_vec"]
    chol_vec = results["chol_vi_vec"]
    variational_lower_triangle = fill_triangular(chol_vec)

    os.makedirs(save_dir_loc, exist_ok=True)
    os.makedirs(save_dir_chol, exist_ok=True)

    base, _ = os.path.splitext(file_name_prefix)
    file_name_np_vi_loc = base + "_loc" + ".npy"
    file_name_np_vi_chol = base + "_chol" + ".npy"
    save_path_vi_loc = os.path.join(save_dir_loc, file_name_np_vi_loc)
    save_path_vi_chol = os.path.join(save_dir_chol, file_name_np_vi_chol)

    np.save(save_path_vi_loc, np.array(variational_loc))
    np.save(save_path_vi_chol, np.array(variational_lower_triangle))


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
    paths_loc = [os.path.join(save_dir_loc, file_loc) for file_loc in files_loc]
    paths_chol = [os.path.join(save_dir_chol, file_chol) for file_chol in files_chol]
    vi_loc_nps = [np.load(path_loc) for path_loc in paths_loc]
    vi_chol_nps = [np.load(path_chol) for path_chol in paths_chol]
    vi_loc_jnp = [jnp.array(vi_loc_np) for vi_loc_np in vi_loc_nps]
    vi_chol_jnp = [jnp.array(vi_chol_np) for vi_chol_np in vi_chol_nps]
    return vi_loc_jnp, vi_chol_jnp


def sim_study_loading_loop(
    save_dir: str,
    do_run: bool = False,
) -> Dict:
    """Traverses the subdirectories until the loc and chol folders with the numpy arrays are found.
    Creates a nested dict.
    """
    if do_run:
        result_dict = {}
        save_dir_abs = os.path.abspath(save_dir)
        for current_dir_abs, dirs, _ in os.walk(save_dir_abs):
            if "loc" in dirs and "chol" in dirs:
                loc_path = os.path.join(current_dir_abs, "loc")
                chol_path = os.path.join(current_dir_abs, "chol")

                # Check if 'loc' and 'chol' directories contain only files
                loc_has_subdirs = any(
                    os.path.isdir(os.path.join(loc_path, name))
                    for name in os.listdir(loc_path)
                )
                chol_has_subdirs = any(
                    os.path.isdir(os.path.join(chol_path, name))
                    for name in os.listdir(chol_path)
                )

                if not loc_has_subdirs and not chol_has_subdirs:
                    # Compute the relative path components
                    relative_path = os.path.relpath(current_dir_abs, save_dir_abs)
                    if relative_path == ".":
                        components = [os.path.basename(current_dir_abs)]
                    else:
                        components = relative_path.split(os.path.sep)

                    # Load the parameters
                    vi_locs, vi_chols = load_vi_loc_and_chol_parameters(
                        loc_path, chol_path
                    )

                    # Build the nested dictionary
                    current_dict = result_dict
                    for component in components:
                        if component not in current_dict:
                            current_dict[component] = {}
                        current_dict = current_dict[component]
                    current_dict["loc"] = vi_locs
                    current_dict["chol"] = vi_chols
        return result_dict
    else:
        warnings.warn(
            "The VI params have not been loaded. Flag do_run=True for sim_study_loading_loop.",
            category=UserWarning,
        )
        return {}


def build_vi_posteriors(
    vi_loc: List[jnp.ndarray], vi_chol: List[jnp.ndarray]
) -> tfd.MultivariateNormalTriL:
    """Note:
    Although one tfd.MultivariateNormalTril, this one contains batched MultivariateNormalTrils
    """
    mvn_tril = tfd.MultivariateNormalTriL(vi_loc, vi_chol)
    return mvn_tril


def gather_posterior_samples_nested(
    nested_svi_results: Dict[str, Any],
    n_samples: tuple = (10000,),
    device: str = "cpu",
    do_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Recursively traverse a nested dictionary of VI parameters and gather samples.

    Each leaf node in that dictionary is a dict with keys 'loc' and 'chol'.
    This function replaces those leaves with a dict containing 'samples' drawn
    from the corresponding MultivariateNormalTriL.
    """
    if not do_run:
        warnings.warn(
            "The VI samples have not been gathered. "
            "Flag do_run=True for gathering samples.",
            category=UserWarning,
        )
        return None

    def _recurse_tree(
        current_dict: Dict[str, Any], rng_key
    ) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
        """Recursively traverse `current_dict`. If a node has 'loc' and 'chol',
        sample from them; otherwise, recurse deeper.

        Returns a tuple of (new_subtree, updated_rng_key).
        """
        new_subtree = {}
        for k, v in current_dict.items():
            if isinstance(v, dict) and ("loc" in v) and ("chol" in v):
                rng_key, subkey = jax.random.split(rng_key)
                locs = v["loc"]  # List[jnp.ndarray]
                chols = v["chol"]  # List[jnp.ndarray]

                # Build batched distribution
                mvtrils = build_vi_posteriors(
                    locs, chols
                )  # Batch shape [B], event shape [D]

                # Sample: shape = (n_samples,) + batch_shape + event_shape
                with jax.default_device(jax.devices(device)[0]):
                    samples = mvtrils.sample(n_samples, seed=subkey)

                # Transpose to (batch_size, n_samples, event_dim)
                samples = jnp.swapaxes(samples, 0, 1)

                new_subtree[k] = {"samples": samples}

            elif isinstance(v, dict):
                # Not a leaf yet; recurse deeper.
                deeper_subtree, rng_key = _recurse_tree(v, rng_key)
                new_subtree[k] = deeper_subtree
            else:
                new_subtree[k] = v  # Copy unexpected values

        return new_subtree, rng_key

    # Start recursion with fresh PRNG key
    final_dict, _ = _recurse_tree(nested_svi_results, jax.random.PRNGKey(0))
    return final_dict


def wasserstein_distance(samples_1: jnp.ndarray, samples_2: jnp.ndarray) -> jnp.ndarray:
    n_posterior_samples = len(samples_1)

    weights = jnp.ones(n_posterior_samples) / n_posterior_samples

    M = jnp.array(ot.dist(samples_1, samples_2))

    return jnp.sqrt(ot.emd2(weights, weights, M))


def compute_wasserstein_nested(
    nested_samples_dict: Dict[str, Any],
    reference_samples: jnp.ndarray,
    device: str = "cpu",
    do_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Recursively traverse a nested dictionary where each leaf has 'samples'.
    For each leaf, compute the Wasserstein distance to `reference_samples` for each
    element in the leading batch dimension, replacing the 'samples' leaf with
    {'wassersteindistance': jnp.ndarray} containing the distances.
    """
    if not do_run:
        warnings.warn(
            "Wasserstein distances have not been computed. "
            "Flag do_run=True to perform the computation.",
            category=UserWarning,
        )
        return None

    def _recurse_tree(current_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively build a new dict with 'wassersteindistance' leaves."""
        new_subtree = {}
        for k, v in current_dict.items():
            # If this is a dict that has 'samples', treat it as a leaf node
            if isinstance(v, dict) and ("samples" in v):
                samples = v["samples"]

                # Compute Wasserstein distance for each batch element
                batch_size = samples.shape[0]
                distances = []
                with jax.default_device(jax.devices(device)[0]):
                    for i in range(batch_size):
                        dist = wasserstein_distance(samples[i], reference_samples)
                        distances.append(dist)
                    distances_array = jnp.stack(distances)

                new_subtree[k] = {"wassersteindistance": distances_array}
            elif isinstance(v, dict):
                # Recurse deeper into nested dicts
                new_subtree[k] = _recurse_tree(v)
            else:
                # Copy other values directly
                new_subtree[k] = v
        return new_subtree

    # Generate the new structure with Wasserstein distances
    return _recurse_tree(nested_samples_dict)


def compute_support_checks(carry, loc_vi, split_indices, transformations, rv_names):
    """JIT-compiled version of support checks"""
    matrix, Y_SYN = carry
    grouped_params = jnp.split(loc_vi, split_indices)
    transformed_params = [
        trans(param) for trans, param in zip(transformations, grouped_params)
    ]
    params_dict = dict(zip(rv_names, transformed_params))

    # Extract parameters
    beta0_loc = params_dict["beta0_loc"]
    gamma_loc = params_dict["gammas_loc"]
    beta0_scale = params_dict["beta0_scale"]
    gamma_scale = params_dict["gammas_scale"]
    beta0_shape = params_dict["beta0_shape"]
    gamma_shape = params_dict["gammas_shape"]

    # Compute linear predictors
    linear_loc = beta0_loc + matrix @ gamma_loc
    linear_scale = beta0_scale + matrix @ gamma_scale
    linear_shape = beta0_shape + matrix @ gamma_shape

    # Apply link functions
    scale = TransformationFunctions.softplus(linear_scale)
    loc = linear_loc
    shape = linear_shape

    # Support checks
    support_lower = Y_SYN >= loc
    upper_bound = jnp.where(shape < 0, loc - scale / shape, jnp.inf)
    support_upper = Y_SYN <= upper_bound
    support_checks = jnp.logical_and(support_lower, support_upper)

    return carry, jnp.mean(support_checks)


def save_support_checks(
    support_means: jnp.ndarray,  # Array of means for all epochs
    save_dir: str,
    file_name_prefix: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{file_name_prefix}.npy"
    save_path = os.path.join(save_dir, file_name)
    np.save(save_path, np.array(support_means))


def supportcheck_loading_loop(
    save_dir: str,
    do_run: bool = False,
) -> Dict:
    if do_run:
        result_dict = {}
        save_dir_abs = os.path.abspath(save_dir)
        for current_dir_abs, dirs, files in os.walk(save_dir_abs):
            if "support_checks" in dirs:
                support_checks_path = os.path.join(current_dir_abs, "support_checks")
                support_files = [
                    f for f in os.listdir(support_checks_path) if f == "all_epochs.npy"
                ]

                if support_files:
                    relative_path = os.path.relpath(current_dir_abs, save_dir_abs)
                    components = (
                        relative_path.split(os.sep) if relative_path != "." else []
                    )
                    current_dict = result_dict
                    for component in components:
                        current_dict = current_dict.setdefault(component, {})

                    # Load full array and create epoch mapping
                    full_array = np.load(
                        os.path.join(support_checks_path, "all_epochs.npy")
                    )
                    support_data = {
                        f"epoch_{i+1}": val for i, val in enumerate(full_array)
                    }
                    current_dict["support_means"] = support_data
        return result_dict
    else:
        warnings.warn(
            "Support checks not loaded. Set do_run=True to enable.", UserWarning
        )
        return {}


def sample_from_vi_posterior(
    final_vi_loc: jnp.ndarray,
    final_vi_chol: jnp.ndarray,
    transformations: List[Callable],
    split_indices: tuple,
    rv_names: List[str],
    num_samples: int,
    device: str = "gpu",
) -> dict:
    """Returns:
    dict: Dictionary mapping random variable names to transformed posterior samples.
    """
    variational_lower_triangle = fill_triangular(final_vi_chol)
    mvn_tril = tfd.MultivariateNormalTriL(final_vi_loc, variational_lower_triangle)
    with jax.default_device(jax.devices(device)[0]):
        loc_vi_parameters_samples = mvn_tril.sample((num_samples,), PRNGKey(0))
    grouped_loc_vi_parameters_samples = tuple(
        jnp.split(loc_vi_parameters_samples, split_indices, axis=1)
    )
    num_loc_vi_groups = len(grouped_loc_vi_parameters_samples)
    transformed_loc_vi_parameters_samples = jax.tree.map(
        lambda trans, x: trans(x),
        transformations[:num_loc_vi_groups],
        grouped_loc_vi_parameters_samples,
    )
    dict_loc_vi_parameters_samples = dict(
        zip(rv_names[:num_loc_vi_groups], transformed_loc_vi_parameters_samples)
    )
    return dict_loc_vi_parameters_samples


def compute_gpd_loc_scale_shape_posterior_samples(
    β0_loc_samples: jnp.ndarray,
    γ_loc_samples: jnp.ndarray,
    β0_scale_samples: jnp.ndarray,
    γ_scale_samples: jnp.ndarray,
    β0_shape_samples: jnp.ndarray,
    γ_shape_samples: jnp.ndarray,
    dm: jnp.ndarray,
    device: str = "gpu",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns posterior samples for location, scale, and shape parameters of GPD."""
    device = jax.devices(device)[0]
    def body_func(
        β0_loc,
        γ_loc,
        β0_scale,
        γ_scale,
        β0_shape,
        γ_shape,
        dm,
    ):
        posterior_loc = β0_loc + γ_loc @ dm.T
        posterior_scale = jax.nn.softplus(β0_scale + γ_scale @ dm.T)
        posterior_shape = β0_shape + γ_shape @ dm.T
        return posterior_loc, posterior_scale, posterior_shape

    jitted_body = jax.jit(body_func)
    with jax.default_device(device):
        posterior_loc_samples, posterior_scale_samples, posterior_shape_samples = (
            jitted_body(
                β0_loc_samples,
                γ_loc_samples,
                β0_scale_samples,
                γ_scale_samples,
                β0_shape_samples,
                γ_shape_samples,
                dm,
            )
        )

    return posterior_loc_samples, posterior_scale_samples, posterior_shape_samples


def compute_cgpd_loc_scale_shape_posterior_samples(
    β0_scale_samples: jnp.ndarray,
    γ_scale_samples: jnp.ndarray,
    β0_shape_samples: jnp.ndarray,
    γ_shape_samples: jnp.ndarray,
    dm: jnp.ndarray,
    device: str = "gpu",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns posterior samples for location, scale, and shape parameters of GPD."""
    device = jax.devices(device)[0]
    def body_func(
        β0_scale,
        γ_scale,
        β0_shape,
        γ_shape,
        dm,
    ):
        posterior_scale = jax.nn.softplus(β0_scale + γ_scale @ dm.T)
        posterior_shape = β0_shape + γ_shape @ dm.T
        return posterior_scale, posterior_shape

    jitted_body = jax.jit(body_func)
    with jax.default_device(device):
        posterior_scale_samples, posterior_shape_samples = (
            jitted_body(
                β0_scale_samples,
                γ_scale_samples,
                β0_shape_samples,
                γ_shape_samples,
                dm,
            )
        )

    return posterior_scale_samples, posterior_shape_samples


def compute_gpd_post_predictive_stats(
    loc_posterior_samples: jnp.ndarray,
    scale_posterior_samples: jnp.ndarray,
    shape_posterior_samples: jnp.ndarray,
    post_pred_samples: int,
    key: jnp.ndarray,
    device: str = "gpu",
) -> dict:
    """Returns:
    dict: Dictionary containing:
        - "mean": jnp.ndarray of predictive means
        - "0.25_quantile": jnp.ndarray of 25th quantiles
        - "0.75_quantile": jnp.ndarray of 75th quantiles
        - "hdi": jnp.ndarray of highest density intervals
    """
    device = jax.devices(device)[0]
    def body_fn(carry, xs):
        current_key = carry
        loc, scale, shape = xs
        new_key, sample_key = jax.random.split(current_key)

        # Generate samples for current posterior parameters
        samples = tfd.GeneralizedPareto(loc, scale, shape).sample(
            sample_shape=(post_pred_samples,), seed=sample_key
        )
        return new_key, samples

    # Scan over posterior samples with key in carry
    with jax.default_device(device):
        _, samples = jax.lax.scan(
            jax.jit(body_fn),
            key,
            (loc_posterior_samples, scale_posterior_samples, shape_posterior_samples),
        )

    # Reshape to (n_post * n_samples, len_X)
    flat_samples = samples.reshape(-1, samples.shape[-1])

    return {
        "mean": jnp.mean(flat_samples, axis=0),
        "0.0_quantile": jnp.quantile(flat_samples, 0.0, axis=0),
        "0.25_quantile": jnp.quantile(flat_samples, 0.25, axis=0),
        "0.75_quantile": jnp.quantile(flat_samples, 0.75, axis=0),
        "hdi": az.hdi(np.array(flat_samples)),
    }


def compute_cgpd_post_predictive_stats(
    posterior_thresholds_map: jnp.ndarray,
    scale_posterior_samples: jnp.ndarray,
    shape_posterior_samples: jnp.ndarray,
    post_pred_samples: int,
    key: jnp.ndarray,
    device: str = "gpu",
) -> dict:
    """Returns:
    dict: Dictionary containing:
        - "mean": jnp.ndarray of predictive means
        - "0.25_quantile": jnp.ndarray of 25th quantiles
        - "0.75_quantile": jnp.ndarray of 75th quantiles
        - "hdi": jnp.ndarray of highest density intervals
    """
    device = jax.devices(device)[0]
    loc_posterior_samples = jnp.tile(posterior_thresholds_map, (len(scale_posterior_samples), 1))
    def body_fn(carry, xs):
        current_key = carry
        loc, scale, shape = xs
        new_key, sample_key = jax.random.split(current_key)

        # Generate samples for current posterior parameters
        samples = tfd.GeneralizedPareto(loc, scale, shape).sample(
            sample_shape=(post_pred_samples,), seed=sample_key
        )
        return new_key, samples

    # Scan over posterior samples with key in carry
    _, samples = jax.lax.scan(
        jax.jit(body_fn, device=device),
        key,
        (loc_posterior_samples, scale_posterior_samples, shape_posterior_samples),
    )

    # Reshape to (n_post * n_samples, len_X)
    flat_samples = samples.reshape(-1, samples.shape[-1])

    return {
        "mean": jnp.mean(flat_samples, axis=0),
        "0.0_quantile": jnp.quantile(flat_samples, 0.0, axis=0),
        "0.25_quantile": jnp.quantile(flat_samples, 0.25, axis=0),
        "0.75_quantile": jnp.quantile(flat_samples, 0.75, axis=0),
        "hdi": az.hdi(np.array(flat_samples)),
    }


def compute_prior_predictive_stats_sim_study(
    dm: jnp.ndarray,
    K: jnp.ndarray,
    n_prior_samples: int,
    key: jnp.ndarray,
) -> dict:
    """Computes prior predictive statistics for the GPD model.

    Args:
        dm: The design matrix used in the model.
        K: The penalty matrix for the degenerate normal distributions.
        n_prior_samples: Number of prior samples to generate.
        key: PRNG key for reproducibility.

    Returns:
        dict: Dictionary containing prior predictive statistics:
            - "mean": jnp.ndarray of predictive means
            - "0.0_quantile": jnp.ndarray of minimum values
            - "0.25_quantile": jnp.ndarray of 25th quantiles
            - "0.75_quantile": jnp.ndarray of 75th quantiles
            - "1.0_quantile": jnp.ndarray of maximum values
            - "hdi": jnp.ndarray of highest density intervals
    """

    def body_fn(key, dm, K, n_prior_samples):
        current_key = key
        new_key, sample_key = jax.random.split(current_key)
        subkeys = jax.random.split(sample_key, 10)
        keys = {
            "beta0_loc": subkeys[0],
            "lambda_loc": subkeys[1],
            "gamma_loc": subkeys[2],
            "beta0_scale": subkeys[3],
            "lambda_scale": subkeys[4],
            "gamma_scale": subkeys[5],
            "beta0_shape": subkeys[6],
            "lambda_shape": subkeys[7],
            "gamma_shape": subkeys[8],
            "y": subkeys[9],
        }

        # Sample intercepts
        beta0_loc = tfd.Normal(0.0, 50.0).sample(
            (n_prior_samples,), seed=keys["beta0_loc"]
        )
        beta0_scale = tfd.Normal(0.0, 50.0).sample(
            (n_prior_samples,), seed=keys["beta0_scale"]
        )
        beta0_shape = tfd.Normal(0.0, 50.0).sample(
            (n_prior_samples,), seed=keys["beta0_shape"]
        )

        # Sample smoothing parameters
        lambda_loc = tfd.HalfCauchy(0.0, 0.01).sample(
            (n_prior_samples,), seed=keys["lambda_loc"]
        )
        lambda_scale = tfd.HalfCauchy(0.0, 0.01).sample(
            (n_prior_samples,), seed=keys["lambda_scale"]
        )
        lambda_shape = tfd.HalfCauchy(0.0, 0.01).sample(
            (n_prior_samples,), seed=keys["lambda_shape"]
        )

        # Sample coefficients using degenerate normal
        gamma_loc = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=1.0 / lambda_loc, pen=K
        ).sample(seed=keys["gamma_loc"])

        gamma_scale = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=1.0 / lambda_scale, pen=K
        ).sample(seed=keys["gamma_scale"])

        gamma_shape = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=1.0 / lambda_shape, pen=K
        ).sample(seed=keys["gamma_shape"])

        loc = jnp.expand_dims(beta0_loc, axis=1) + gamma_loc @ dm.T
        scale = TransformationFunctions.softplus(
            jnp.expand_dims(beta0_scale, axis=1) + gamma_scale @ dm.T
        )
        shape = jnp.expand_dims(beta0_shape, axis=1) + gamma_shape @ dm.T

        # Sample from GPD
        y_sample = tfd.GeneralizedPareto(
            loc=loc, scale=scale, concentration=shape
        ).sample(seed=keys["y"])

        return new_key, (y_sample, loc, scale, shape)

    # Use scan to accumulate samples
    _, prior_predictives = jax.jit(partial(body_fn, n_prior_samples=n_prior_samples))(
        key, dm, K
    )
    y_samples, loc, scale, shape = prior_predictives

    return {
        "response": list(y_samples),
        "loc": list(loc),
        "scale": list(scale),
        "shape": list(shape),
    }


def compute_prior_predictive_stats_case_study(
    dm: jnp.ndarray,
    K: jnp.ndarray,
    n_prior_samples: int,
    key: jnp.ndarray,
) -> dict:
    """Computes prior predictive statistics for the GPD model.

    Args:
        dm: The design matrix used in the model.
        K: The penalty matrix for the degenerate normal distributions.
        n_prior_samples: Number of prior samples to generate.
        key: PRNG key for reproducibility.

    Returns:
        dict: Dictionary containing prior predictive statistics:
            - "mean": jnp.ndarray of predictive means
            - "0.0_quantile": jnp.ndarray of minimum values
            - "0.25_quantile": jnp.ndarray of 25th quantiles
            - "0.75_quantile": jnp.ndarray of 75th quantiles
            - "1.0_quantile": jnp.ndarray of maximum values
            - "hdi": jnp.ndarray of highest density intervals
    """

    def body_fn(key, dm, K, n_prior_samples):
        current_key = key
        new_key, sample_key = jax.random.split(current_key)
        subkeys = jax.random.split(sample_key, 10)
        keys = {
            "beta0_loc": subkeys[0],
            "lambda_loc": subkeys[1],
            "gamma_loc": subkeys[2],
            "beta0_scale": subkeys[3],
            "lambda_scale": subkeys[4],
            "gamma_scale": subkeys[5],
            "beta0_shape": subkeys[6],
            "lambda_shape": subkeys[7],
            "gamma_shape": subkeys[8],
            "y": subkeys[9],
        }

        # Sample intercepts
        beta0_loc = tfd.Normal(0.0, 50.0).sample(
            (n_prior_samples,), seed=keys["beta0_loc"]
        )
        beta0_scale = tfd.Normal(0.0, 50.0).sample(
            (n_prior_samples,), seed=keys["beta0_scale"]
        )
        beta0_shape = tfd.Normal(0.0, 50.0).sample(
            (n_prior_samples,), seed=keys["beta0_shape"]
        )

        # Sample smoothing parameters
        lambda_loc = tfd.HalfCauchy(0.0, 0.01).sample(
            (n_prior_samples,), seed=keys["lambda_loc"]
        )
        lambda_scale = tfd.HalfCauchy(0.0, 0.01).sample(
            (n_prior_samples,), seed=keys["lambda_scale"]
        )
        lambda_shape = tfd.HalfCauchy(0.0, 0.01).sample(
            (n_prior_samples,), seed=keys["lambda_shape"]
        )

        # Sample coefficients using degenerate normal
        gamma_loc = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=1.0 / lambda_loc, pen=K
        ).sample(seed=keys["gamma_loc"])

        gamma_scale = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=1.0 / lambda_scale, pen=K
        ).sample(seed=keys["gamma_scale"])

        gamma_shape = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=1.0 / lambda_shape, pen=K
        ).sample(seed=keys["gamma_shape"])

        loc = jnp.expand_dims(beta0_loc, axis=1) + gamma_loc @ dm.T
        scale = TransformationFunctions.softplus(
            jnp.expand_dims(beta0_scale, axis=1) + gamma_scale @ dm.T
        )
        shape = jnp.expand_dims(beta0_shape, axis=1) + gamma_shape @ dm.T

        # Sample from GPD
        y_sample = tfd.GeneralizedPareto(
            loc=loc, scale=scale, concentration=shape
        ).sample(seed=keys["y"])

        return new_key, (y_sample, loc, scale, shape)

    # Use scan to accumulate samples
    _, prior_predictives = jax.jit(partial(body_fn, n_prior_samples=n_prior_samples))(
        key, dm, K
    )
    y_samples, loc, scale, shape = prior_predictives

    return {
        "response": list(y_samples),
        "loc": list(loc),
        "scale": list(scale),
        "shape": list(shape),
    }
