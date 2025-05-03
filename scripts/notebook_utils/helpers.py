from functools import partial
import os
import pickle
from typing import Any, Tuple, List, Dict, Optional, Callable, Union
import warnings

import arviz as az
import numpy as np
import jax.numpy as jnp
import jax
from jax.random import PRNGKey, split
from liesel.distributions.mvn_degen import MultivariateNormalDegenerate
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.bijectors import Softplus, TransformDiagonal
from tensorflow_probability.substrates.jax.math import fill_triangular
from tqdm.notebook import tqdm
import ot
from ott.geometry import pointcloud
from ott.solvers import linear

from scripts.vigamlss.utils.transformations import TransformationFunctions
from scripts.vigamlss.utils.custom_tf_distributions import CustomGPD


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


def sim_study_svi_loading_loop(
    save_dir: str, do_run: bool = False
) -> Tuple[Dict[str, Any], List, List, List]:
    """Loads all run data from pickle files and returns nested dict + parameter lists."""
    if do_run:
        result_dict = {}
        rv_names_list = []
        transformations_list = []
        split_indices_list = []

        save_dir_abs = os.path.abspath(save_dir)
        for current_dir_abs, dirs, files in os.walk(save_dir_abs):
            # Process directories in sorted order
            dirs.sort()
            files.sort()

            # Check for .pkl files
            pkl_files = [f for f in files if f.endswith(".pkl")]
            if pkl_files:
                # Sort runs numerically (run_0.pkl, run_1.pkl, ...)
                pkl_files_sorted = sorted(
                    pkl_files, key=lambda x: int(x.split("_")[1].split(".")[0])
                )

                # Load all runs in this directory
                locs, chols = [], []
                for pkl_file in pkl_files_sorted:
                    file_path = os.path.join(current_dir_abs, pkl_file)
                    with open(file_path, "rb") as f:
                        run_data = pickle.load(f)
                        locs.append(run_data["loc"])
                        chols.append(run_data["chol"])
                        rv_names_list.append(run_data["rv_names"])
                        transformations_list.append(run_data["transformations"])
                        split_indices_list.append(run_data["split_indices"])

                # Build relative path components
                relative_path = os.path.relpath(current_dir_abs, save_dir_abs)
                components = (
                    relative_path.split(os.path.sep)
                    if relative_path != "."
                    else [os.path.basename(current_dir_abs)]
                )

                # Build nested dictionary
                current_dict = result_dict
                for component in components:
                    if component not in current_dict:
                        current_dict[component] = {}
                    current_dict = current_dict[component]
                current_dict["loc"] = locs
                current_dict["chol"] = chols

        return (result_dict, rv_names_list, transformations_list, split_indices_list)
    else:
        warnings.warn(
            "VI params not loaded. Flag do_run=True.",
            category=UserWarning,
        )
        return ({}, [], [], [])


def gather_posterior_samples_nested(
    nested_svi_results: Dict[str, Any],
    rv_names: List[List[str]],
    transformations: List[List[Callable]],
    split_indices: List[tuple],
    n_samples: tuple = (10000,),
    device: str = "cpu",
    do_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Recursively traverse a nested dictionary of VI parameters and gather samples.

    Each leaf node in that dictionary is a dict with keys 'loc' and 'chol'.
    This function replaces those leaves with a dict containing 'samples' which is
    a list of dictionaries mapping RV names to transformed samples.

    Args:
        nested_svi_results: Nested dictionary containing 'loc' and 'chol' at leaves.
        rv_names: List of lists, each sublist contains RV names for a run's groups.
        transformations: List of lists, each sublist contains callables for a run's groups.
        split_indices: List of tuples, each tuple defines splits for a run's parameters.
        n_samples: Number of samples to draw per run.
        device: Device to place samples on.
        do_run: If True, gather samples; else, warn and return None.

    Returns:
        Nested dictionary with 'samples' entries as lists of RV dictionaries.
    """
    if not do_run:
        warnings.warn(
            "The VI samples have not been gathered. "
            "Flag do_run=True for gathering samples.",
            category=UserWarning,
        )
        return None

    def build_vi_posteriors(locs, chols):
        # Assuming locs and chols are lists of parameters for batched distribution
        loc_batch = jnp.stack(locs)
        chol_batch = jnp.stack(
            [
                TransformDiagonal(diag_bijector=Softplus()).forward(fill_triangular(c))
                for c in chols
            ]
        )
        return tfd.MultivariateNormalTriL(loc_batch, chol_batch)

    def _recurse_tree(
        current_dict: Dict[str, Any], rng_key: jax.random.PRNGKey, run_idx: int
    ) -> Tuple[Dict[str, Any], jax.random.PRNGKey, int]:
        new_subtree = {}
        for k, v in current_dict.items():
            if isinstance(v, dict) and ("loc" in v) and ("chol" in v):
                # Process leaf node: generate samples for each run in the batch
                rng_key, subkey = jax.random.split(rng_key)
                locs = v["loc"]  # List[jnp.ndarray], each of shape (D,)
                chols = v["chol"]  # List[jnp.ndarray], each of shape (D*(D+1)/2,)
                B = len(locs)  # Number of runs in this leaf

                # Check if there are enough parameters for all runs
                if run_idx + B > len(rv_names):
                    raise ValueError("Insufficient parameters for all runs.")

                # Build batched distribution and sample
                mvtrils = build_vi_posteriors(locs, chols)
                samples = mvtrils.sample(n_samples, seed=subkey)  # (n_samples, B, D)
                samples = jnp.swapaxes(samples, 0, 1)  # (B, n_samples, D)
                # Move samples to CPU
                samples = jax.device_put(samples, jax.devices(device)[0])

                # Process each run in the batch
                run_samples_list = []
                for i in range(B):
                    current_run_samples = samples[i]  # (n_samples, D)
                    current_rvs = rv_names[run_idx + i]
                    current_trans = transformations[run_idx + i]
                    current_split = split_indices[run_idx + i]

                    # Split and transform
                    grouped = jnp.split(current_run_samples, current_split, axis=1)
                    transformed = [t(g) for t, g in zip(current_trans, grouped)]
                    run_dict = dict(zip(current_rvs, transformed))
                    run_samples_list.append(run_dict)

                new_subtree[k] = {"samples": run_samples_list}
                run_idx += B  # Update run index

            elif isinstance(v, dict):
                # Recurse deeper
                deeper_subtree, rng_key, run_idx = _recurse_tree(v, rng_key, run_idx)
                new_subtree[k] = deeper_subtree
            else:
                new_subtree[k] = v  # Preserve other values

        return new_subtree, rng_key, run_idx

    # Initialize recursion with PRNG key and run index 0
    final_dict, _, _ = _recurse_tree(nested_svi_results, jax.random.PRNGKey(0), 0)
    return final_dict


def wasserstein_distance_ot_1d(
    samples_1: jnp.ndarray, samples_2: jnp.ndarray, device: str
) -> jnp.ndarray:
    with jax.default_device(jax.devices(device)[0]):
        n_samples_1 = len(samples_1)
        n_samples_2 = len(samples_2)

        # Create uniform weights (optional, as these are the defaults)
        weights_1 = jnp.ones(n_samples_1) / n_samples_1
        weights_2 = jnp.ones(n_samples_2) / n_samples_2

        return jnp.sqrt(ot.emd2_1d(samples_1, samples_2, weights_1, weights_2))


@jax.jit
def wasserstein_distance_ott(samples_1, samples_2):
    n_samples = samples_1.shape[0]

    weights = jnp.ones(n_samples) / n_samples

    geom = pointcloud.PointCloud(
        samples_1, samples_2, epsilon=0.01  # Small epsilon for good EMD approximation
    )

    sinkhorn_solution = linear.solve(
        geom,
        a=weights,
        b=weights,
        # threshold=1e-6,        # Strict convergence threshold
        # max_iterations=100000  # High iteration limit
    )

    # Calculate 2-Wasserstein distance from primal cost
    return jnp.sqrt(sinkhorn_solution.primal_cost)


def thin_samples(
    samples: jnp.ndarray, thinning_degree: Optional[int] = None
) -> jnp.ndarray:
    """Apply thinning to samples by taking every thinning_degree-th sample."""
    if thinning_degree is None or thinning_degree <= 1:
        return samples

    # For 1D arrays (after squeezing)
    if samples.ndim == 1:
        return samples[::thinning_degree]

    # For 2D arrays (e.g., (32000, 5))
    elif samples.ndim == 2:
        return samples[::thinning_degree, :]

    # Handle unexpected dimensions
    else:
        raise ValueError(f"Unexpected sample shape: {samples.shape}")


def process_mcmc_param(param_data: jnp.ndarray, num_samples: int) -> jnp.ndarray:
    # Truncate samples dimension
    truncated = param_data[:, :num_samples]
    # Handle different parameter types
    if truncated.ndim == 2:  # Scalar parameters
        reshaped = truncated.reshape(-1, 1)
        return reshaped
    elif truncated.ndim == 3:  # Vector parameters
        reshaped = truncated.reshape(-1, truncated.shape[-1])
        return reshaped
    else:
        raise ValueError(f"Unexpected MCMC parameter shape: {param_data.shape}")


def process_svi_param(param_data: jnp.ndarray) -> jnp.ndarray:
    """Process SVI parameter to 2D format."""
    if param_data.ndim == 1:
        return param_data.reshape(-1, 1)
    return param_data


def samples_dict_to_array(
    samples_dict: Dict[str, jnp.ndarray], rv_order: List[str]
) -> jnp.ndarray:
    """Convert dictionary of samples to 2D array."""
    return jnp.concatenate([samples_dict[name] for name in rv_order], axis=1)


def compute_wasserstein_distances(
    svi_samples_dict: Dict[str, Any],
    mcmc_samples_dict: Dict[str, Any],
    reference_dict: Dict[str, Any],
    mcmc_samples_per_chain: int,
    device: str = "cpu",
    do_run: bool = False,
    save_dir: Optional[str] = None,
    file_name_prefix: Optional[str] = None,
    thinning_degree: Optional[int] = None,
    pot_1d_device: str = "cpu",
) -> Dict[str, Any]:
    if not do_run:
        warnings.warn("WD computation skipped - set do_run=True", UserWarning)
        return {}
    target_device = jax.devices(device)[0]

    # Define which parameters use which distance function
    use_ot_1d_keys = [
        "lambda_smooth_shape",
        "lambda_smooth_scale",
        "intercept_shape",
        "intercept_scale",
    ]

    epoch_keys = list(svi_samples_dict.keys())
    sample_size_keys = list(mcmc_samples_dict.keys())

    # Verification checks
    for size_key in sample_size_keys:
        if size_key not in reference_dict or "samples" not in reference_dict[size_key]:
            raise ValueError(f"Invalid reference structure for {size_key}")
        for epoch_key in epoch_keys:
            if size_key not in svi_samples_dict[epoch_key]:
                raise ValueError(f"Missing {size_key} in svi_samples_dict[{epoch_key}]")

    # Precompute MCMC results
    mcmc_results = {}
    with jax.default_device(jax.devices(device)[0]):
        for size_key in tqdm(sample_size_keys, desc="MCMC"):
            # Process reference samples (same as MCMC processing)
            ref_samples = {
                k: process_mcmc_param(v, mcmc_samples_per_chain)
                for k, v in reference_dict[size_key]["samples"].items()
            }
            rv_order = sorted(ref_samples.keys())

            ref_full = samples_dict_to_array(ref_samples, rv_order)

            mcmc_samples = mcmc_samples_dict[size_key].get("samples", [])
            if not mcmc_samples:
                continue

            mcmc_dists = {"marginal": {}, "full": []}

            # Process MCMC runs with truncation
            processed_mcmc = [
                {
                    k: process_mcmc_param(v, mcmc_samples_per_chain)
                    for k, v in run.items()
                }
                for run in mcmc_samples
            ]

            # Compute distances
            for rv in tqdm(rv_order, desc="MCMC params", leave=False):
                mcmc_params = [run[rv] for run in processed_mcmc]

                # Choose distance function based on parameter name
                if rv in use_ot_1d_keys:
                    mcmc_dists["marginal"][rv] = jnp.array(
                        [
                            wasserstein_distance_ot_1d(
                                thin_samples(
                                    jax.device_put(
                                        p.squeeze(), device=jax.devices("cpu")[0]
                                    ),
                                    thinning_degree,
                                ),
                                thin_samples(
                                    jax.device_put(
                                        ref_samples[rv].squeeze(),
                                        device=jax.devices("cpu")[0],
                                    ),
                                    thinning_degree,
                                ),
                                pot_1d_device,
                            )
                            for p in mcmc_params
                        ]
                    )
                else:  # Use ott for other keys (spline coefficients)
                    mcmc_dists["marginal"][rv] = jnp.array(
                        [
                            wasserstein_distance_ott(
                                thin_samples(
                                    jax.device_put(p, device=target_device),
                                    thinning_degree,
                                ),
                                thin_samples(
                                    jax.device_put(
                                        ref_samples[rv], device=target_device
                                    ),
                                    thinning_degree,
                                ),
                            )
                            for p in mcmc_params
                        ]
                    )

            mcmc_arrays = [
                samples_dict_to_array(run, rv_order) for run in processed_mcmc
            ]

            # Always use wasserstein_distance_ott for "full"
            mcmc_dists["full"] = jnp.array(
                [
                    wasserstein_distance_ott(
                        thin_samples(
                            jax.device_put(arr, device=target_device), thinning_degree
                        ),
                        thin_samples(
                            jax.device_put(ref_full, device=target_device),
                            thinning_degree,
                        ),
                    )
                    for arr in mcmc_arrays
                ]
            )

            mcmc_results[size_key] = mcmc_dists

    # Process SVI samples
    result = {}
    with jax.default_device(jax.devices(device)[0]):
        for epoch_key in tqdm(epoch_keys, desc="Epochs"):
            epoch_result = {}
            for size_key in tqdm(sample_size_keys, desc="Sample sizes", leave=False):
                # Get processed reference data
                ref_samples = {
                    k: process_mcmc_param(v, mcmc_samples_per_chain)
                    for k, v in reference_dict[size_key]["samples"].items()
                }
                rv_order = sorted(ref_samples.keys())
                ref_full = samples_dict_to_array(ref_samples, rv_order)

                svi_samples = svi_samples_dict[epoch_key][size_key].get("samples", [])
                if not svi_samples:
                    epoch_result[size_key] = {"svi": {}, "mcmc": {}}
                    continue

                svi_dists = {"marginal": {}, "full": []}

                # Process SVI runs
                processed_svi = [
                    {k: process_svi_param(v) for k, v in run_samples.items()}
                    for run_samples in svi_samples
                ]

                # Compute distances
                for rv in tqdm(rv_order, desc="SVI params", leave=False):
                    svi_params = [run[rv] for run in processed_svi]

                    # Choose distance function based on parameter name
                    if rv in use_ot_1d_keys:
                        svi_dists["marginal"][rv] = jnp.array(
                            [
                                wasserstein_distance_ot_1d(
                                    thin_samples(
                                        jax.device_put(
                                            p.squeeze(), device=target_device
                                        ),
                                        thinning_degree,
                                    ),
                                    thin_samples(
                                        jax.device_put(
                                            ref_samples[rv].squeeze(),
                                            device=target_device,
                                        ),
                                        thinning_degree,
                                    ),
                                    pot_1d_device,
                                )
                                for p in svi_params
                            ]
                        )
                    else:  # Use ott for other keys (spline coefficients)
                        svi_dists["marginal"][rv] = jnp.array(
                            [
                                wasserstein_distance_ott(
                                    thin_samples(
                                        jax.device_put(p, device=target_device),
                                        thinning_degree,
                                    ),
                                    thin_samples(
                                        jax.device_put(
                                            ref_samples[rv], device=target_device
                                        ),
                                        thinning_degree,
                                    ),
                                )
                                for p in svi_params
                            ]
                        )

                svi_arrays = [
                    samples_dict_to_array(run, rv_order) for run in processed_svi
                ]

                # Always use wasserstein_distance_ott for "full"
                svi_dists["full"] = jnp.array(
                    [
                        wasserstein_distance_ott(
                            thin_samples(
                                jax.device_put(arr, device=target_device),
                                thinning_degree,
                            ),
                            thin_samples(
                                jax.device_put(ref_full, device=target_device),
                                thinning_degree,
                            ),
                        )
                        for arr in svi_arrays
                    ]
                )

                epoch_result[size_key] = {
                    "svi": {"wassersteindistances": svi_dists},
                    "mcmc": {"wassersteindistances": mcmc_results.get(size_key, {})},
                }

            result[epoch_key] = epoch_result

    if save_dir and file_name_prefix:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_name_prefix}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved Wasserstein distances to {save_path}")

    return result


def load_wasserstein_results(
    save_dir: str,
    file_name_prefix: str,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Load previously computed Wasserstein distances from a pickle file."""
    file_name = f"{file_name_prefix}.pkl"
    load_path = os.path.join(save_dir, file_name)

    # Check if file exists
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No Wasserstein distances found at {load_path}")

    # Load the pickle file
    with open(load_path, "rb") as f:
        result = pickle.load(f)

    print(f"Wasserstein distances loaded from {load_path}")
    return result


def compute_log_pdfs(carry, loc_vi, split_indices, transformations, rv_names):
    """Ready to be JIT-compiled version for log pdfs"""
    matrix, Y_SYN = carry
    grouped_params = jnp.split(loc_vi, split_indices)
    transformed_params = [
        trans(param) for trans, param in zip(transformations, grouped_params)
    ]
    params_dict = dict(zip(rv_names, transformed_params))

    # Extract parameters
    beta0_scale = params_dict["intercept_scale"]
    gamma_scale = params_dict["spline_scale_coef"]
    beta0_shape = params_dict["intercept_shape"]
    gamma_shape = params_dict["spline_shape_coef"]

    # Compute linear predictors
    linear_scale = beta0_scale + matrix @ gamma_scale
    linear_shape = beta0_shape + matrix @ gamma_shape

    # Apply link functions
    scale = TransformationFunctions.softplus(linear_scale)
    shape = linear_shape

    # log pdfs
    distribution = CustomGPD(loc=0.0, scale=scale, shape=shape)
    log_pdfs = distribution.log_prob(Y_SYN)

    return carry, jnp.sum(log_pdfs)


def compute_support_checks(carry, loc_vi, split_indices, transformations, rv_names):
    """Ready to be JIT-compiled version for support checks"""
    matrix, Y_SYN, where_sample_X = carry
    grouped_params = jnp.split(loc_vi, split_indices)
    transformed_params = [
        trans(param) for trans, param in zip(transformations, grouped_params)
    ]
    params_dict = dict(zip(rv_names, transformed_params))

    # Extract parameters
    beta0_scale = params_dict["intercept_scale"]
    gamma_scale = params_dict["spline_scale_coef"]
    beta0_shape = params_dict["intercept_shape"]
    gamma_shape = params_dict["spline_shape_coef"]

    # Compute linear predictors
    linear_scale = beta0_scale + matrix @ gamma_scale
    linear_shape = beta0_shape + matrix @ gamma_shape

    # Apply link functions
    scale = TransformationFunctions.softplus(linear_scale)[where_sample_X]
    loc = 0.0
    shape = linear_shape[where_sample_X]

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


def process_epoch_data(
    beta_samples: List[jnp.ndarray],
    design_matrix: jnp.ndarray,
    split_indices: Tuple[int, ...],
    rv_names: Tuple[str, ...],
    transformations: List[callable],
    Y: jnp.ndarray,
    elbo_values: List[float],
    response_log_pdf: List[jnp.ndarray],
    log_joint_pdfs_component_wise_stacked: List[jnp.ndarray],
    log_q_pdf: List[jnp.ndarray],
    param_updates: Optional[List[jnp.ndarray]] = None,
    where_sample_X: jnp.ndarray = None,
) -> Dict[str, np.ndarray]:

    with jax.default_device(jax.devices("cpu")[0]):
        beta_samples_arr = jnp.stack(beta_samples)
        response_log_pdf_arr = jnp.stack(response_log_pdf)
        log_joint_stacked_arr = jnp.stack(log_joint_pdfs_component_wise_stacked)
        log_q_arr = jnp.stack(log_q_pdf)
        elbo_values_arr = jnp.array(elbo_values)

        def transform_params(beta_sample):
            grouped = jnp.split(beta_sample, split_indices)
            transformed = [
                trans(group) for trans, group in zip(transformations, grouped)
            ]
            return dict(zip(rv_names, transformed))

        transformed_params = jax.vmap(jax.vmap(transform_params))(beta_samples_arr)

        def compute_support(params, where_sample_X):
            with jax.default_device(jax.devices("cpu")[0]):
                beta0_shape = params["intercept_shape"]
                gamma_shape = params["spline_shape_coef"]
                linear_shape = beta0_shape + design_matrix @ gamma_shape
                filtered_linear_shape = jnp.take(linear_shape, where_sample_X, axis=-1)

                beta0_scale = params["intercept_scale"]
                gamma_scale = params["spline_scale_coef"]
                linear_scale = beta0_scale + design_matrix @ gamma_scale
                scale = jnp.take(jax.nn.softplus(linear_scale), where_sample_X, axis=-1)
                upper_bound = jnp.where(
                    filtered_linear_shape < 0, -scale / filtered_linear_shape, jnp.inf
                )
                diffs = upper_bound - Y
                valid_diffs = jnp.where(
                    (filtered_linear_shape < 0) & (diffs > 0) & (Y >= 0.0),
                    diffs,
                    jnp.inf,
                )
                support = (Y >= 0.0) & (Y <= upper_bound)
                return support, filtered_linear_shape, valid_diffs

        support_checks, shapes, valid_diffs = jax.vmap(
            jax.vmap(compute_support, in_axes=(0, None)),  # Inner vmap
            in_axes=(0, None),  # Outer vmap
        )(transformed_params, where_sample_X)

        # Compute OOS counts for VI samples
        oos_vi_counts = jnp.sum(
            ~support_checks, axis=(1, 2)
        )  # Sum over samples and data points
        oos_vi_counts = np.asarray(oos_vi_counts)

        # Compute OOS counts for param_updates if provided
        oos_update_counts = None
        if param_updates is not None:
            param_updates_arr = jnp.stack(param_updates)
            transformed_param_updates = jax.vmap(transform_params)(param_updates_arr)
            support_checks_updates, _, _ = jax.vmap(compute_support, in_axes=(0, None))(
                transformed_param_updates, where_sample_X
            )
            oos_update_counts = jnp.sum(
                ~support_checks_updates, axis=1
            )  # Sum over data points
            oos_update_counts = np.asarray(jax.device_get(oos_update_counts))

        negative_mask = (shapes < 0) & support_checks
        positive_mask = (shapes > 0) & support_checks
        oos_mask = ~support_checks

        # prior_contrib = -jnp.mean(log_joint_stacked_arr[..., :-1].sum(axis=1), axis=1)
        prior_contrib = -jnp.mean(
            jnp.sum(log_joint_stacked_arr[..., :-1], axis=2), axis=1
        )
        # response_neg = -jnp.mean(jnp.where(negative_mask, response_log_pdf_arr, 0), axis=(1, 2))
        response_neg = -jnp.mean(
            jnp.sum(jnp.where(negative_mask, response_log_pdf_arr, 0), axis=2), axis=1
        )
        # response_pos = -jnp.mean(jnp.where(positive_mask, response_log_pdf_arr, 0), axis=(1, 2))
        response_pos = -jnp.mean(
            jnp.sum(jnp.where(positive_mask, response_log_pdf_arr, 0), axis=2), axis=1
        )
        # response_oos = -jnp.mean(jnp.where(oos_mask, response_log_pdf_arr, 0), axis=(1, 2))
        response_oos = -jnp.mean(
            jnp.sum(jnp.where(oos_mask, response_log_pdf_arr, 0), axis=2), axis=1
        )
        log_q_contrib = -jnp.mean(log_q_arr, axis=1)

        min_pos_diffs = jnp.min(valid_diffs, axis=(1, 2))
        diff_min = jnp.min(min_pos_diffs)
        diff_max = jnp.max(min_pos_diffs)
        normalized_diffs = (min_pos_diffs - diff_min) / (diff_max - diff_min + 1e-8)

        return {
            "log_components": {
                "logq": log_q_contrib,
                "prior": prior_contrib,
                "response_neg": response_neg,
                "response_pos": response_pos,
                "response_oos": response_oos,
            },
            "normalized_diffs": normalized_diffs,
            "support_proportions": jnp.mean(support_checks, axis=(1, 2)),
            "elbo_peaks": elbo_values_arr > elbo_values_arr[0],
            "oos_vi_counts": oos_vi_counts,
            "oos_update_counts": oos_update_counts,
        }


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
    mvn_tril = tfd.MultivariateNormalTriL(
        final_vi_loc,
        TransformDiagonal(diag_bijector=Softplus()).forward(variational_lower_triangle),
    )
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


def compute_zcgpd_loc_scale_shape_posterior_samples(
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
        posterior_scale_samples, posterior_shape_samples = jitted_body(
            β0_scale_samples,
            γ_scale_samples,
            β0_shape_samples,
            γ_shape_samples,
            dm,
        )

    return posterior_scale_samples, posterior_shape_samples


def compute_case_study_posterior_maps_hdis_and_quick_support_check(
    svi_posterior_dict: dict,
    mcmc_posterior_dict: dict,
    matrix: jnp.ndarray,
    results_CGPD: dict,
    y: jnp.ndarray,
    mask_X_filtered,
    subset_excess_mask,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute HDIs and MAP/mean estimates for both methods."""

    def process_method(posterior_dict: dict, is_svi: bool = False):
        # Extract parameters based on method
        if is_svi:
            # Get MAP estimates from VI results
            β0_scale = results_CGPD["transformed_loc_vi_parameters"]["intercept_scale"]
            γ_scale = results_CGPD["transformed_loc_vi_parameters"]["spline_scale_coef"]
            β0_shape = results_CGPD["transformed_loc_vi_parameters"]["intercept_shape"]
            γ_shape = results_CGPD["transformed_loc_vi_parameters"]["spline_shape_coef"]

            # Compute MAP predictions
            scale_map = jax.nn.softplus(β0_scale + matrix @ γ_scale)
            shape_map = β0_shape + matrix @ γ_shape
            filtered_scale = scale_map[mask_X_filtered][subset_excess_mask]
            filtered_shape = shape_map[mask_X_filtered][subset_excess_mask]

            # Support checks
            loc = 0.0
            support_lower = y >= loc
            upper_bound = jnp.where(
                filtered_shape < 0, loc - filtered_scale / filtered_shape, jnp.inf
            )
            support_upper = y <= upper_bound
            support_checks = jnp.logical_and(support_lower, support_upper)
            print("Supportcheck SVI", jnp.mean(support_checks))
        else:
            # Compute means from MCMC posterior
            β0_scale = process_mcmc_param(
                posterior_dict["samples"][0]["intercept_scale"], 8000
            ).mean(axis=0)
            γ_scale = process_mcmc_param(
                posterior_dict["samples"][0]["spline_scale_coef"], 8000
            ).mean(axis=0)
            β0_shape = process_mcmc_param(
                posterior_dict["samples"][0]["intercept_shape"], 8000
            ).mean(axis=0)
            γ_shape = process_mcmc_param(
                posterior_dict["samples"][0]["spline_shape_coef"], 8000
            ).mean(axis=0)
            # Compute mean predictions
            scale_map = jax.nn.softplus(β0_scale + matrix @ γ_scale)
            shape_map = β0_shape + matrix @ γ_shape
            filtered_scale = scale_map[mask_X_filtered][subset_excess_mask]
            filtered_shape = shape_map[mask_X_filtered][subset_excess_mask]

            # Support checks
            loc = 0.0
            support_lower = y >= loc
            upper_bound = jnp.where(
                filtered_shape < 0, loc - filtered_scale / filtered_shape, jnp.inf
            )
            support_upper = y <= upper_bound
            support_checks = jnp.logical_and(support_lower, support_upper)
            print("Supportcheck MCMC", jnp.mean(support_checks))

        # Compute posterior samples for HDI
        if is_svi:
            # For SVI use full posterior samples
            β0_scale_samples = svi_posterior_dict["samples"][0]["intercept_scale"]
            γ_scale_samples = svi_posterior_dict["samples"][0]["spline_scale_coef"]
            β0_shape_samples = svi_posterior_dict["samples"][0]["intercept_shape"]
            γ_shape_samples = svi_posterior_dict["samples"][0]["spline_shape_coef"]
        else:
            # For MCMC use existing samples
            β0_scale_samples = process_mcmc_param(
                mcmc_posterior_dict["samples"][0]["intercept_scale"], 8000
            )
            γ_scale_samples = process_mcmc_param(
                mcmc_posterior_dict["samples"][0]["spline_scale_coef"], 8000
            )
            β0_shape_samples = process_mcmc_param(
                mcmc_posterior_dict["samples"][0]["intercept_shape"], 8000
            )
            γ_shape_samples = process_mcmc_param(
                mcmc_posterior_dict["samples"][0]["spline_shape_coef"], 8000
            )

        # Compute posterior samples
        scale_samples, shape_samples = compute_zcgpd_loc_scale_shape_posterior_samples(
            β0_scale_samples,
            γ_scale_samples,
            β0_shape_samples,
            γ_shape_samples,
            matrix,
            "gpu",
        )

        return {
            "scale_map": np.array(scale_map),
            "shape_map": np.array(shape_map),
            "scale_hdi": az.hdi(np.array(scale_samples), hdi_prob=0.95),
            "shape_hdi": az.hdi(np.array(shape_samples), hdi_prob=0.95),
        }

    return {
        "svi": process_method(svi_posterior_dict, is_svi=True),
        "mcmc": process_method(mcmc_posterior_dict),
    }


def compute_sim_study_mcmc_scale_shape_map(
    posterior_dict: dict, matrix: jnp.ndarray, thinning_degree: Optional[int]
):
    # Compute means from MCMC posterior
    β0_scale = thin_samples(
        process_mcmc_param(posterior_dict["intercept_scale"], 8000), thinning_degree
    ).mean(axis=0)
    γ_scale = thin_samples(
        process_mcmc_param(posterior_dict["spline_scale_coef"], 8000), thinning_degree
    ).mean(axis=0)
    β0_shape = thin_samples(
        process_mcmc_param(posterior_dict["intercept_shape"], 8000), thinning_degree
    ).mean(axis=0)
    γ_shape = thin_samples(
        process_mcmc_param(posterior_dict["spline_shape_coef"], 8000), thinning_degree
    ).mean(axis=0)
    # Compute mean predictions
    scale_map = jax.nn.softplus(β0_scale + matrix @ γ_scale)
    shape_map = β0_shape + matrix @ γ_shape

    return scale_map, shape_map


def compute_zcgpd_post_predictive_stats(
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
    # device = jax.devices(device)[0]
    with jax.default_device(jax.devices(device)[0]):
        loc_posterior_samples = jnp.tile(
            posterior_thresholds_map, (len(scale_posterior_samples), 1)
        )

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
            jax.jit(body_fn),  # , device=device),
            key,
            (loc_posterior_samples, scale_posterior_samples, shape_posterior_samples),
        )

        # Reshape to (n_post * n_samples, len_X)
        flat_samples = samples.reshape(-1, samples.shape[-1])

        return {
            "mean": jnp.mean(flat_samples, axis=0),
            "0.0_quantile": jnp.quantile(flat_samples, 0.0, axis=0),
            "0.25_quantile": jnp.quantile(flat_samples, 0.25, axis=0),
            "0.50_quantile": jnp.quantile(flat_samples, 0.5, axis=0),
            "0.75_quantile": jnp.quantile(flat_samples, 0.75, axis=0),
            "hdi": az.hdi(np.array(flat_samples)),
        }


def compute_prior_predictive_stats(
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
            "intercept_scale": subkeys[3],
            "lambda_scale": subkeys[4],
            "gamma_scale": subkeys[5],
            "intercept_shape": subkeys[6],
            "lambda_shape": subkeys[7],
            "gamma_shape": subkeys[8],
        }

        # Sample intercepts
        beta0_loc = tfd.Normal(0.0, 30.0).sample(
            (n_prior_samples,), seed=keys["beta0_loc"]
        )
        beta0_scale = tfd.Normal(0.0, 1.0).sample(
            (n_prior_samples,), seed=keys["intercept_scale"]
        )
        beta0_shape = tfd.Normal(0.0, 0.1).sample(
            (n_prior_samples,), seed=keys["intercept_shape"]
        )

        # Sample smoothing parameters
        lambda_loc = tfd.HalfCauchy(0.0, 0.00005).sample(
            (n_prior_samples,), seed=keys["lambda_loc"]
        )
        lambda_scale = tfd.HalfCauchy(0.0, 0.00005).sample(
            (n_prior_samples,), seed=keys["lambda_scale"]
        )
        lambda_shape = tfd.HalfCauchy(0.0, 0.00005).sample(
            (n_prior_samples,), seed=keys["lambda_shape"]
        )

        # Sample coeficients using degenerate normal
        gamma_loc = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=lambda_loc, pen=K
        ).sample(seed=keys["gamma_loc"])

        gamma_scale = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=lambda_scale, pen=K
        ).sample(seed=keys["gamma_scale"])

        gamma_shape = MultivariateNormalDegenerate.from_penalty(
            loc=jnp.zeros(K.shape[0]), var=lambda_shape, pen=K
        ).sample(seed=keys["gamma_shape"])

        loc = jnp.expand_dims(beta0_loc, axis=1) + gamma_loc @ dm.T
        scale = TransformationFunctions.softplus(
            jnp.expand_dims(beta0_scale, axis=1) + gamma_scale @ dm.T
        )
        shape = jnp.expand_dims(beta0_shape, axis=1) + gamma_shape @ dm.T

        return new_key, (loc, scale, shape)

    # Use scan to accumulate samples
    _, prior_predictives = jax.jit(partial(body_fn, n_prior_samples=n_prior_samples))(
        key, dm, K
    )
    loc, scale, shape = prior_predictives

    return {
        "loc": list(loc),
        "scale": list(scale),
        "shape": list(shape),
    }


def save_svi_results(result: dict, save_dir: str, file_name_prefix: str):
    if save_dir and file_name_prefix:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_name_prefix}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved SVI results to {save_path}")


def load_svi_results(
    save_dir: str,
    file_name_prefix: str,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Load previously computed SVI results from a pickle file."""
    file_name = f"{file_name_prefix}.pkl"
    load_path = os.path.join(save_dir, file_name)

    # Check if file exists
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No SVI result found at {load_path}")

    # Load the pickle file
    with open(load_path, "rb") as f:
        result = pickle.load(f)

    print(f"SVI result loaded from {load_path}")
    return result


def compute_wd_statistics(data):
    def get_rounding_precision(key):
        key_str = str(key)

        if key_str == "intercept_scale" or key_str == "intercept_shape":
            return 4
        elif key_str == "lambda_smooth_scale" or key_str == "lambda_smooth_shape":
            return 5
        elif (
            key_str == "spline_scale_coef"
            or key_str == "spline_shape_coef"
            or key_str == "full"
        ):
            return 5
        else:
            return 5  # Default rounding precision

    def compute_stats(arr, key):
        # Filter out NaN values
        arr_filtered = arr[~jnp.isnan(arr)]
        # If all values are NaN, return NaN for all statistics
        if arr_filtered.size == 0:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "IQR": float("nan"),
                "range": float("nan"),
            }

        # Get the appropriate rounding precision based on the key
        precision = get_rounding_precision(key)

        # Compute the interquartile range (IQR)
        quantile_025 = jnp.quantile(arr_filtered, 0.25)
        quantile_075 = jnp.quantile(arr_filtered, 0.75)
        iqr = quantile_075 - quantile_025

        return {
            "mean": round(jnp.mean(arr_filtered).item(), precision),
            "median": round(jnp.median(arr_filtered).item(), precision),
            "025_quantile": round(quantile_025.item(), precision),
            "075_quantile": round(quantile_075.item(), precision),
            "IQR": round(iqr.item(), precision),
            "range": round(
                (jnp.max(arr_filtered) - jnp.min(arr_filtered)).item(), precision
            ),
        }

    def process_dict(d):
        processed = {}
        for key, value in d.items():
            if isinstance(value, dict):
                processed[key] = process_dict(value)
            elif isinstance(value, jnp.ndarray):
                processed[key] = compute_stats(value, key)
            else:
                processed[key] = value
        return processed

    return process_dict(data)


def check_nans_in_svi_structure(data):
    def check_array(arr):
        return bool(jnp.isnan(arr).any())

    def process_dict(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = process_dict(v)
            elif k in ["loc", "chol"] and isinstance(v, list):
                result[k] = [check_array(arr) for arr in v]
            else:
                result[k] = v
        return result

    return process_dict(data)


def check_nans_in_mcmc_structure(data):
    target_params = [
        "lambda_smooth_scale",
        "lambda_smooth_shape",
        "spline_shape_coef",
        "spline_scale_coef",
        "intercept_shape",
        "intercept_scale",
    ]

    def check_array(arr):
        return bool(jnp.isnan(arr).any())

    def process_samples(samples):
        results = {param: [] for param in target_params}
        for sample in samples:
            for param in target_params:
                param_value = sample.get(param, [])
                if not isinstance(param_value, list):
                    param_value = [param_value]
                for arr in param_value:
                    if isinstance(arr, jnp.ndarray):
                        results[param].append(check_array(arr))
                    else:
                        results[param].append(False)
        return results

    def process_entry(entry):
        if isinstance(entry, dict):
            result = {}
            for key, value in entry.items():
                if key == "samples" and isinstance(value, list):
                    result[key] = process_samples(value)
                else:
                    result[key] = process_entry(value)
            return result
        elif isinstance(entry, list):
            return [process_entry(item) for item in entry]
        else:
            return entry

    return process_entry(data)


def combine_loc_chol_nan_checks(data):
    def process_node(node):
        if isinstance(node, dict):
            # Check if this dict contains exactly 'loc' and 'chol' keys
            keys = set(node.keys())
            if keys == {"loc", "chol"}:
                # If so, replace the entire dict with the combined list
                return sum([a or b for a, b in zip(node["loc"], node["chol"])])
            else:
                # Otherwise, process all keys recursively
                return {key: process_node(value) for key, value in node.items()}
        elif isinstance(node, list):
            return [process_node(item) for item in node]
        else:
            return node

    return process_node(data)
