from typing import Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax.random import PRNGKey


def compute_neg_mc_elbo(
    variational_parameters: Tuple[jnp.ndarray, jnp.ndarray],
    responses_mb: jnp.ndarray,
    design_matrix_mb: jnp.ndarray,
    masks_mb: jnp.ndarray,
    vi_sampling_prngkey: PRNGKey,
    joint_log_pdf_funcs: Tuple[Callable, ...],
    transformations: Tuple[Callable, ...],
    split_idxs: Tuple[int, ...],
    dp_idxs: Tuple[Tuple[int, Tuple[int, int]]],
    add_idxs: Tuple[Tuple[int, Tuple[int, ...]]],
    arg_idxs: Tuple[Tuple[int, ...]],
    vi_sample_func: Callable,
    vi_log_pdf_func: Callable,
) -> float:
    beta_samples = vi_sample_func(
        variational_parameters[0], variational_parameters[1], vi_sampling_prngkey
    )
    log_q_pdf = vi_log_pdf_func(
        beta_samples, variational_parameters[0], variational_parameters[1]
    )

    beta_samples_split: Tuple[jnp.ndarray] = tuple(
        jnp.split(beta_samples, split_idxs, axis=1)
    )
    dp_pairs: Tuple[Tuple[jnp.ndarray]] = jax.tree.map(
        lambda dp_idxs: (
            jax.lax.dynamic_slice_in_dim(
                design_matrix_mb, dp_idxs[1][0], dp_idxs[1][1], 1
            ),
            beta_samples_split[dp_idxs[0]],
        ),
        dp_idxs,
        is_leaf=lambda x: isinstance(x[0], int),
    )
    dp_results: Tuple[jnp.ndarray] = jax.tree.map(
        lambda x: jnp.einsum("nk,sk->sn", x[0], x[1]),
        dp_pairs,
        is_leaf=lambda x: isinstance(x[0], jnp.ndarray),
    )
    add_vi_elements: Tuple[Tuple[jnp.ndarray]] = jax.tree.map(
        lambda add_idx: jax.tree.map(lambda idx: beta_samples_split[idx], add_idx[0]),
        add_idxs,
        is_leaf=lambda x: isinstance(x[0][0], int),
    )
    add_dp_elements: Tuple[Tuple[jnp.ndarray]] = jax.tree.map(
        lambda add_idx: jax.tree.map(lambda idx: dp_results[idx], add_idx[1]),
        add_idxs,
        is_leaf=lambda x: isinstance(x[0][0], int),
    )
    add_pairs: Tuple[Tuple[jnp.ndarray]] = jax.tree.map(
        lambda add_vi, add_dp: (add_vi, add_dp), add_vi_elements, add_dp_elements
    )
    add_results: Tuple[Tuple[jnp.ndarray]] = jax.tree_map(
        lambda x: (x[0][:, jnp.newaxis] if x[0].ndim == 1 else x[0])
        + (x[1][:, jnp.newaxis] if x[1].ndim == 1 else x[1]),
        add_pairs,
        is_leaf=lambda x: isinstance(x[0], jnp.ndarray),
    )
    add_results_flattened: Tuple[jnp.ndarray] = tuple(
        jax.tree.flatten(add_results, is_leaf=lambda x: isinstance(x, jnp.ndarray))[0]
    )
    params_for_transformation: Tuple[jnp.ndarray] = jax.tree.map(
        lambda x: x,
        tuple(jax.tree.flatten((beta_samples_split, add_results_flattened))[0]),
    )
    transformed_params: list[jnp.ndarray] = jax.tree.map(
        lambda transform, values: transform(values),
        transformations,
        params_for_transformation,
    )
    all_log_pdf_arguments: list[jnp.ndarray] = jax.tree.map(
        lambda x: x,
        jax.tree.flatten((transformed_params, masks_mb, responses_mb))[0],
    )
    selected_log_pdf_args: Tuple[Tuple[jnp.ndarray]] = jax.tree.map(
        lambda idxs: jax.tree.map(lambda idx: all_log_pdf_arguments[idx], idxs),
        arg_idxs,
        is_leaf=lambda x: isinstance(x[0], int),
    )
    log_joint_pdfs: Tuple[jnp.ndarray] = jax.tree.map(
        lambda f, args: f(*args),
        joint_log_pdf_funcs,
        selected_log_pdf_args,
        is_leaf=lambda x: callable(x) or isinstance(x, jnp.ndarray),
    )
    log_joint_pdfs_nd_array = jnp.column_stack(log_joint_pdfs)
    log_joint_pdfs_collapsed = jnp.sum(log_joint_pdfs_nd_array, axis=1)
    sum_log_joint_pdfs = jnp.sum(log_joint_pdfs_collapsed)
    elbo = sum_log_joint_pdfs - log_q_pdf
    return -jnp.mean(elbo)


def update_step(
    carry: Tuple[
        Tuple[jnp.ndarray],
        jnp.ndarray,
        jnp.ndarray,
        optax.GradientTransformation,
        PRNGKey,
    ],
    xs_slice: Tuple[jnp.ndarray, jnp.ndarray],
    value_and_grad_obj: Callable,
    optimizer: optax.GradientTransformation,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, optax.GradientTransformation, PRNGKey],
    float,
]:
    """Update step for the SVI optimization loop."""
    mb_pointers, masks = xs_slice
    vi_parameters, responses_padded, design_matrix_padded, opt_state, prng_key = carry

    subkey_carry, subkey_elbo_vi_sampling = jax.random.split(prng_key)

    Y_mini_batch = jnp.take(
        responses_padded,
        mb_pointers,
        axis=0,
        mode="fill",
        unique_indices=True,
        indices_are_sorted=True,
    )

    X_mini_batch = jnp.take(
        design_matrix_padded,
        mb_pointers,
        axis=0,
        mode="fill",
        unique_indices=True,
        indices_are_sorted=True,
    )

    loss, grads = value_and_grad_obj(
        vi_parameters,
        Y_mini_batch,
        X_mini_batch,
        masks,
        subkey_elbo_vi_sampling,
    )

    updates, new_opt_state = optimizer.update(grads, opt_state, vi_parameters)
    new_vi_parameters = optax.apply_updates(vi_parameters, updates)
    new_carry = (
        new_vi_parameters,
        responses_padded,
        design_matrix_padded,
        new_opt_state,
        subkey_carry,
    )

    return new_carry, loss


def core_svi_optimization(
    responses_padded: jnp.ndarray,
    design_matrix_padded: jnp.ndarray,
    mb_pointers: jnp.ndarray,
    masks: jnp.ndarray,
    joint_log_pdfs_funcs: Tuple[Callable, ...],
    transformations: Tuple[Callable, ...],
    vi_sample_func: Callable,
    vi_log_pdf_func: Callable,
    optimizer: optax.GradientTransformation,
    init_opt_state: optax.GradientTransformation,
    init_vi_parameters: Tuple[jnp.ndarray, jnp.ndarray],
    prng_key: PRNGKey,
    split_idxs: Tuple[int, ...],
    dp_idxs: Tuple[Tuple[int, Tuple[int, int]]],
    add_idxs: Tuple[Tuple[int, Tuple[int, ...]]],
    arg_idxs: Tuple[Tuple[int, ...]],
) -> tuple:
    """The core SVI optimization loop."""
    curried_compute_neg_mc_elbo = partial(
        compute_neg_mc_elbo,
        joint_log_pdf_funcs=joint_log_pdfs_funcs,
        transformations=transformations,
        split_idxs=split_idxs,
        dp_idxs=dp_idxs,
        add_idxs=add_idxs,
        arg_idxs=arg_idxs,
        vi_sample_func=vi_sample_func,
        vi_log_pdf_func=vi_log_pdf_func,
    )
    value_and_grad_obj = jax.value_and_grad(curried_compute_neg_mc_elbo, argnums=0)

    curried_update_step = partial(
        update_step,
        value_and_grad_obj=value_and_grad_obj,
        optimizer=optimizer,
    )

    jitted_update_step = jax.jit(curried_update_step)

    xs = (mb_pointers, masks)
    carry = (
        init_vi_parameters,
        responses_padded,
        design_matrix_padded,
        init_opt_state,
        prng_key,
    )

    final_carry, losses = jax.lax.scan(f=jitted_update_step, init=carry, xs=xs)
    return final_carry, losses
