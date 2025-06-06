from typing import Callable, Tuple, Optional
from functools import partial
from operator import add

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
    capture_intermediate: bool,
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
            # Replace dynamic_slice_in_dim with NumPy-style indexing
            design_matrix_mb[:, dp_idxs[1][0] : dp_idxs[1][0] + dp_idxs[1][1]],
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
    add_vi_elements: Tuple[Tuple[jnp.ndarray]] = jax.tree_map(
        lambda add_idx: jax.tree_map(
            lambda idx: jnp.array([0]) if idx == -1 else beta_samples_split[idx],
            add_idx[0],
        ),
        add_idxs,
        is_leaf=lambda x: isinstance(x[0][0], int),
    )
    add_dp_elements: Tuple[Tuple[jnp.ndarray]] = jax.tree.map(
        lambda add_idx: jax.tree.map(
            lambda idx: jnp.array([0]) if idx == -1 else dp_results[idx], add_idx[1]
        ),
        add_idxs,
        is_leaf=lambda x: isinstance(x[0][0], int),
    )
    add_pairs_unadjusted_axis: Tuple[Tuple[jnp.ndarray]] = jax.tree_map(
        lambda a, b: (*a, *b),
        add_vi_elements,
        add_dp_elements,
        is_leaf=lambda x: isinstance(x[0], jnp.ndarray),
    )
    add_pairs_adjusted_axis = jax.tree.map(
        lambda x: x[:, jnp.newaxis] if x.ndim == 1 else x,
        add_pairs_unadjusted_axis,
        is_leaf=lambda x: isinstance(x, jnp.ndarray),
    )
    add_results: Tuple[Tuple[jnp.ndarray]] = jax.tree.map(
        lambda x: jax.tree.reduce(add, x),
        add_pairs_adjusted_axis,
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
        is_leaf=lambda x: callable(x),
    )
    response_log_pdf = log_joint_pdfs[-1]
    summed_over_y_response_log_joint_pdf = jnp.sum(response_log_pdf, axis=1)
    log_joint_pdfs_component_wise_stacked = jnp.column_stack(
        jax.tree.leaves(log_joint_pdfs[:-1]) + [summed_over_y_response_log_joint_pdf]
    )  # (vi_samples, #logpdfs)
    row_wise_summed_log_joint_pdfs = jnp.sum(
        log_joint_pdfs_component_wise_stacked, axis=1
    )  # (vi_samples,)
    elbo = row_wise_summed_log_joint_pdfs - log_q_pdf
    final_loss = -jnp.mean(elbo)
    if capture_intermediate:
        return final_loss, (
            response_log_pdf,
            log_joint_pdfs_component_wise_stacked,
            log_q_pdf,
            beta_samples,
        )
    else:
        return final_loss, ()


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
    capture_intermediate: bool,
    use_mb: bool,
) -> Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, optax.GradientTransformation, PRNGKey],
    float,
]:
    """Update step for the SVI optimization loop."""
    vi_parameters, responses_padded, design_matrix_padded, opt_state, prng_key = carry

    subkey_carry, subkey_elbo_vi_sampling = jax.random.split(prng_key)

    if use_mb:
        mb_pointers_slice, masks_slice = xs_slice
        Y_mini_batch = jnp.take(
            responses_padded,
            mb_pointers_slice,
            axis=0,
            mode="fill",
            unique_indices=True,
            indices_are_sorted=True,
        )
        X_mini_batch = jnp.take(
            design_matrix_padded,
            mb_pointers_slice,
            axis=0,
            mode="fill",
            unique_indices=True,
            indices_are_sorted=True,
        )
        masks_mb = masks_slice
    else:
        Y_mini_batch = responses_padded
        X_mini_batch = design_matrix_padded
        masks_mb = jnp.ones_like(Y_mini_batch, dtype=bool)

    (loss, aux), grads = value_and_grad_obj(
        vi_parameters,
        Y_mini_batch,
        X_mini_batch,
        masks_mb,
        subkey_elbo_vi_sampling,
    )

    if capture_intermediate:
        response_log_pdf, log_pdfs, log_q, beta_samples = aux
    else:
        response_log_pdf, log_pdfs, log_q, beta_samples = None, None, None, None

    updates, new_opt_state = optimizer.update(grads, opt_state, vi_parameters)
    new_vi_parameters = optax.apply_updates(vi_parameters, updates)
    new_carry = (
        new_vi_parameters,
        responses_padded,
        design_matrix_padded,
        new_opt_state,
        subkey_carry,
    )

    return new_carry, (
        loss,
        new_vi_parameters[0],
        response_log_pdf,
        log_pdfs,
        log_q,
        beta_samples,
    )


def core_svi_optimization(
    responses_padded: jnp.ndarray,
    design_matrix_padded: jnp.ndarray,
    mb_pointers: Optional[jnp.ndarray],  # Now Optional
    masks: Optional[jnp.ndarray],  # Now Optional
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
    use_mb: bool,
    capture_intermediate: bool,
    capture_profil: bool,
    num_iterations: int,
) -> tuple:
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
        capture_intermediate=capture_intermediate,
    )

    value_and_grad_obj = jax.value_and_grad(
        curried_compute_neg_mc_elbo, has_aux=True, argnums=0
    )

    curried_update_step = partial(
        update_step,
        value_and_grad_obj=value_and_grad_obj,
        optimizer=optimizer,
        capture_intermediate=capture_intermediate,
        use_mb=use_mb,
    )

    if use_mb:
        xs = (mb_pointers, masks)
    else:
        # Create dummy xs with shape (num_iterations, 0) to satisfy scan
        xs = (
            jnp.zeros((num_iterations,), dtype=bool),
            jnp.zeros((num_iterations,), dtype=bool),
        )

    carry = (
        init_vi_parameters,
        responses_padded,
        design_matrix_padded,
        init_opt_state,
        prng_key,
    )

    @jax.jit
    def jitted_scan_loop(carry, xs):
        return jax.lax.scan(f=curried_update_step, init=carry, xs=xs)

    if capture_profil:
        with jax.profiler.trace("/tmp/tensorboard"):
            final_carry, outputs = jitted_scan_loop(carry, xs)
            jax.tree_map(lambda x: x.block_until_ready(), outputs)
    else:
        final_carry, outputs = jitted_scan_loop(carry, xs)

    losses, vi_params_history, response_log_pdf, log_pdfs, log_q, beta_samples = outputs
    return (
        final_carry,
        losses,
        vi_params_history,
        response_log_pdf,
        log_pdfs,
        log_q,
        beta_samples,
    )
