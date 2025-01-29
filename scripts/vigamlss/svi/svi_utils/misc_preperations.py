from functools import partial
from typing import Callable, Optional, Tuple, Type

import jax
from jax import numpy as jnp
import optax

from .. import VariationalDistribution


def clip_min_max() -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        mins = jax.tree.map(
            lambda g: jnp.where(jnp.isneginf(g), +jnp.inf, g).min(), updates
        )
        maxs = jax.tree.map(
            lambda g: jnp.where(jnp.isposinf(g), -jnp.inf, g).max(), updates
        )

        updates = jax.tree.map(
            lambda g, min: jnp.where(jnp.isneginf(g), min, g), updates, mins
        )
        updates = jax.tree.map(
            lambda g, max: jnp.where(jnp.isposinf(g), max, g), updates, maxs
        )

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def prepare_opt_state(
    sgd_method: Type[optax.GradientTransformation],
    lr: float,
    init_vi_parameters: Tuple[jnp.ndarray, jnp.ndarray],
    max_norm: Optional[float] = None,
    clip_min_max_enabled: bool = False,
    zero_nans_enabled: bool = False,
    optax_scheduler: optax.Schedule = optax.constant_schedule,
) -> Tuple[optax.OptState, optax.GradientTransformation]:
    scheduler = optax_scheduler(value=lr)
    optimizer = sgd_method(scheduler)  # e.g. optax.sgd or optax.adam or ...

    transformations = []

    if zero_nans_enabled:
        transformations.append(optax.zero_nans())

    if max_norm is not None:
        transformations.append(optax.clip_by_global_norm(max_norm))

    if clip_min_max_enabled:
        transformations.append(clip_min_max())

    transformations.append(optimizer)

    chained_optimizer = optax.chain(*transformations)

    init_opt_state = chained_optimizer.init(init_vi_parameters)
    return init_opt_state, chained_optimizer


def prepare_vi_dist(
    vi_dist: VariationalDistribution,
    vi_sample_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Callable, Callable]:
    """Prepares the VI distribution for the optimization loop."""
    vi_log_pdf_func = vi_dist.log_pdf
    vi_sample_func = vi_dist.sample
    init_vi_dist_params = vi_dist.initialize_parameters()
    curried_vi_dist_sample = partial(vi_sample_func, sample_size=vi_sample_size)
    return init_vi_dist_params, curried_vi_dist_sample, vi_log_pdf_func
