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


import optax
import jax.numpy as jnp

def warmup_constant_schedule(init_value: float, peak_value: float, warmup_steps: int) -> optax.Schedule:
    """
    Returns a schedule that linearly increases the scalar from init_value to peak_value
    over warmup_steps, then remains constant at peak_value for all subsequent steps.
    """
    def schedule(step: int) -> float:
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup_steps_f = jnp.asarray(warmup_steps, dtype=jnp.float32)
        warmup_value = init_value + (peak_value - init_value) * (step / warmup_steps_f)
        return jnp.where(step < warmup_steps, warmup_value, peak_value)
    return schedule

def prepare_scheduler(scheduler_type: str, lr: float, total_steps: int, **kwargs):
    """Returns a learning rate scheduler based on the specified type.

    Args:
        scheduler_type: Either 'constant', 'step', 'warmup_cosine_decay', 'warmup_constant',
                        or 'cosine_decay'
        lr: Base learning rate (peak value)
        total_steps: Total number of training steps
        **kwargs: Additional parameters needed for the scheduler:
            For 'step' scheduler:
                - step_events: List with number of iteration after which to drop the lr
                - drop_magnitude: Factor by which to drop the LR
            For 'warmup_cosine_decay' scheduler:
                - warmup_fraction: Fraction of total steps for warmup phase
                - end_value: Final learning rate (default: 1e-7)
                - exponent: Cosine decay exponent (default: 1.0)
                - init_value: Initial learning rate before warmup (default: 1e-7)
            For 'warmup_constant' scheduler:
                - warmup_fraction: Fraction of total steps for warmup phase
                - init_value: Initial learning rate before warmup (default: 1e-7)
            For 'cosine_decay' scheduler:
                - alpha: Minimum learning rate multiplier (default: 0.0)
                - exponent: Cosine decay exponent (default: 1.0)

    Returns:
        optax.Schedule: The requested learning rate schedule
    """
    if scheduler_type == "constant":
        return optax.constant_schedule(lr)

    elif scheduler_type == "warmup_cosine_decay":
        # Validate required parameters
        required = ["warmup_fraction"]
        if any(key not in kwargs for key in required):
            missing = [key for key in required if key not in kwargs]
            raise ValueError(
                f"Missing required parameters for warmup_cosine_decay scheduler: {missing}"
            )

        warmup_fraction = kwargs["warmup_fraction"]
        init_value = kwargs.get("init_value", 1e-7)
        end_value = kwargs.get("end_value", 1e-7)
        exponent = kwargs.get("exponent", 1.0)

        if not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")

        warmup_steps = int(warmup_fraction * total_steps)

        return optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=end_value,
            exponent=exponent,
        )

    elif scheduler_type == "warmup_constant":
        # New branch: linear warmup followed by a constant schedule.
        required = ["warmup_fraction"]
        if any(key not in kwargs for key in required):
            missing = [key for key in required if key not in kwargs]
            raise ValueError(
                f"Missing required parameters for warmup_constant scheduler: {missing}"
            )
        warmup_fraction = kwargs["warmup_fraction"]
        if not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        warmup_steps = int(total_steps * warmup_fraction)
        init_value = kwargs.get("init_value", 1e-7)
        return warmup_constant_schedule(init_value=init_value, peak_value=lr, warmup_steps=warmup_steps)

    elif scheduler_type == "step":
        # Validate required parameters
        required = ["step_events", "drop_magnitude"]
        if any(key not in kwargs for key in required):
            missing = [key for key in required if key not in kwargs]
            raise ValueError(
                f"Missing required parameters for step scheduler: {missing}"
            )

        step_events = kwargs["step_events"]
        drop_magnitude = kwargs["drop_magnitude"]

        # Validate parameter values
        if not isinstance(step_events, list) or not step_events:
            raise ValueError("step_events must be a non-empty list of integers")
        for step in step_events:
            if not isinstance(step, int) or step < 0:
                raise ValueError(f"Invalid step {step} in step_events: must be non-negative integers")
            if step >= total_steps:
                raise ValueError(f"Step event {step} exceeds total_steps {total_steps}")
        if not 0 < drop_magnitude <= 1:
            raise ValueError("drop_magnitude should be between 0 (exclusive) and 1 (inclusive)")

        # Use step_events as the boundaries
        boundaries_and_scales = {int(step): drop_magnitude for step in step_events}

        return optax.piecewise_constant_schedule(lr, boundaries_and_scales)

    elif scheduler_type == "cosine_decay":
        # Optional parameters for cosine decay
        alpha = kwargs.get("alpha", 0.0)
        exponent = kwargs.get("exponent", 1.0)
        return optax.cosine_decay_schedule(
            init_value=lr, decay_steps=total_steps, alpha=alpha, exponent=exponent
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")



def prepare_opt_state(
    sgd_method: Type[optax.GradientTransformation],
    init_vi_parameters: Tuple[jnp.ndarray, jnp.ndarray],
    optax_scheduler: optax.Schedule,
    max_norm: Optional[float] = None,
    clip_min_max_enabled: bool = False,
    zero_nans_enabled: bool = False,
) -> Tuple[optax.OptState, optax.GradientTransformation]:
    optimizer = sgd_method(optax_scheduler)  # e.g. optax.sgd or optax.adam or ...

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
