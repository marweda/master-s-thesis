from typing import Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey


def calculate_batch_metrics(
    len_responses: int, batch_size: int, epochs: int
) -> Tuple[int, int, int, int]:
    """Calculate batch-related metrics.
    Returns:
        Tuple of (num_complete_mini_batches, num_mini_batches, missing_data, total_num_mini_batches)
    """
    num_complete_mini_batches, remainder = jnp.divmod(len_responses, batch_size)
    num_mini_batches = num_complete_mini_batches + (1 if remainder > 0 else 0)
    missing_data = (batch_size - remainder) * (1 if remainder > 0 else 0)
    total_num_mini_batches = num_mini_batches * epochs

    return (
        num_complete_mini_batches,
        num_mini_batches,
        missing_data,
        total_num_mini_batches,
    )


def create_observation_pointers(
    len_responses: int, missing_data: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create observation pointers and padding mask."""
    observation_pointers = jnp.arange(len_responses, dtype=jnp.int32)
    observation_pointers_for_padding = jnp.arange(
        len_responses, len_responses + missing_data, dtype=jnp.int32
    )

    padded_pointers = jnp.concatenate(
        [observation_pointers, observation_pointers_for_padding]
    )

    padding_mask = jnp.concatenate([jnp.ones(len_responses), jnp.zeros(missing_data)])

    return padded_pointers, padding_mask


def create_epoch_arrays(
    pointers: jnp.ndarray, padding_mask: jnp.ndarray, epochs: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create arrays for all epochs."""
    return (
        jnp.tile(pointers, epochs).reshape(epochs, -1),
        jnp.tile(padding_mask, epochs).reshape(epochs, -1),
    )


def permute_arrays(
    pointers: jnp.ndarray, padding_mask: jnp.ndarray, prng_key: PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Permute the arrays randomly."""
    return (
        jax.random.permutation(prng_key, pointers, axis=1).ravel(),
        jax.random.permutation(prng_key, padding_mask, axis=1).ravel(),
    )


def reshape_and_sort_batches(
    pointers: jnp.ndarray,
    padding_mask: jnp.ndarray,
    total_num_mini_batches: int,
    batch_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reshape arrays into batches and sort them."""
    # Reshape into batches
    mini_batch_pointers = pointers.reshape(total_num_mini_batches, batch_size)
    mini_batch_padding_masks = padding_mask.reshape(total_num_mini_batches, batch_size)

    # Sort for efficiency
    sort_indices = jnp.argsort(mini_batch_pointers, axis=1)
    batch_indices = jnp.arange(total_num_mini_batches)[:, jnp.newaxis]

    return (
        mini_batch_pointers[batch_indices, sort_indices],
        mini_batch_padding_masks[batch_indices, sort_indices],
    )


def create_mini_batch_pointers(
    len_responses: int, epochs: int, batch_size: int, prng_key: PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Creates mini batch pointers and padding masks using function composition."""
    _, _, missing_data, total_num_mini_batches = calculate_batch_metrics(
        len_responses, batch_size, epochs
    )

    pointers, padding_mask = create_observation_pointers(len_responses, missing_data)

    epoch_pointers, epoch_padding = create_epoch_arrays(pointers, padding_mask, epochs)

    permuted_pointers, permuted_padding = permute_arrays(
        epoch_pointers, epoch_padding, prng_key
    )

    sorted_batches, sorted_masks = reshape_and_sort_batches(
        permuted_pointers, permuted_padding, total_num_mini_batches, batch_size
    )

    return sorted_batches, sorted_masks, missing_data


def pad_array(array: jnp.ndarray, padding_data: jnp.ndarray) -> jnp.ndarray:
    """Pad an array with the given padding data."""
    return jnp.concatenate([array, padding_data])


def prepare_mini_batching(
    responses: jnp.ndarray,
    design_matrix: jnp.ndarray,
    epochs: int,
    batch_size: int,
    prng_key: PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare data for mini batching using functional composition.

    Returns:
        (sorted_mini_batches, sorted_padding_masks, responses_padded, design_matrix_padded)
    """
    sorted_mini_batches, sorted_padding_masks, missing_data = (
        create_mini_batch_pointers(len(responses), epochs, batch_size, prng_key)
    )

    # Prepare padding data
    responses_padding = responses[:missing_data]
    responses_padded = pad_array(responses, responses_padding)

    # Create functions for padding design matrices
    get_padding = lambda x: x[:missing_data]
    design_matrices_padding = get_padding(design_matrix)

    pad_design_matrix = lambda x, y: pad_array(x, y)
    design_matrix_padded = pad_design_matrix(design_matrix, design_matrices_padding)

    return (
        sorted_mini_batches,
        sorted_padding_masks,
        responses_padded,
        design_matrix_padded,
    )
