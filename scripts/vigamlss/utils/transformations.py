from dataclasses import dataclass
from typing import TypeAlias

import jax.numpy as jnp
from tensorflow_probability.substrates.jax.bijectors import Softplus, Tanh


@dataclass
class TransformationFunctions:
    """Standard parameter transformation functions."""

    @staticmethod
    def identity(x: jnp.ndarray) -> jnp.ndarray:
        """Identity transformation leaving input unchanged."""
        return x

    @staticmethod
    def softplus(x: jnp.ndarray) -> jnp.ndarray:
        """Softplus transformation ensuring positive outputs with minimum threshold.

        Applies f(x) = log(1 + exp(x)) with a minimum value of 1e-6 to ensure
        numerical stability.
        """
        return jnp.maximum(Softplus().forward(x), 1e-6)
    
    @staticmethod
    def real_to_negative_half(x: jnp.ndarray) -> jnp.ndarray:
        """A transformation that maps real numbers to (-∞, 0.5)."""
        return jnp.maximum(0.5 - Softplus().forward(x), 1e-6)
    
    @staticmethod
    def tanh(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(Tanh().forward(x) * 0.5, 1e-6)