from typing import Tuple

import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.random import PRNGKey
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.bijectors import Softplus, TransformDiagonal
from tensorflow_probability.substrates.jax.math import (
    fill_triangular_inverse,
    fill_triangular,
)
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalTriL,
    MultivariateNormalDiag,
)

tfd = tfp.distributions


class VariationalDistribution:
    """Base class for variational distributions."""

    def __init__(self, name: str, num_vi: int):
        """Initialize the variational distribution."""
        if num_vi is None or num_vi < 0:
            raise ValueError("Dimension must be a positive integer.")
        self.name = name
        self.num_vi = num_vi


class FullCovarianceNormal(VariationalDistribution):
    """Multivariate normal distribution with full covariance matrix."""

    def __init__(self, num_vi: int):
        super().__init__("full_covariance_normal", num_vi)

    def initialize_parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        eye_matrix = jnp.eye(self.num_vi)
        inv_sp_cholesky = TransformDiagonal(Softplus()).inverse(eye_matrix)
        flattened_inv_sp_cholesky = fill_triangular_inverse(inv_sp_cholesky)
        return tuple([jnp.zeros(self.num_vi), stop_gradient(flattened_inv_sp_cholesky)])

    @staticmethod
    def sample(
        variational_loc: jnp.ndarray,
        variational_scale_tril: jnp.ndarray,
        vi_sample_prngkey: PRNGKey,
        sample_size: int,
    ) -> jnp.ndarray:
        variational_lower_triangle = fill_triangular(variational_scale_tril)
        mvn_tril = MultivariateNormalTriL(variational_loc, variational_lower_triangle)
        return mvn_tril.sample((sample_size,), vi_sample_prngkey)

    @staticmethod
    def log_pdf(
        samples: jnp.ndarray,
        variational_loc: jnp.ndarray,
        variational_scale_tril: jnp.ndarray,
    ) -> jnp.ndarray:
        variational_lower_triangle = fill_triangular(variational_scale_tril)
        mvn_tril = MultivariateNormalTriL(variational_loc, variational_lower_triangle)
        return jnp.sum(mvn_tril.log_prob(samples))


class MeanFieldNormal(VariationalDistribution):
    """Multivariate normal distribution with diagonal covariance matrix."""

    def __init__(self, num_vi: int):
        super().__init__("diagonal_normal", num_vi)

    def initialize_parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        variances = jnp.ones(self.num_vi)
        inv_sp_variances = Softplus().inverse(variances)
        return tuple([jnp.zeros(self.num_vi), stop_gradient(inv_sp_variances)])

    @staticmethod
    def sample(
        variational_loc: jnp.ndarray,
        variational_diag_scale: jnp.ndarray,
        vi_sample_prngkey: PRNGKey,
        sample_size: int,
    ) -> jnp.ndarray:
        mvn_diag = MultivariateNormalDiag(variational_loc, variational_diag_scale)
        return mvn_diag.sample((sample_size,), vi_sample_prngkey)

    @staticmethod
    def log_pdf(
        samples: jnp.ndarray,
        variational_loc: jnp.ndarray,
        variational_diag_scale: jnp.ndarray,
    ) -> jnp.ndarray:
        mvn_diag = MultivariateNormalDiag(variational_loc, variational_diag_scale)
        return jnp.sum(mvn_diag.log_prob(samples))
