import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class CustomTFDGPD(tfd.GeneralizedPareto):
    def __init__(
        self,
        loc,
        scale,
        shape,
        support_penalty: float = 1e6,
        validate_args=False,
        allow_nan_stats=True,
        name="CustomGPD",
    ):
        super().__init__(
            loc=loc,
            scale=scale,
            concentration=shape,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        self.support_penalty = support_penalty

    @property
    def shape(self):
        """Distribution parameter for shape."""
        return self.concentration

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype, num_classes)

    def log_prob(self, value):
        left = self.loc
        right = self.loc - self.scale / self.shape

        x_safe = self.loc  # (left + right) / 2

        safe_value = jnp.where(
            value >= left,
            jnp.where(self.shape >= 0, value, jnp.where(value <= right, value, x_safe)),
            x_safe,
        )

        y = jnp.where(
            self.shape >= 0,
            value - self.loc,
            jnp.where(
                value <= self.loc,
                value - self.loc,
                1 + self.concentration * ((value - self.loc) / self.scale),
            ),
        )

        log_prob = super().log_prob(safe_value)

        support_violations = jnp.where(
            value >= self.loc,
            jnp.where(self.shape >= 0, False, jnp.where(value <= right, False, True)),
            True,
        )

        n_support_violations = jnp.maximum(support_violations.sum(), 1.0)

        return jnp.where(
            support_violations,
            (y - 0.5) * (self.support_penalty / n_support_violations),
            log_prob,
        )

    def cdf(self, value):
        left = self.loc
        right = self.loc - self.scale / self.shape

        x_safe = self.loc  # (left + right) / 2

        safe_value = jnp.where(
            value >= left,
            jnp.where(self.shape >= 0, value, jnp.where(value <= right, value, x_safe)),
            x_safe,
        )

        y = jnp.where(
            self.shape >= 0,
            value - self.loc,
            jnp.where(
                value <= self.loc,
                value - self.loc,
                1 + self.concentration * ((value - self.loc) / self.scale),
            ),
        )

        cdf = super().cdf(safe_value)

        support_violations = jnp.where(
            value >= self.loc,
            jnp.where(self.shape >= 0, False, jnp.where(value <= right, False, True)),
            True,
        )

        n_support_violations = jnp.maximum(support_violations.sum(), 1.0)

        return jnp.where(
            support_violations,
            (y - 0.5) * (self.support_penalty / n_support_violations),
            cdf,
        )
