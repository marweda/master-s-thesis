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


class CustomTFDGEV(tfd.GeneralizedExtremeValue):
    def __init__(
        self,
        loc,
        scale,
        shape,
        support_penalty: float = 1e6,
        validate_args=False,
        allow_nan_stats=True,
        name="CustomGEV",
    ):
        super().__init__(
            loc=loc,
            scale=scale,
            concentration=shape,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        # Create a Gumbel distribution (used if concentration == 0)
        self.gumbel_dist = tfd.Gumbel(loc=loc, scale=scale)
        self.support_penalty = support_penalty

    @property
    def shape(self):
        """Distribution parameter for shape."""
        return self._gev_bijector.shape

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # Explicitly using the _parameter_properties classmethod of the parent class,
        # because otherwise TFP will raise a warning. Since the parameters are exactly
        # the same as for the parent class, this is appropriate.
        return super()._parameter_properties(dtype, num_classes)

    def log_prob(self, value):
        # standardized value for testing support
        # y <= 0 means the value is outside the support of the distribution
        y = 1 + self.concentration * ((value - self.loc) / self.scale)

        # Using the inner-outer jnp.where pattern to obtain working gradients
        # see here: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where # noqa
        # The values created here are only used to avoid NaNs;
        # they will be deselected by the final call to jnp.where.
        x_gumbel_safe = self.loc + 0.5 * self.scale
        x_gev_safe = self.loc + self.scale * (0.5 - 1.0) / self.concentration

        # safe_value = jnp.where(
        #     y <= 0,
        #     jnp.where(tf.equal(self.concentration, 0.0), x_gumbel_safe, x_gev_safe),
        #     value,
        # )

        safe_value = jnp.where(
            y <= 0,
            x_gev_safe,
            value,
        )

        # Explicitly fall back to Gumbel distribution for concentration == 0
        # I think the GEV class gives the wrong support in this case, see this issue:
        # https://github.com/tensorflow/probability/issues/1839
        # log_prob = jnp.where(
        #     tf.equal(self.concentration, 0.0),
        #     self.gumbel_dist.log_prob(safe_value),
        #     super().log_prob(safe_value),
        # )

        log_prob = super().log_prob(safe_value)

        # the jnp.max() is necessary to avoid dividing by zero in one branch of
        # the jnp.where.
        n_support_violations = jnp.maximum((y <= 0).sum(), 1.0)

        # returns a penalized log prob, with the penalty getting stronger for stronger
        # deviations from the needed support.
        # say, y = -1, then we have -1.5 * support penalty
        # say, y = -0.1, then we have -0.6 * support penalty
        # The second case has a smaller penalty, which is what we want.
        return jnp.where(
            y <= 0, (y - 0.5) * (self.support_penalty / n_support_violations), log_prob
        )

    def cdf(self, value):
        y = 1 + self.concentration * ((value - self.loc) / self.scale)

        x_gumbel_safe = 0.5 * self.scale + self.loc
        x_gev_safe = self.scale * (0.5 - 1.0) / self.concentration + self.loc

        # safe_value = jnp.where(
        #     y <= 0,
        #     jnp.where(tf.equal(self.concentration, 0.0), x_gumbel_safe, x_gev_safe),
        #     value,
        # )

        safe_value = jnp.where(
            y <= 0,
            x_gev_safe,
            value,
        )

        # cdf = jnp.where(
        #     tf.equal(self.concentration, 0.0),
        #     self.gumbel_dist.cdf(safe_value),
        #     super().cdf(safe_value),
        # )

        cdf = super().cdf(safe_value)

        n_support_violations = jnp.max(jnp.array([(y <= 0).sum(), 1.0]))

        return jnp.where(
            y <= 0, (y - 0.5) * (self.support_penalty / n_support_violations), cdf
        )
