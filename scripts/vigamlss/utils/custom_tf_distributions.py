import jax.numpy as jnp
import jax.random as random
from tensorflow_probability.substrates import jax as tfp
import tensorflow as tf

tfd = tfp.distributions


class CustomGPD(tfd.GeneralizedPareto):
    """This code was given to me by my supervisor."""

    def __init__(
        self,
        loc,
        scale,
        shape,
        support_penalty: float = 1000,
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


class CustomGEV(tfd.GeneralizedExtremeValue):
    """This code was given to me by my supervisor."""

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


class CustomALD(tfd.Distribution):
    """Asymmetric Laplace Distribution (ALD) with a modified check function.

    This distribution is parameterized by:
      - loc: The location (or quantile) parameter, denoted by u.
      - scale: The positive scale parameter (scale > 0).
      - tau: The quantile level with (0 < tau < 1).

    This implementation supports broadcasting over the parameters.

    Notes:
        This implementation is based on the paper Benjamin D. Youngman -
        "Generalized Additive Models for Exceedances of High Thresholds
        With an Application to Return Level Estimation for U.S. Wind Gusts"
        (https://doi.org/10.1080/01621459.2018.1529596) where the ALD parameterization
        stems from Yu & Moyeed - "Bayesian quantile regression"
        (https://doi.org/10.1016/S0167-7152(01)00124-9) BUT WITHOUT the modified check
        function from, but with the unmodified one, see Oh, Lee, and Nychka -
        "Fast Nonparametric Quantile Regression With Arbitrary Smoothing Methods"
        (https://doi.org/10.1198/jcgs.2010.10063), BECAUSE no numerical instabilities
        have been observed using the unmodified one.
        The unmodified one for the three parameter ALD is shown in
        "A Three-Parameter Asymmetric Laplace
        Distribution and Its Extension" by Yu, and Zhang 
        (http://dx.doi.org/10.1080/03610920500199018)

        There is an error regarding the modified check fctn. first case in Youngman.
        See Yu & Moyeed for the correct first case condition.

        Methods were compared against:
        https://cran.r-project.org/web/packages/ald/index.html
    """

    def __init__(
        self,
        loc,
        scale,
        tau,
        validate_args=False,
        allow_nan_stats=True,
        name="AsymmetricLaplace",
    ):
        parameters = dict(loc=loc, scale=scale, tau=tau)
        with tf.name_scope(name) as name:
            # Determine a common dtype.
            dtype = jnp.result_type(loc, scale, tau)
            # Convert inputs to JAX arrays.
            self._loc = jnp.asarray(loc, dtype=dtype)
            self._scale = jnp.asarray(scale, dtype=dtype)
            self._tau = jnp.asarray(tau, dtype=dtype)

            # Determine the broadcast shape for the parameters.
            self._broadcast_shape = jnp.broadcast_shapes(
                jnp.shape(self._loc),
                jnp.shape(self._scale),
                jnp.shape(self._tau),
            )
            super().__init__(
                dtype=dtype,
                reparameterization_type=tfd.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name,
            )

    def _log_prob(self, x):
        """Log probability density function.

        For input `x`, the log density is computed as

            log f(x) = log(tau*(1-tau)) - log(scale) - rho_(tau)((x - loc)/scale),

        where the check function, rho_(tau)(z) with z=(x - loc)/scale, is defined piecewise as

        - if z<0:     (tau-1)*z,
        - if z>=0:    tau*z    
        """
        z = (x - self._loc) / self._scale
        # Modified Check-function
        # check_val = jnp.where(
        #     z < -self._c,
        #     (self._tau - 1.0) * (2.0 * z + self._c),
        #     jnp.where(
        #         z < 0.0,
        #         (1.0 - self._tau) * (z**2) / self._c,
        #         jnp.where(
        #             z < self._c,
        #             self._tau * (z**2) / self._c,
        #             self._tau * (2.0 * z - self._c),
        #         ),
        #     ),
        # )
        check_val = jnp.where(
            z < 0,
            (self._tau - 1),
            self._tau
        )
        log_normalizer = jnp.log(self._tau * (1.0 - self._tau)) - jnp.log(self._scale)
        return log_normalizer - z * check_val

    def _mean(self):
        """Mean of the ALD.
        """
        raise NotImplementedError("Mean is not implemented for AsymmetricLaplace.")

    def _variance(self):
        """Variance of the ALD.
        """
        raise NotImplementedError("Mean is not implemented for AsymmetricLaplace.")

    def _stddev(self):
        """Standard deviation of the ALD."""
        return jnp.sqrt(self._variance())

    def _cdf(self, x):
        raise NotImplementedError("CDF is not implemented for AsymmetricLaplace.")

    def _log_cdf(self, x):
        raise NotImplementedError("log CDF is not implemented for AsymmetricLaplace.")

    def _quantile(self, p, lower_tail=True):
        """Quantile function (inverse CDF) of the ALD taken 
        from the paper "A Three-Parameter Asymmetric Laplace
        Distribution and Its Extension" by Yu, and Zhang 
        (http://dx.doi.org/10.1080/03610920500199018)
        """
        prob = p if lower_tail else (1 - p)
        return jnp.where(
            prob <= self._tau,
            self._loc + (self._scale / (1 - self._tau)) * jnp.log(prob / self._tau),
            self._loc - (self._scale / self._tau) * jnp.log((1 - prob) / (1 - self._tau))
        )

    def _sample_n(self, n, seed):
        """This sample method is taken from the paper "A Three-Parameter Asymmetric Laplace
        Distribution and Its Extension" by Yu, and Zhang 
        (http://dx.doi.org/10.1080/03610920500199018)
        """
        key1, key2 = random.split(seed)
        batch_shape = self._broadcast_shape
        u = tfd.Exponential(rate=1).sample((n, *batch_shape), key1)
        v = tfd.Exponential(rate=1).sample((n, *batch_shape), key2)
        ald_samples_standard = (u / self._tau) - (v / (1.0 - self._tau))
        return self._loc + self._scale * ald_samples_standard

    def _batch_shape_tensor(self):
        return jnp.array(self._broadcast_shape, dtype=jnp.int32)

    def _batch_shape(self):
        return tf.TensorShape(self._broadcast_shape)

    def _event_shape_tensor(self):
        return jnp.array([], dtype=jnp.int32)

    def _event_shape(self):
        return tf.TensorShape([])
