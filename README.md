# VIGAMLSS: Variational Inference for Generalized Additive Models for Location, Scale and Shape

This repository was made for my Master's Thesis "Stochastic Variational Inference for Structured Additive Distributional Regression in Peak-over-Threshold Extreme Value Modeling".

The VIGAMLSS scripts contain the implementation of Stochastic Variational Inference (SVI) for Bayesian Structured Additive Distributional Regression (SADR) based on Generalized Additive Models for Location, Scale and Shape (GAMLSS). The notebook uses the VIGAMLSS scripts to compare SVI with the Markov Chain Monte Carlo (MCMC) No-U Turn Sampler (NUTS) for Generalized Pareto (GP) distributed responses and to have a look at the computational SVI challenges posed by the parameter-dependent support of the GP distribution. It also implements a Bayesian peak-over-threshold extreme value modeling approach utilizing the Asymmetric Laplace Distribution (ALD).

## Overview

The repository provides a Python implementation of SVI as detailed in the accompanying thesis using the JAX framework. The implementation is structured as a package named VIGAMLSS, which provides a modeling pipeline for:

- Design matrix construction for P-splines
- SADR GAMLSS Model specification
- SVI posterior inference

## Features

VIGAMLSS:
- Automatic Differentiation Variational Inference (ADVI) optimization algorithm with Full Covariance Multivariate Normal (FCMN) variational distribution
- Modified GP distribution for handling parameter-dependent support during SVI optimization
- Asymmetric Laplace (AL) distribution implementation for Two-stage Peaks-over-Threshold extreme value modeling
- Inverse Gamma, Normal, (Degenerate) Multivariate Normal, and Half-Cauchy prior implementation

Notebook:
- Comparative analysis with MCMC NUTS (using Liesel)
- Evaluation metrics including Wasserstein and Sinkhorn distances
- Simulation study with varying sample sizes
- Peak-over-Threshold Extreme Value Modeling Case study using Danish Fire Insurance dataset

## Core Dependencies

- jax==0.5.0 
- "jax[cuda12]"
- optax==0.2.4
- tensorflow-probability==0.24.0 
- tensorflow==2.18.0 
- liesel==0.3.3 
- ArviZ==0.18.0
- pot==0.9.5 
- ott-jax==0.5.0

## Implementation Details

The core of the implementation is the SVI algorithm including a SADR modeling pipeline:

```python
# Example usage
import jax.numpy as jnp
from optax import adam
from vigamlss import (
    DataPreparator,
    DegenerateNormal,
    IG,
    FCMNormal,
    ZeroCenteredGP,
    AL,
)

# Prepare design matrix
DesignMatrix, K, knots = DataPreparator(
    name="DesignMatrix",
    data=X,
    basis_transformation="pspline",
    intercept=False,
    standardize=False,
    degree=3,
    num_knots=20,
    use_quantile=False,
    return_knots=True,
)()

# Define model components
β0_scale = Normal("intercept_scale", jnp.array([0.0]), jnp.array([100.0]), size=1)
λ_scale = IG("lambda_smooth_scale", jnp.array([1.0]), jnp.array([0.005]), size=1)
γ_scale = DMN("spline_scale_coef", K, λ_scale)
β0_shape = Normal("intercept_shape", jnp.array([0.0]), jnp.array([10.0]), size=1)
λ_shape = IG("lambda_smooth_shape", jnp.array([1.0]), jnp.array([0.005]), size=1)
γ_shape = DMN("spline_shape_coef", K, λ_shape)

# Define response distribution
Y = ZeroCenteredGP(
    "y_GP",
    β0_scale + DesignMatrix @ γ_scale,
    β0_shape + DesignMatrix @ γ_shape,
    responses=Y_SYN,
)

# Run SVI optimization
Y.model.run_svi_optimization(
    optimizer=adam,
    vi_dist=FCMN,
    vi_sample_size=64,
    epochs=20000,
    mb_size=None,
    lr=0.001,
    max_norm=1.0,
    clip_min_max_enabled=True,
    zero_nans_enabled=True,
    prng_key=1,
    scheduler_type="constant",
)
```

## References

- Balkema, A. A., & de Haan, L. (1974). Residual life time at great age.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians.
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
- Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., ... & Saurous, R. A. (2017). Tensorflow distributions.
- Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties.
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes.
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference.
- Lang, S., & Brezger, A. (2004). Bayesian P-splines.
- Riebl, H., Wiemann, F. V. P., & Kneib, T. (2023). Liesel.
- Stasinopoulos, D. M., Kneib, T., Klein N., Mayr A., & Heller Z. G. (2024). Generalized additive models for location, scale and shape.
- Youngman, D. B. (2019). Generalized additive models for exceedances of high thresholds with an application to return level estimation for U.S. wind gusts.

