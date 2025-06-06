# VIGAMLSS: Variational Inference for Generalized Additive Models for Location, Scale and Shape

This repository was made for my Master's Thesis "Stochastic Variational Inference for Structured Additive Distributional Regression in Peak-over-Threshold Extreme Value Modeling".

## Overview

The repository provides a Python implementation of SVI using automatic differentiation (Automatic Differentiation Variational Inference) as detailed in the accompanying thesis using the JAX framework. VIGAMLSS's probabilistic modeling of Bayesian Structured Additive Distributional Regression (SADR) models based on Generalized Additive Models for Location, Scale and Shape (GAMLSS) is based on the Distribution and Bijector classes of TensorFlow Probability. The implementation is structured as a package named VIGAMLSS, which provides a modeling pipeline for:

- Design matrix construction for P-splines
- SADR GAMLSS Model specification
- SVI posterior inference via gradient-based optimization using automatic differentiation gradient estimates of the ELBO

The notebook uses the VIGAMLSS scripts to compare SVI with the Markov Chain Monte Carlo (MCMC) No-U Turn Sampler (NUTS) for Generalized Pareto (GP) distributed responses and to have a look at the computational SVI challenges posed by the parameter-dependent support of the GP distribution. It also implements a Bayesian peak-over-threshold extreme value modeling approach utilizing the Asymmetric Laplace Distribution (ALD).

## Features

VIGAMLSS:
- Automatic Differentiation Variational Inference (ADVI) optimization algorithm with Full Covariance Multivariate Normal (FCMN) variational distribution
- Modified GP distribution for handling parameter-dependent support during SVI optimization
- Asymmetric Laplace (AL) distribution implementation for Two-stage Peaks-over-Threshold extreme value modeling
- Inverse Gamma, Normal, (Degenerate) Multivariate Normal, and Half-Cauchy prior implementation

Notebook:
- Comparative analysis with MCMC NUTS (using Liesel)
- Posterior approximation evaluation metrics including Wasserstein and Sinkhorn distances
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
    FCMN,
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

- Balkema, A. A., & de Haan, L. (1974). Residual life time at great age. The Annals of Probability, 2(5), 792–804. https://doi.org/10.1214/aop/1176996548
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. Journal of the American Statistical Association, 112(518), 859–877. https://doi.org/10.1080/01621459.2017.1285773
- Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs. Google. http://github.com/google/jax
- Callegher, G., Kneib, T., Söding, J., & Wiemann, P. (2025). Stochastic Variational Inference for Structured Additive Distributional Regression. arXiv. https://arxiv.org/abs/2412.10038
- Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer London. https://doi.org/10.1007/978-1-4471-3675-0
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport [Editors: C.J. Burges and L. Bottou and M. Welling and Z. Ghahramani and K.Q. Weinberger]. Advances in Neural Information Processing Systems, 26. https://proceedings.neurips.cc/paper_files/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
- DeepMind, Babuschkin, I., Baumli, K., Bell, A., Bhupatiraju, S., Bruce, J., Buchlovsky, P., Budden, D., Cai, T., Clark, A., Danihelka, I., Dedieu, A., Fantacci, C., Godwin, J., Jones, C., Hemsley, R., Hennigan, T., Hessel, M., Hou, S., . . . Viola, F. (2020). The DeepMind JAX Ecosystem. http://github.com/google-deepmind
- Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., Patton, B., Alemi, A., Hoffman, M., & Saurous, R. A. (2017). Tensorflow distributions. arXiv. http://arxiv.org/abs/1711.10604v1
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. Statistical Science,
- Fisher, R. A., & Tippett, L. H. C. (1928). Limiting forms of the frequency distribution of the largest or smallest member of a sample. Mathematical Proceedings of the Cambridge Philosophical Society, 24(2), 180–190. https://doi.org/10.1017/S0305004100015681
- Flamary, R., Courty, N., Gramfort, A., Alaya, M. Z., Boisbunon, A., Chambon, S., Chapel, L., Corenflos, A., Fatras, K., Fournier, N., Gautheron, L., Gayraud, N. T., Janati, H., Rakotomamonjy, A., Redko, I., Rolet, A., Schutz, A., Seguy, V., Sutherland, D. J., . . . Vayer, T. (2021). Pot: Python optimaltransport. Journal of Machine Learning Research, 22(78), 1–8. http://jmlr.org/papers/v22/20-451.html
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic variational inference. Journal of Machine Learning Research, 14(40), 1303–1347. http://jmlr.org/papers/v14/hoffman13a.html
- Hoffman, M. D., & Gelman, A. (2014). The no-u-turn sampler: Adaptively setting path lengths in hamiltonian monte carlo. J. Mach. Learn. Res., 15(1), 1593–1623.11(2), 89–121. https://doi.org/10.1214/ss/1038425655
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv. http://arxiv.org/abs/1312.6114
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2016). Automatic differentiation variational inference. https://arxiv.org/abs/1603.00788
- Lang, S., & Brezger, A. (2004). Bayesian p-splines. Journal of Computational and Graphical Statistics, 13(1),183–212. https://doi.org/10.1198/1061860043010
- Peyré, G., & Cuturi, M. (2020). Computational optimal transport. arXiv. http://arxiv.org/abs/1803.00567v4
- Pickands III, J. (1975). Statistical inference using extreme order statistics. The Annals of Statistics, 3(1), 119–131. https://doi.org/10.1214/aos/1176343002
- Riebl, H., Wiemann, P. F. V., & Kneib, T. (2023). Liesel: A probabilistic programming framework for developing semi-parametric regression models and custom bayesian inference algorithms. https://arxiv.org/abs/2209.10975
- Stasinopoulos, M. D., Kneib, T., Klein, N., Mayr, A., & Heller, G. Z. (2024). Generalized additive models for location, scale, and shape: A distributional regression approach, with applications. Cambridge University Press. https://doi.org/10.1017/9781009410076
- Youngman, B. D. (2019). Generalized Additive Models for Exceedances of High Thresholds With an Application to Return Level Estimation for U.S. Wind Gusts. Journal of the American Statistical Association, 114(528), 1865–1879. https://doi.org/10.1080/01621459.2018.1529596

