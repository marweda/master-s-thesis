"""Approach and code for Liesel Model Building with changes to model specifications taken from: https://github.com/Seb-Lorek/bbvi/tree/main"""

import os
import re
import sys
from typing import Dict, Optional
import warnings

import jax
from jax import numpy as jnp
from liesel.distributions.mvn_degen import MultivariateNormalDegenerate
import liesel.model as lsl
import liesel.goose as gs
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from tqdm.notebook import tqdm


sys.path.append(os.path.abspath(".."))
warnings.filterwarnings("ignore", category=FutureWarning)


class VarianceHC(lsl.Group):
    def __init__(self, name: str, scale: float, start_value: float = 1.0) -> None:
        # Create Var objects for hyperparameters following foreign code pattern
        scale_var = lsl.Var.new_value(scale, name=f"{name}_scale")

        u_init = jnp.arctan(start_value / scale)
        # Uniform prior on (0, pi/2)
        u_prior = lsl.Dist(tfd.Uniform, low=0.0, high=jnp.pi / 2)
        u = lsl.param(u_init, distribution=u_prior, name=f"{name}_u")

        # # Half-Cauchy prior setup (location=0.0 fixed)
        # prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=scale_var)
        # lambda_ = lsl.param(start_value, distribution=prior, name=name)

        # Transform to Half-Cauchy using tan function
        lambda_ = lsl.Var(
            lsl.Calc(lambda u, scale: scale * jnp.tan(u), u, scale_var), name=name
        )

        super().__init__(name=name, scale=scale_var, lambda_=lambda_)


class VarianceIG(lsl.Group):
    def __init__(self, name: str, a: float, b: float, start_value: float = 1.0) -> None:
        a_var = lsl.Var(a, name=f"{name}_a")
        b_var = lsl.Var(b, name=f"{name}_b")

        prior = lsl.Dist(tfd.InverseGamma, concentration=a_var, scale=b_var)
        lambda_ = lsl.param(start_value, distribution=prior, name=name)
        super().__init__(name=name, a=a_var, b=b_var, lambda_=lambda_)


class SplineIntercept(lsl.Group):
    def __init__(self, name: str, scale: float, start_value: float) -> None:
        # Create Var objects for hyperparameters following foreign code pattern
        scale_var = lsl.Var.new_value(scale, name=f"{name}_scale")

        prior = lsl.Dist(tfd.Normal, loc=0.0, scale=scale)
        beta0 = lsl.param(start_value, distribution=prior, name=name)

        super().__init__(name=name, scale=scale_var, beta0=beta0)


class SplineCoefNonCentered(lsl.Group):
    def __init__(
        self,
        name: str,
        penalty: jnp.ndarray,
        lambda_: lsl.param,
        start_value: jnp.ndarray,
        tol: float = 1e-6,
    ) -> None:
        def sqrt_pcov(precision, tol):
            eigenvalues, evecs = jnp.linalg.eigh(precision)

            sqrt_eval = jnp.sqrt(1 / eigenvalues)
            sqrt_eval = jnp.where(eigenvalues < tol, 0.0, sqrt_eval)

            event_shape = sqrt_eval.shape[-1]
            shape = sqrt_eval.shape + (event_shape,)

            r = tuple(range(event_shape))
            diags = jnp.zeros(shape).at[..., r, r].set(sqrt_eval)
            return evecs @ diags

        tol_var = lsl.Var.new_value(tol, name=f"{name}_tol")
        penalty_var = lsl.Var.new_value(penalty, name=f"{name}_penalty")
        precision_var = lsl.Var(
            lsl.Calc(jnp.divide, penalty_var, lambda_), name=f"{name}_precision"
        )
        sqrt_pcov_var = lsl.Var(
            lsl.Calc(sqrt_pcov, precision_var, tol_var),
            name=f"{name}_eigenvalues",
        )

        # Standard normal prior for z
        std_normal_prior = lsl.Dist(
            tfd.Normal,
            loc=jnp.zeros_like(start_value),
            scale=jnp.ones_like(start_value),
        )

        # Parameter z with standard normal prior
        z = lsl.param(
            jnp.zeros_like(start_value), distribution=std_normal_prior, name=f"{name}_z"
        )

        coef = lsl.Var(
            lsl.Calc(lambda sqrt_pcov, z: sqrt_pcov @ z, sqrt_pcov_var, z),
            name=f"{name}_centered_samples",
        )

        evals = jnp.linalg.eigvalsh(penalty)
        rank = lsl.Data(jnp.sum(evals > 0.0), _name=f"{name}_rank")

        super().__init__(
            name=name,
            coef=coef,
            z=z,
            penalty=penalty_var,
            lambda_=lambda_,
            tol=tol_var,
            precision=precision_var,
            sqrt_pcov_=sqrt_pcov_var,
            rank=rank,
        )


class SplineCoef(lsl.Group):
    def __init__(
        self,
        name: str,
        penalty: jnp.ndarray,
        lambda_: lsl.param,
        start_value: jnp.ndarray,
    ) -> None:
        penalty_var = lsl.Var.new_value(penalty, name=f"{name}_penalty")

        evals = jnp.linalg.eigvalsh(penalty)
        rank = lsl.Data(jnp.sum(evals > 0.0), _name=f"{name}_rank")
        log_pdet = lsl.Data(
            jnp.sum(jnp.log(jnp.where(evals > 0.0, evals, 1.0))),
            _name=f"{name}_log_pdet",
        )

        prior = lsl.Dist(
            MultivariateNormalDegenerate.from_penalty,
            loc=0.0,
            var=lambda_,
            pen=penalty_var,
            rank=rank,
            log_pdet=log_pdet,
        )
        coef = lsl.param(start_value, distribution=prior, name=name)

        super().__init__(
            name, coef=coef, penalty=penalty_var, lambda_=lambda_, rank=rank
        )


class PSpline(lsl.Group):
    def __init__(
        self,
        name: str,
        basis_matrix: jnp.ndarray,
        penalty: jnp.ndarray,
        non_centered: bool,
        lambda_group: lsl.Group,
        start_value: jnp.ndarray,
    ) -> None:
        if non_centered:
            coef_group = SplineCoefNonCentered(
                name=f"{name}_coef",
                penalty=penalty,
                lambda_=lambda_group["lambda_"],
                start_value=start_value,
            )
        else:
            coef_group = SplineCoef(
                name=f"{name}_coef",
                penalty=penalty,
                lambda_=lambda_group["lambda_"],
                start_value=start_value,
            )

        basis_matrix = lsl.obs(basis_matrix, name=f"{name}_basis")
        smooth = lsl.Var(
            lsl.Calc(jnp.dot, basis_matrix, coef_group["coef"]), name=f"{name}_smooth"
        )

        group_vars = coef_group.nodes_and_vars | lambda_group.nodes_and_vars

        super().__init__(
            name=name,
            basis_matrix=basis_matrix,
            smooth=smooth,
            **group_vars,
        )


def tau2_gibbs_kernel(p_spline: PSpline) -> gs.GibbsKernel:
    """Builds a Gibbs kernel for a smoothing parameter with an inverse gamma prior."""
    position_key = p_spline["lambda_"].name

    def transition(prng_key, model_state):
        a_prior = p_spline.value_from(model_state, "a")
        b_prior = p_spline.value_from(model_state, "b")

        rank = p_spline.value_from(model_state, "rank")
        K = p_spline.value_from(model_state, "penalty")

        beta = p_spline.value_from(model_state, "coef")

        a_gibbs = jnp.squeeze(a_prior + 0.5 * rank)
        b_gibbs = jnp.squeeze(b_prior + 0.5 * (beta @ K @ beta))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {position_key: draw}

    return gs.GibbsKernel([position_key], transition)


def build_model(
    y: jnp.ndarray,
    basis: jnp.ndarray,
    penalty: jnp.ndarray,
    non_centered: bool,
    pspline_scale_coef_init: jnp.ndarray,
    pspline_shape_coef_init: jnp.ndarray,
    pspline_scale_inter_init: float,
    pspline_shape_inter_init: float,
    pspline_scale_lambda_init: float,
    pspline_shape_lambda_init: float,
    pspline_scale_hyperprior: float,
    pspline_shape_hyperprior: float,
    lambda_scale_hyperprior: float,
    lambda_shape_hyperprior: float,
    use_half_cauchy: bool = False,
) -> lsl.Model:
    # Create variance components based on chosen prior type
    if use_half_cauchy:
        # Use Half-Cauchy prior
        lambda_group_1 = VarianceHC(
            "lambda_smooth_scale",
            scale=lambda_scale_hyperprior,
            start_value=pspline_scale_lambda_init,
        )
        lambda_group_2 = VarianceHC(
            "lambda_smooth_shape",
            scale=lambda_shape_hyperprior,
            start_value=pspline_shape_lambda_init,
        )
    else:
        # Use Inverse Gamma prior
        lambda_group_1 = VarianceIG(
            name="lambda_smooth_scale",
            a=1.0,
            b=lambda_scale_hyperprior,
            start_value=pspline_scale_lambda_init,
        )
        lambda_group_2 = VarianceIG(
            name="lambda_smooth_shape",
            a=1.0,
            b=lambda_shape_hyperprior,
            start_value=pspline_shape_lambda_init,
        )

    # Create PSpline components
    pspline_scale = PSpline(
        "spline_scale",
        basis,
        penalty,
        non_centered,
        lambda_group_1,
        pspline_scale_coef_init,
    )
    pspline_shape = PSpline(
        "spline_shape",
        basis,
        penalty,
        non_centered,
        lambda_group_2,
        pspline_shape_coef_init,
    )

    # Create Intercepts
    pspline_intercept_scale = SplineIntercept(
        "intercept_scale", pspline_scale_hyperprior, pspline_scale_inter_init
    )
    pspline_intercept_shape = SplineIntercept(
        "intercept_shape",
        pspline_shape_hyperprior,
        pspline_shape_inter_init,
    )

    # Linear predictor setup
    linear_pred_scale = lsl.Var(
        lsl.Calc(
            jnp.add,
            pspline_intercept_scale["beta0"],
            pspline_scale["smooth"],
        ),
        name="linear_pred_scale",
    ).transform(tfb.Softplus())
    linear_pred_shape = lsl.Var(
        lsl.Calc(
            jnp.add,
            pspline_intercept_shape["beta0"],
            pspline_shape["smooth"],
        ),
        name="linear_pred_shape",
    )

    # Generalized Pareto response (loc=0.0 fixed)
    y_dist = lsl.Dist(
        tfd.GeneralizedPareto,
        # CustomGPD,
        loc=0.0,
        scale=linear_pred_scale,
        concentration=linear_pred_shape,
        # shape=linear_pred_shape,
    )
    y_var = lsl.Var(y, distribution=y_dist, name="response")
    gb = lsl.GraphBuilder().add(y_var)

    return gb.build_model(), pspline_scale, pspline_shape


# Save/load with proper path handling
def save_results_mcmc(results: gs.SamplingResults, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.pkl_save(path)


def load_results_mcmc(dir: str, file_name: str) -> gs.SamplingResults:
    full_path = os.path.join(
        dir,
        file_name if file_name.endswith(".pkl") else f"{file_name}.pkl",
    )
    return gs.SamplingResults.pkl_load(full_path)


def run_gpd_mcmc(
    y: jnp.ndarray,
    basis: jnp.ndarray,
    penalty: jnp.ndarray,
    non_centered: bool,
    pspline_scale_coef_init: jnp.ndarray,
    pspline_shape_coef_init: jnp.ndarray,
    pspline_scale_inter_init: float,
    pspline_shape_inter_init: float,
    pspline_scale_lambda_init: float,
    pspline_shape_lambda_init: float,
    pspline_scale_hyperprior: float,
    pspline_shape_hyperprior: float,
    lambda_scale_hyperprior: float,
    lambda_shape_hyperprior: float,
    target_accept: float = 0.8,
    use_half_cauchy: bool = False,
    num_chains: int = 4,
    warmup: int = 1000,
    draws: int = 1000,
    term_duration: int = 50,
    seed: int = 42,
    save_dir: str = None,
    file_name: str = None,
    do_run: bool = False,
    show_progress: bool = True,
) -> None:
    if do_run:
        if not (file_name or save_dir):
            raise ValueError(
                "Both file_name and save_dir must be provided to save the MCMC results."
            )
        else:
            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(
                save_dir,
                file_name if file_name.endswith(".pkl") else f"{file_name}.pkl",
            )

        model, pspline_scale, pspline_shape = build_model(
            y,
            basis,
            penalty,
            non_centered,
            pspline_scale_coef_init,
            pspline_shape_coef_init,
            pspline_scale_inter_init,
            pspline_shape_inter_init,
            pspline_scale_lambda_init,
            pspline_shape_lambda_init,
            pspline_scale_hyperprior,
            pspline_shape_hyperprior,
            lambda_scale_hyperprior,
            lambda_shape_hyperprior,
            use_half_cauchy,
        )

        # Engine setup
        builder = gs.EngineBuilder(seed=seed, num_chains=num_chains)
        builder.set_model(gs.LieselInterface(model))
        builder.set_initial_values(model.state)

        # Configure appropriate kernels based on prior choice and parameterization

        # 1. Set up hyperparameter kernels based on prior choice
        if use_half_cauchy:
            # For Half-Cauchy prior with parameter u
            builder.add_kernel(
                gs.NUTSKernel(
                    ["lambda_smooth_scale_u"],
                    da_target_accept=target_accept,
                ),
            )
            builder.add_kernel(
                gs.NUTSKernel(
                    ["lambda_smooth_shape_u"],
                    da_target_accept=target_accept,
                ),
            )
        else:
            # For Inverse Gamma prior with Gibbs sampler
            builder.add_kernel(tau2_gibbs_kernel(pspline_scale))
            builder.add_kernel(tau2_gibbs_kernel(pspline_shape))

        # 2. Set up coeficient kernels based on centering parameterization
        if non_centered:
            # For non-centered parameterization, sample the standard normal parameters z
            builder.add_kernel(
                gs.NUTSKernel(
                    ["spline_scale_coef_z"],
                    da_target_accept=target_accept,
                    max_treedepth=20,
                    initial_step_size=0.001,
                )
            )
            builder.add_kernel(
                gs.NUTSKernel(
                    ["spline_shape_coef_z"],
                    da_target_accept=target_accept,
                    max_treedepth=20,
                    initial_step_size=0.001,
                )
            )
        else:
            # For centered parameterization, sample coeficients directly
            builder.add_kernel(
                gs.NUTSKernel(
                    ["spline_scale_coef"],
                    da_target_accept=target_accept,
                    max_treedepth=20,
                    initial_step_size=0.001,
                )
            )
            builder.add_kernel(
                gs.NUTSKernel(
                    ["spline_shape_coef"],
                    da_target_accept=target_accept,
                    max_treedepth=20,
                    initial_step_size=0.001,
                )
            )

        # 3. kernels for intercepts
        builder.add_kernel(
            gs.NUTSKernel(
                ["intercept_scale"],
                da_target_accept=target_accept,
                max_treedepth=20,
                initial_step_size=0.001,
            )
        )
        builder.add_kernel(
            gs.NUTSKernel(
                ["intercept_shape"],
                da_target_accept=target_accept,
                max_treedepth=20,
                initial_step_size=0.001,
            )
        )

        # Sampling duration
        builder.set_duration(
            warmup_duration=warmup,
            posterior_duration=draws,
            term_duration=term_duration,
        )

        # Run inference
        engine = builder.build()
        engine._show_progress = show_progress
        engine.sample_all_epochs()
        results = engine.get_results()
        results.pkl_save(full_path)
    else:
        warnings.warn(
            "No MCMC inference. Flag do_run=True for running MCMC inference.",
            category=UserWarning,
        )


def reshape_posterior_samples(
    posterior_samples: Dict[str, jnp.ndarray], num_samples: int = None
) -> jnp.ndarray:
    param_arrays = []

    # Process parameters in sorted order for consistent concatenation
    for param_name in sorted(posterior_samples.keys()):
        param_data = posterior_samples[param_name]

        # Apply slicing if num_samples is specified
        if num_samples is not None:
            if param_data.ndim >= 2:  # For parameters with at least chains and samples
                param_data = param_data[:, :num_samples, ...]  # Slice samples dimension

        # Reshape based on parameter type
        if param_data.ndim == 2:  # Scalar parameters (chains, samples)
            reshaped = param_data.reshape(-1, 1)
        elif param_data.ndim == 3:  # Vector parameters (chains, samples, coeficients)
            reshaped = param_data.reshape(-1, param_data.shape[-1])
        else:
            raise ValueError(
                f"Unexpected shape {param_data.shape} for parameter {param_name}"
            )

        param_arrays.append(reshaped)

    # Concatenate all parameters across columns
    return jnp.concatenate(param_arrays, axis=1)


# def reshape_posterior_samples(
#     posterior_samples: Dict[str, jnp.ndarray], num_samples: int
# ) -> Dict[str, jnp.ndarray]:
#     """Process posterior samples by trimming and flattening chains."""
#     processed = {}
#     for param_name, param_data in posterior_samples.items():
#         # Trim samples if needed
#         if num_samples is not None and param_data.ndim >= 2:
#             param_data = param_data[:, :num_samples]

#         # Flatten chains and samples
#         n_chains = param_data.shape[0]
#         n_samples = param_data.shape[1]
#         processed[param_name] = jax.device_put(
#             param_data.reshape(n_chains * n_samples, -1), jax.devices("cpu")[0]
#         )
#     return processed


def thin_samples(
    samples: jnp.ndarray, thinning_degree: Optional[int] = None
) -> jnp.ndarray:
    """Apply thinning to samples by taking every thinning_degree-th sample."""
    if thinning_degree is None or thinning_degree <= 1:
        return samples

    # For 2D arrays (e.g., (4, 32000))
    elif samples.ndim == 2:
        return samples[:, ::thinning_degree]

    # For 3D arrays (e.g., (4, 32000, 5))
    elif samples.ndim == 3:
        return samples[:, ::thinning_degree, :]

    # Handle unexpected dimensions
    else:
        raise ValueError(f"Unexpected sample shape: {samples.shape}")


def sim_study_mcmc_loading_loop(
    save_dir: str,
    do_run: bool = False,
    device: str = "cpu",
    get_sampling_results_only: bool = False,
    thinning_degree: Optional[int] = None,
) -> Dict:
    if do_run:
        target_device = jax.devices(device)[0]
        results_dict = {}
        pattern = re.compile(r".*mcmc_loop_N(\d+)_run(\d+)\.pkl$")
        for root, dirs, files in os.walk(os.path.abspath(save_dir)):
            for filename in tqdm(files):
                match = pattern.match(filename)
                if not match:
                    continue
                # Extract metadata from filename
                n_value = int(match.group(1))
                run_number = int(match.group(2))
                file_base = filename[:-4]  # Remove .pkl
                n_key = f"N={n_value}"
                # Load results
                sampling_result = load_results_mcmc(root, file_base)
                # Initialize N-key structure if needed
                if n_key not in results_dict:
                    results_dict[n_key] = {"samples": []}
                # Process samples or store raw results
                if get_sampling_results_only:
                    results_dict[n_key]["samples"].append(sampling_result)
                else:
                    posterior_samples = jax.device_put(
                        sampling_result.get_posterior_samples(), device=target_device
                    )

                    # Apply thinning to each JAX array in the posterior_samples
                    thinned_posterior_samples = {}
                    for sample_key, sample_value in posterior_samples.items():
                        if isinstance(sample_value, list):
                            # Handle the case where we have a list of sample dictionaries
                            thinned_samples_list = []
                            for sample_dict in sample_value:
                                thinned_sample_dict = {}
                                for param_name, param_array in sample_dict.items():
                                    # Apply thinning to the JAX array
                                    thinned_sample_dict[param_name] = thin_samples(
                                        param_array, thinning_degree
                                    )
                                thinned_samples_list.append(thinned_sample_dict)
                            thinned_posterior_samples[sample_key] = thinned_samples_list
                        elif isinstance(sample_value, jnp.ndarray):
                            # Directly apply thinning to JAX arrays
                            thinned_posterior_samples[sample_key] = thin_samples(
                                sample_value, thinning_degree
                            )
                        else:
                            # For any other type, keep as is
                            thinned_posterior_samples[sample_key] = sample_value

                    results_dict[n_key]["samples"].append(thinned_posterior_samples)
        return results_dict
    else:
        warnings.warn("MCMC loading not executed", UserWarning)
        return {}
