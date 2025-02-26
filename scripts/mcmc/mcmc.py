import jax
import jax.numpy as jnp
import liesel.model as lsl
import liesel.goose as gs
from liesel.distributions.mvn_degen import MultivariateNormalDegenerate
import tensorflow_probability.substrates.jax.distributions as tfjd
import tensorflow_probability.substrates.jax.bijectors as tfjb

# Helper function to set up each model component (location, scale, shape)
def setup_component(name, K, beta0_prior_scale, lambda_prior_scale):
    # Beta0 parameter (Normal prior)
    beta0 = lsl.param(
        jnp.array([0.0]),
        distribution=lsl.Dist(tfjd.Normal, loc=0.0, scale=beta0_prior_scale),
        name=f"beta0_{name}"
    )
    
    # Lambda parameter (HalfCauchy prior)
    lambda_ = lsl.param(
        jnp.array([1.0]),
        distribution=lsl.Dist(tfjd.HalfCauchy, loc=0.0, scale=lambda_prior_scale),
        name=f"lambda_{name}"
    )
    
    # Gamma's penalty matrix and distribution (DegenerateNormal)
    penalty = lsl.obs(K, name=f"gamma_{name}_penalty")
    evals = jnp.linalg.eigvalsh(K)
    rank = lsl.Data(jnp.sum(evals > 0.0), _name=f"gamma_{name}_rank")
    log_pdet = lsl.Data(jnp.sum(jnp.log(jnp.where(evals > 0.0, evals, 1.0))), _name=f"gamma_{name}_log_pdet")
    
    gamma_dist = lsl.Dist(
        MultivariateNormalDegenerate.from_penalty,
        loc=0.0,
        var=lambda_,
        pen=penalty,
        rank=rank,
        log_pdet=log_pdet
    )
    gamma = lsl.param(
        jnp.zeros(K.shape[1]),
        distribution=gamma_dist,
        name=f"gamma_{name}"
    )
    
    return beta0, lambda_, gamma

# Assuming DesignMatrix, K_loc, K_scale, K_shape, and Y_SYN are predefined
# Setup components for location, scale, and shape
beta0_loc, lambda_loc, gamma_loc = setup_component("loc", K_loc, 10.0, 0.01)
beta0_scale, lambda_scale, gamma_scale = setup_component("scale", K_scale, 10.0, 0.01)
beta0_shape, lambda_shape, gamma_shape = setup_component("shape", K_shape, 10.0, 0.01)

# Design matrix (observed variable)
design_matrix = lsl.obs(DesignMatrix, name="design_matrix")

# Linear predictors for each parameter
linear_predictor_loc = lsl.Calc(
    lambda beta0, gamma, dm: beta0 + dm @ gamma,
    beta0_loc, gamma_loc, design_matrix,
    name="linear_predictor_loc"
)
linear_predictor_scale = lsl.Calc(
    lambda beta0, gamma, dm: beta0 + dm @ gamma,
    beta0_scale, gamma_scale, design_matrix,
    name="linear_predictor_scale"
)
linear_predictor_shape = lsl.Calc(
    lambda beta0, gamma, dm: beta0 + dm @ gamma,
    beta0_shape, gamma_shape, design_matrix,
    name="linear_predictor_shape"
)

# Apply exp transformation to scale predictor to ensure positivity
scale = lsl.Calc(jnp.exp, linear_predictor_scale, name="scale")

# GPD response distribution
y_dist = lsl.Dist(
    tfjd.GeneralizedPareto,
    loc=linear_predictor_loc,
    scale=scale,
    concentration=linear_predictor_shape
)
Y = lsl.obs(Y_SYN, distribution=y_dist, name="Y")

# Build the model graph
gb = lsl.GraphBuilder().add(Y)

# Apply log transformation to lambda parameters for NUTS sampling
gb.transform(lambda_loc, tfjb.Log())
gb.transform(lambda_scale, tfjb.Log())
gb.transform(lambda_shape, tfjb.Log())

model = gb.build_model()

# Configure MCMC with NUTS
builder = gs.EngineBuilder(seed=42, num_chains=4)
builder.set_model(gs.LieselInterface(model))
builder.set_initial_values(model.state)

# Parameters to sample (transformed lambdas and others)
position_keys = [
    "beta0_loc", "lambda_loc_transformed", "gamma_loc",
    "beta0_scale", "lambda_scale_transformed", "gamma_scale",
    "beta0_shape", "lambda_shape_transformed", "gamma_shape"
]

builder.add_kernel(gs.NUTSKernel(position_keys))
builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()

# Run MCMC sampling
engine.sample_all_epochs()

# Retrieve and inspect results
results = engine.get_results()
summary = gs.Summary(results)
print(summary)