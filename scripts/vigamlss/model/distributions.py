from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
from jax import vmap
from jax.lax import fori_loop
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.bijectors import Softplus

from .linear_predictor import LinearPredictor
from .node import Node
from .model_builder import ModelDAG
from ..utils.custom_tf_distributions import CustomGPD, CustomGEV, CustomALD
from ..utils.transformations import TransformationFunctions


tfd = tfp.distributions


@dataclass
class ParameterType:
    is_linearpredictor: bool = False
    is_randomvariable: bool = False
    is_array: bool = False


@dataclass
class DistributionState:
    is_likelihood: bool = False
    is_inner_prior: bool = False
    is_leaf_prior: bool = False


class Distribution:
    def __init__(
        self,
        rv_name: str,
        parameters: Dict[str, Any],
        size: Optional[int] = None,
        responses: Optional[jnp.ndarray] = None,
    ):
        self.rv_name = rv_name
        self.parameters = parameters
        self.size = size
        self.responses = responses
        self.num_distribution_parameters = len(self.parameters)
        self.node_type = "Distribution"


class NormalDistributionValidator:
    """Handles parameter validation and state management specifically for Normal distributions."""

    def __init__(self, distribution):
        self.distribution = distribution
        self.parameter_types = {}
        self.distribution_state = None

    def validate_and_setup(self):
        """Main entry point for validation and state setup."""
        self.parameter_types = {
            name: self._set_parameter_type(param)
            for name, param in self.distribution.parameters.items()
        }
        self._validate_parameters()
        self.distribution_state = self._set_distribution_state()
        self._validate_likelihood_related_parameters()
        return self.distribution_state

    @staticmethod
    def _set_parameter_type(param) -> ParameterType:
        """Determine the type of a parameter."""
        parameter_type = ParameterType()
        if hasattr(param, "node_type"):
            if param.node_type == "LinearPredictor":
                parameter_type.is_linearpredictor = True
                return parameter_type
            elif param.node_type == "Distribution":
                parameter_type.is_randomvariable = True
                return parameter_type
        elif isinstance(param, jnp.ndarray):
            parameter_type.is_array = True
            return parameter_type
        raise ValueError(f"Invalid parameter type: {type(param)}")

    def _validate_parameters(self) -> None:
        """Validate all parameters and their relationships."""
        param_types = list(self.parameter_types.values())

        # Validate GAMLSS requirements
        if any(pt.is_linearpredictor for pt in param_types) and not all(
            pt.is_linearpredictor for pt in param_types
        ):
            raise ValueError(
                "GAMLSS requires all likelihood parameters to be parameterized by linear predictors"
            )

        # Validate prior requirements
        has_rv = any(pt.is_randomvariable for pt in param_types)
        all_rv = all(pt.is_randomvariable for pt in param_types)
        if has_rv and not all_rv:
            raise NotImplementedError(
                "For simplicity, only priors are supported whose parameters are either all random variables or none."
            )

        # Validate size requirements
        if not any(pt.is_linearpredictor for pt in param_types):
            for name, param in self.distribution.parameters.items():
                if hasattr(param, "size") and param.size != self.distribution.size:
                    raise ValueError(
                        f"All parameters must have the same size as the size parameter"
                    )

    def _set_distribution_state(self) -> DistributionState:
        """Setup distribution state based on parameter types."""
        distribution_state = DistributionState()
        parameter_types = self.parameter_types.values()

        all_linear = all(pt.is_linearpredictor for pt in parameter_types)
        all_rv = all(pt.is_randomvariable for pt in parameter_types)
        all_leaf = all(pt.is_array for pt in parameter_types)

        if all_linear:
            distribution_state.is_likelihood = True
        elif all_rv:
            distribution_state.is_inner_prior = True
        elif all_leaf:
            distribution_state.is_leaf_prior = True

        return distribution_state

    def _validate_likelihood_related_parameters(self):
        """Validate likelihood-related parameters."""
        if (
            self.distribution_state.is_likelihood
            and self.distribution.responses is None
        ):
            raise ValueError("Responses must be provided for likelihood parameters")

        if (
            not self.distribution_state.is_likelihood
            and self.distribution.responses is not None
        ):
            raise ValueError(
                "Responses should not be provided for hyperprior parameters"
            )

        if self.distribution_state.is_likelihood and self.distribution.size is not None:
            raise ValueError(
                "Size should not be provided for a likelihood distribution. The size is implicitly determined by the num of responses."
            )

    @staticmethod
    def validate_addition(summand_l, summand_r):
        """Validate addition operation between distributions."""
        summand_rclass_type = summand_r.node_type
        if summand_rclass_type == "Distribution":
            if summand_l.size != 1 or summand_l.dim != 1:
                raise ValueError(
                    f"Invalid dimension for addition with a Distribution. Distribution has to have size 1."
                )
        if summand_rclass_type == "DesignMatrix":
            raise ValueError("Invalid right summand for addition with a Distribution.")

    @staticmethod
    def validate_multiplication(factor_r):
        """Validate matrix multiplication operation."""
        factor_rclass_type = factor_r.node_type
        if factor_rclass_type != "DesignMatrix":
            raise ValueError(
                "Matmul is only supported between Distribution and DesignMatrix instances."
            )


class Normal(Distribution):
    """Normal distribution implementation."""

    def __init__(
        self,
        rv_name: str,
        location: Union[jnp.ndarray, Distribution, LinearPredictor],
        scale: Union[jnp.ndarray, Distribution, LinearPredictor],
        size: int = None,
        responses: Optional[jnp.ndarray] = None,
    ):
        super().__init__(
            rv_name, {"location": location, "scale": scale}, size, responses
        )

        # Normal distribution specific
        self.realization_transformation = [TransformationFunctions.identity]
        self.parameter_transforms = [
            TransformationFunctions.identity,
            TransformationFunctions.softplus,
        ]

        # Validate and setup distribution state using the validator
        self.validator = NormalDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI related
        self.node = self.setup_node()

        # Model building related
        if self._distribution_state.is_likelihood:
            self.model = ModelDAG(self.responses, self.node)

    def setup_node(self) -> None:
        if self._distribution_state.is_likelihood:
            return self._setup_for_gamlss()
        elif self._distribution_state.is_inner_prior:
            return self._setup_for_inner_prior()
        elif self._distribution_state.is_leaf_prior:
            return self._setup_for_leaf_prior()

    def _setup_for_gamlss(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(self._compute_gamlss_log_pdf, in_axes=(None, None, 0, 0)),
            local_dim=self.size,
            transformations=self.parameter_transforms,
            parents=[self.parameters["location"].node, self.parameters["scale"].node],
        )

    def _setup_for_inner_prior(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(self._compute_prior_log_pdf, in_axes=(0, 0, 0)),
            local_dim=self.size,
            transformations=self.realization_transformation,
            parents=[self.parameters["location"].node, self.parameters["scale"].node],
        )

    def _setup_for_leaf_prior(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(
                partial(
                    self._compute_prior_log_pdf,
                    loc=self.parameters["location"],
                    scale=self.parameters["scale"],
                ),
                in_axes=(0,),
            ),
            local_dim=self.size,
            transformations=self.realization_transformation,
        )

    @staticmethod
    def _compute_gamlss_log_pdf(
        realizations: jnp.ndarray,
        mask: jnp.ndarray,
        loc: jnp.ndarray,
        scale: jnp.ndarray,
    ) -> jnp.ndarray:
        """Original GAMLSS log PDF computation."""
        log_pdf = tfd.Normal(loc, scale).log_prob(realizations)
        return jnp.sum(log_pdf * mask)

    @staticmethod
    def _compute_prior_log_pdf(
        realizations: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray
    ) -> jnp.ndarray:
        """Not curried leaf log PDF computation."""
        log_pdf = tfd.MultivariateNormalDiag(loc, scale).log_prob(realizations)
        return log_pdf

    def __add__(self, other):
        self.validator.validate_addition(self, other)
        return LinearPredictor(self, other, operation="+")

    def __rmatmul__(self, other):
        self.validator.validate_multiplication(other)
        return LinearPredictor(other, self, operation="@")


class GammaDistributionValidator:
    """Handles parameter validation and state management specifically for Gamma distributions."""

    def __init__(self, distribution):
        self.distribution = distribution
        self.parameter_types = {}
        self.distribution_state = None

    def validate_and_setup(self):
        """Main entry point for validation and state setup."""
        self.parameter_types = {
            name: self._set_parameter_type(param)
            for name, param in self.distribution.parameters.items()
        }
        self._validate_parameters()
        self.distribution_state = self._set_distribution_state()
        self._validate_likelihood_related_parameters()
        return self.distribution_state

    @staticmethod
    def _set_parameter_type(param) -> ParameterType:
        """Determine the type of a parameter."""
        parameter_type = ParameterType()
        if hasattr(param, "node_type"):
            if param.node_type == "LinearPredictor":
                parameter_type.is_linearpredictor = True
                return parameter_type
            elif param.node_type == "Distribution":
                parameter_type.is_randomvariable = True
                return parameter_type
        elif isinstance(param, jnp.ndarray):
            parameter_type.is_array = True
            return parameter_type
        raise ValueError(f"Invalid parameter type: {type(param)}")

    def _validate_parameters(self) -> None:
        """Validate all parameters and their relationships."""
        param_types = list(self.parameter_types.values())

        # Validate GAMLSS requirements
        if any(pt.is_linearpredictor for pt in param_types) and not all(
            pt.is_linearpredictor for pt in param_types
        ):
            raise ValueError(
                "GAMLSS requires all likelihood parameters to be parameterized by linear predictors"
            )

        # Validate prior requirements
        has_rv = any(pt.is_randomvariable for pt in param_types)
        all_rv = all(pt.is_randomvariable for pt in param_types)
        if has_rv and not all_rv:
            raise NotImplementedError(
                "For simplicity, only priors are supported whose parameters are either all random variables or none."
            )

        # Validate size requirements
        if not any(pt.is_linearpredictor for pt in param_types):
            for name, param in self.distribution.parameters.items():
                if hasattr(param, "size") and param.size != self.distribution.size:
                    raise ValueError(
                        f"All parameters must have the same size as the size parameter"
                    )

    def _set_distribution_state(self) -> DistributionState:
        """Setup distribution state based on parameter types."""
        distribution_state = DistributionState()
        parameter_types = self.parameter_types.values()

        all_linear = all(pt.is_linearpredictor for pt in parameter_types)
        all_rv = all(pt.is_randomvariable for pt in parameter_types)
        all_leaf = all(pt.is_array for pt in parameter_types)

        if all_linear:
            distribution_state.is_likelihood = True
        elif all_rv:
            distribution_state.is_inner_prior = True
        elif all_leaf:
            distribution_state.is_leaf_prior = True

        return distribution_state

    def _validate_likelihood_related_parameters(self):
        """Validate likelihood-related parameters."""
        if (
            self.distribution_state.is_likelihood
            and self.distribution.responses is None
        ):
            raise ValueError("Responses must be provided for likelihood parameters")

        if (
            not self.distribution_state.is_likelihood
            and self.distribution.responses is not None
        ):
            raise ValueError(
                "Responses should not be provided for hyperprior parameters"
            )

        if self.distribution_state.is_likelihood and self.distribution.size is not None:
            raise ValueError(
                "Size should not be provided for a likelihood distribution. The size is implicitly determined by the num of responses."
            )


class Gamma(Distribution):
    """Gamma distribution implementation."""

    def __init__(
        self,
        rv_name: str,
        concentration: Union[jnp.ndarray, Distribution, LinearPredictor],
        rate: Union[jnp.ndarray, Distribution, LinearPredictor],
        size: int = None,
        responses: Optional[jnp.ndarray] = None,
    ):
        super().__init__(
            rv_name, {"concentration": concentration, "rate": rate}, size, responses
        )

        # Gamma distribution specific
        self.realization_transformation = [TransformationFunctions.softplus]
        self.parameter_transforms = [
            TransformationFunctions.softplus,
            TransformationFunctions.softplus,
        ]

        # Validate and setup distribution state using the validator
        self.validator = GammaDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI related
        self.node = self.setup_node()

        # Model building related
        if self._distribution_state.is_likelihood:
            self.model = ModelDAG(self.responses, self.node)

    def setup_node(self) -> None:
        if self._distribution_state.is_likelihood:
            return self._setup_for_gamlss()
        elif self._distribution_state.is_inner_prior:
            return self._setup_for_inner_prior()
        elif self._distribution_state.is_leaf_prior:
            return self._setup_for_leaf_prior()

    def _setup_for_gamlss(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(self._compute_gamlss_log_pdf, in_axes=(None, None, 0, 0)),
            local_dim=self.size,
            transformations=self.parameter_transforms,
            parents=[
                self.parameters["concentration"].node,
                self.parameters["rate"].node,
            ],
        )

    def _setup_for_inner_prior(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(self._compute_prior_log_pdf, in_axes=(0, 0, 0)),
            local_dim=self.size,
            transformations=self.realization_transformation,
            parents=[
                self.parameters["concentration"].node,
                self.parameters["rate"].node,
            ],
        )

    def _setup_for_leaf_prior(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(
                partial(
                    self._compute_prior_log_pdf,
                    concentration=self.parameters["concentration"],
                    rate=self.parameters["rate"],
                ),
                in_axes=(0,),
            ),
            local_dim=self.size,
            transformations=self.realization_transformation,
        )

    @staticmethod
    def _compute_gamlss_log_pdf(
        realizations: jnp.ndarray,
        mask: jnp.ndarray,
        concentration: jnp.ndarray,
        rate: jnp.ndarray,
    ) -> jnp.ndarray:
        """Original GAMLSS log PDF computation."""
        log_pdf = tfd.Gamma(concentration, rate).log_prob(realizations)
        return jnp.sum(log_pdf * mask)

    @staticmethod
    def _compute_prior_log_pdf(
        realizations: jnp.ndarray, concentration: jnp.ndarray, rate: jnp.ndarray
    ) -> jnp.ndarray:
        """Not curried leaf log PDF computation."""
        log_pdf = tfd.Gamma(concentration, rate).log_prob(realizations)
        #  -inverse_log_det_jacobian=forward_log_det_jacobian
        log_det_jacobian = -Softplus().forward_log_det_jacobian(realizations)
        return log_pdf + log_det_jacobian

    def __add__(self, other):
        raise NotImplementedError("Addition is not supported for Gamma.")

    def __rmatmul__(self, other):
        raise NotImplementedError("Matrix multiplication is not supported for Gamma.")


class DegenerateNormalDistributionValidator:
    """Handles parameter validation and state management specifically for DegenerateNormal distributions."""

    def __init__(self, distribution):
        self.distribution = distribution
        self.parameter_types = {}
        self.distribution_state = None

    def validate_and_setup(self):
        """Main entry point for validation and state setup."""
        self.parameter_types = {
            name: self._set_parameter_type(param)
            for name, param in self.distribution.parameters.items()
        }
        self._validate_parameters()
        self.distribution_state = self._set_distribution_state()
        self._validate_likelihood_related_parameters()
        return self.distribution_state

    @staticmethod
    def _set_parameter_type(param) -> ParameterType:
        """Determine the type of a parameter."""
        parameter_type = ParameterType()
        if hasattr(param, "node_type"):
            if param.node_type == "LinearPredictor":
                raise NotImplementedError(
                    "LinearPredictor is not supported for tau2 parameter in DegenerateNormal."
                )
            elif param.node_type == "Distribution":
                parameter_type.is_randomvariable = True
                return parameter_type
        elif isinstance(param, jnp.ndarray):
            parameter_type.is_array = True
            return parameter_type
        raise ValueError(f"Invalid parameter type: {type(param)}")

    def _validate_parameters(self) -> None:
        """Validate all parameters and their relationships."""
        # Check tau2 size requirement
        tau2 = self.distribution.parameters["tau2"]
        if hasattr(tau2, "size") and tau2.size != 1:
            raise ValueError("tau2 parameter must have size 1")

    def _set_distribution_state(self) -> DistributionState:
        """Setup distribution state based on parameter types."""
        distribution_state = DistributionState()
        parameter_types = self.parameter_types.values()

        all_rv = all(pt.is_randomvariable for pt in parameter_types)
        all_leaf = all(pt.is_array for pt in parameter_types)

        if all_rv:
            distribution_state.is_inner_prior = True
        elif all_leaf:
            distribution_state.is_leaf_prior = True

        return distribution_state

    def _validate_likelihood_related_parameters(self):
        """Validate likelihood-related parameters."""
        if self.distribution_state.is_likelihood:
            raise NotImplementedError(
                "DegenerateNormal cannot be used as a likelihood."
            )

    @staticmethod
    def validate_multiplication(factor_r):
        """Validate matrix multiplication operation."""
        factor_rclass_type = factor_r.node_type
        if factor_rclass_type != "DesignMatrix":
            raise ValueError(
                "Matmul is only supported between Distribution and DesignMatrix instances."
            )


class DegenerateNormal(Distribution):
    """Degenerate Normal distribution with penalty matrix."""

    def __init__(
        self,
        rv_name: str,
        penalty_matrix: jnp.ndarray,
        tau2: Union[jnp.ndarray, Distribution],
    ):
        super().__init__(rv_name, {"tau2": tau2}, penalty_matrix.shape[0])
        # Degenerate Normal specific
        self.penalty_matrix = penalty_matrix
        self.location = jnp.zeros(penalty_matrix.shape[0])
        self.realization_transformation = [TransformationFunctions.identity]

        # Validate and setup distribution state using the validator
        self.validator = DegenerateNormalDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI related
        self.node = self.setup_node()

    def setup_node(self) -> None:
        if self._distribution_state.is_inner_prior:
            return self._setup_for_inner_prior()
        elif self._distribution_state.is_leaf_prior:
            return self._setup_for_leaf_prior()
        else:
            raise NotImplementedError(
                "DegenerateNormal can only be used as inner or leaf prior."
            )

    def _setup_for_inner_prior(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(
                partial(
                    self._compute_log_pdf,
                    penalty_matrix=self.penalty_matrix,
                    location=self.location,
                ),
                in_axes=(0, 0),
            ),
            local_dim=self.size,
            transformations=self.realization_transformation,
            parents=[self.parameters["tau2"].node],
        )

    def _setup_for_leaf_prior(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(
                partial(
                    self._compute_log_pdf,
                    tau2=self.parameters["tau2"],
                    location=self.location,
                    penalty_matrix=self.penalty_matrix,
                ),
                in_axes=(0,),
            ),
            local_dim=self.size,
            transformations=self.realization_transformation,
        )

    @staticmethod
    def _compute_log_pdf(
        realizations: jnp.ndarray,
        tau2: jnp.ndarray,
        location: jnp.ndarray,
        penalty_matrix: jnp.ndarray,
    ) -> jnp.ndarray:
        """Original log PDF computation."""
        # Compute eigenvalues and rank
        eigenvalues = jnp.linalg.eigvalsh(penalty_matrix)
        mask = eigenvalues > 1e-6
        rank = jnp.sum(mask, axis=-1)

        # Compute log pseudo-determinant
        max_index = len(eigenvalues) - rank

        def fn(i, coefficients):
            return coefficients.at[i].set(i >= max_index)

        eig_mask = fori_loop(0, eigenvalues.shape[-1], fn, eigenvalues)
        selected = jnp.where(eig_mask, eigenvalues, 1.0)
        log_pdet = jnp.sum(jnp.log(selected))
        # Adjust log pseudo-determinant for variance parameter
        log_pdet -= rank * jnp.log(tau2)

        # Compute precision matrix and quadratic form
        precision_matrix = penalty_matrix / tau2
        z = realizations - location
        quad_form = -z @ precision_matrix @ z.T

        # Combine terms for final log probability
        return 0.5 * (quad_form - (rank * jnp.log(2 * jnp.pi) - log_pdet))

    def __add__(self, other):
        raise NotImplementedError(
            "Addition operation is not supported for DegenerateNormal."
        )

    def __rmatmul__(self, other):
        self.validator.validate_multiplication(other)
        return LinearPredictor(other, self, operation="@")


class GPDDistributionValidator:
    """Handles parameter validation and state management specifically for CustomGPD distributions."""

    def __init__(self, distribution):
        self.distribution = distribution
        self.parameter_types = {}
        self.distribution_state = None

    def validate_and_setup(self):
        """Main entry point for validation and state setup."""
        self.parameter_types = {
            name: self._set_parameter_type(param)
            for name, param in self.distribution.parameters.items()
        }
        self._validate_parameters()
        self.distribution_state = self._set_distribution_state()
        return self.distribution_state

    @staticmethod
    def _set_parameter_type(param) -> ParameterType:
        """Determine the type of a parameter."""
        parameter_type = ParameterType()
        if hasattr(param, "node_type"):
            if param.node_type == "LinearPredictor":
                parameter_type.is_linearpredictor = True
                return parameter_type
            else:
                raise ValueError(
                    "All parameters must be LinearPredictors for CustomGPD"
                )
        else:
            raise ValueError("All parameters must be LinearPredictors for CustomGPD")

    def _validate_parameters(self) -> None:
        """Validate all parameters and their relationships."""
        # Ensure all parameters are LinearPredictors
        if not all(pt.is_linearpredictor for pt in self.parameter_types.values()):
            raise ValueError("All parameters must be LinearPredictors for CustomGPD")

        # Ensure responses are provided
        if self.distribution.responses is None:
            raise ValueError("Responses must be provided for CustomGPD")

    def _set_distribution_state(self) -> DistributionState:
        """Setup distribution state for likelihood-only case."""
        distribution_state = DistributionState()
        distribution_state.is_likelihood = True
        return distribution_state


class GPD(Distribution):
    """Generalized Pareto Distribution implementation."""

    def __init__(
        self,
        rv_name: str,
        location: LinearPredictor,
        scale: LinearPredictor,
        shape: LinearPredictor,
        responses: jnp.ndarray,
    ):
        super().__init__(
            rv_name,
            {"location": location, "scale": scale, "shape": shape},
            responses=responses,
        )

        # CustomGPD specific
        self.realization_transformation = [TransformationFunctions.identity]
        self.parameter_transforms = [
            TransformationFunctions.identity,  # location
            TransformationFunctions.softplus,  # scale
            TransformationFunctions.identity,  # shape
        ]

        # Validate and setup distribution state using the validator
        self.validator = GPDDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI related
        self.node = self.setup_node()

        # Model building related
        if self._distribution_state.is_likelihood:
            self.model = ModelDAG(self.responses, self.node)

    def setup_node(self) -> None:
        """Setup node for GAMLSS case only."""
        return self._setup_for_gamlss()

    def _setup_for_gamlss(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(self._compute_gamlss_log_pdf, in_axes=(None, None, 0, 0, 0)),
            transformations=self.parameter_transforms,
            parents=[
                self.parameters["location"].node,
                self.parameters["scale"].node,
                self.parameters["shape"].node,
            ],
        )

    @staticmethod
    def _compute_gamlss_log_pdf(
        realizations: jnp.ndarray,
        mask: jnp.ndarray,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        shape: jnp.ndarray,
    ) -> jnp.ndarray:
        """GAMLSS log PDF computation using CustomGPD."""
        # Create CustomGPD instance
        distribution = CustomGPD(loc=location, scale=scale, shape=shape)

        # Compute log probability
        log_pdf = distribution.log_prob(realizations)

        # Apply mask and sum
        return jnp.sum(log_pdf * mask)

    def __add__(self, other):
        raise NotImplementedError("Addition operation is not supported for CustomGPD.")

    def __rmatmul__(self, other):
        raise NotImplementedError(
            "Matrix multiplication is not supported for CustomGPD."
        )


class GEVDistributionValidator:
    """Handles parameter validation and state management specifically for the CustomGEV distribution."""

    def __init__(self, distribution):
        self.distribution = distribution
        self.parameter_types = {}
        self.distribution_state = None

    def validate_and_setup(self):
        """Main entry point for validation and state setup."""
        self.parameter_types = {
            name: self._set_parameter_type(param)
            for name, param in self.distribution.parameters.items()
        }
        self._validate_parameters()
        self.distribution_state = self._set_distribution_state()
        return self.distribution_state

    @staticmethod
    def _set_parameter_type(param) -> ParameterType:
        """Determine the type of a parameter."""
        parameter_type = ParameterType()
        if hasattr(param, "node_type"):
            if param.node_type == "LinearPredictor":
                parameter_type.is_linearpredictor = True
                return parameter_type
            else:
                raise ValueError(
                    "All parameters must be LinearPredictors for CustomGEV"
                )
        else:
            raise ValueError("All parameters must be LinearPredictors for CustomGEV")

    def _validate_parameters(self) -> None:
        """Validate all parameters and their relationships."""
        # Ensure all parameters are LinearPredictors
        if not all(pt.is_linearpredictor for pt in self.parameter_types.values()):
            raise ValueError("All parameters must be LinearPredictors for CustomGEV")

        # Ensure responses are provided
        if self.distribution.responses is None:
            raise ValueError("Responses must be provided for CustomGEV")

    def _set_distribution_state(self) -> DistributionState:
        """Setup distribution state for likelihood-only case."""
        distribution_state = DistributionState()
        distribution_state.is_likelihood = True
        return distribution_state


class GEV(Distribution):
    """Generalized Extreme Value distribution implementation."""

    def __init__(
        self,
        rv_name: str,
        location: LinearPredictor,
        scale: LinearPredictor,
        shape: LinearPredictor,
        responses: jnp.ndarray,
    ):
        super().__init__(
            rv_name,
            {"location": location, "scale": scale, "shape": shape},
            responses=responses,
        )

        # CustomGPD specific
        self.realization_transformation = [TransformationFunctions.identity]
        self.parameter_transforms = [
            TransformationFunctions.identity,  # location
            TransformationFunctions.softplus,  # scale
            TransformationFunctions.identity,  # shape
        ]

        # Validate and setup distribution state using the validator
        self.validator = GEVDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI related
        self.node = self.setup_node()

        # Model building related
        if self._distribution_state.is_likelihood:
            self.model = ModelDAG(self.responses, self.node)

    def setup_node(self) -> None:
        """Setup node for GAMLSS case only."""
        return self._setup_for_gamlss()

    def _setup_for_gamlss(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(self._compute_gamlss_log_pdf, in_axes=(None, None, 0, 0, 0)),
            transformations=self.parameter_transforms,
            parents=[
                self.parameters["location"].node,
                self.parameters["scale"].node,
                self.parameters["shape"].node,
            ],
        )

    @staticmethod
    def _compute_gamlss_log_pdf(
        realizations: jnp.ndarray,
        mask: jnp.ndarray,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        shape: jnp.ndarray,
    ) -> jnp.ndarray:
        """GAMLSS log PDF computation using CustomGPD."""
        # Create CustomGPD instance
        distribution = CustomGEV(loc=location, scale=scale, shape=shape)

        # Compute log probability
        log_pdf = distribution.log_prob(realizations)

        # Apply mask and sum
        return jnp.sum(log_pdf * mask)

    def __add__(self, other):
        raise NotImplementedError("Addition operation is not supported for CustomGEV.")

    def __rmatmul__(self, other):
        raise NotImplementedError(
            "Matrix multiplication is not supported for CustomGEV."
        )


class GEVFixedShape(Distribution):
    """Custom Generalized Extreme Value distribution implementation."""

    def __init__(
        self,
        rv_name: str,
        location: LinearPredictor,
        scale: LinearPredictor,
        shape: float,
        responses: jnp.ndarray,
    ):
        super().__init__(
            rv_name,
            {"location": location, "scale": scale},
            responses=responses,
        )

        # CustomGPD specific
        self.shape = shape
        self.realization_transformation = [TransformationFunctions.identity]
        self.parameter_transforms = [
            TransformationFunctions.identity,  # location
            TransformationFunctions.softplus,  # scale
        ]

        # Validate and setup distribution state using the validator
        self.validator = GEVDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI related
        self.node = self.setup_node()

        # Model building related
        if self._distribution_state.is_likelihood:
            self.model = ModelDAG(self.responses, self.node)

    def setup_node(self) -> None:
        """Setup node for GAMLSS case only."""
        return self._setup_for_gamlss()

    def _setup_for_gamlss(self) -> None:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(
                partial(self._compute_gamlss_log_pdf, shape=self.shape),
                in_axes=(None, None, 0, 0),
            ),
            transformations=self.parameter_transforms,
            parents=[
                self.parameters["location"].node,
                self.parameters["scale"].node,
            ],
        )

    @staticmethod
    def _compute_gamlss_log_pdf(
        realizations: jnp.ndarray,
        mask: jnp.ndarray,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        shape: jnp.ndarray,
    ) -> jnp.ndarray:
        """GAMLSS log PDF computation using CustomGPD."""
        # Create CustomGPD instance
        distribution = CustomGEV(loc=location, scale=scale, shape=shape)

        # Compute log probability
        log_pdf = distribution.log_prob(realizations)

        # Apply mask and sum
        return jnp.sum(log_pdf * mask)

    def __add__(self, other):
        raise NotImplementedError("Addition operation is not supported for CustomGEV.")

    def __rmatmul__(self, other):
        raise NotImplementedError(
            "Matrix multiplication is not supported for CustomGEV."
        )


class HalfCauchyDistributionValidator:
    """Handles parameter validation and state management specifically for HalfCauchy distributions."""

    def __init__(self, distribution):
        self.distribution = distribution
        self.parameter_types = {}
        self.distribution_state = None

    def validate_and_setup(self):
        """Main entry point for validation and state setup."""
        self.parameter_types = {
            name: self._set_parameter_type(param)
            for name, param in self.distribution.parameters.items()
        }
        self._validate_parameters()
        self.distribution_state = self._set_distribution_state()
        self._validate_likelihood_related_parameters()
        return self.distribution_state

    @staticmethod
    def _set_parameter_type(param) -> ParameterType:
        """Determine the type of a parameter."""
        parameter_type = ParameterType()
        if hasattr(param, "node_type"):
            if param.node_type == "LinearPredictor":
                raise ValueError(
                    "HalfCauchy cannot have LinearPredictor parameters as it cannot be a likelihood."
                )
            elif param.node_type == "Distribution":
                parameter_type.is_randomvariable = True
                return parameter_type
        elif isinstance(param, jnp.ndarray):
            parameter_type.is_array = True
            return parameter_type
        raise ValueError(f"Invalid parameter type: {type(param)}")

    def _validate_parameters(self) -> None:
        """Validate all parameters and their relationships."""
        scale_param = self.distribution.parameters["scale"]
        distribution_size = self.distribution.size

        # Check if scale's size matches the distribution's size
        if hasattr(scale_param, "size"):
            if scale_param.size != distribution_size:
                raise ValueError(
                    f"Scale parameter size {scale_param.size} does not match distribution size {distribution_size}"
                )
        elif isinstance(scale_param, jnp.ndarray):
            if scale_param.shape[0] != distribution_size:
                raise ValueError(
                    f"Scale array size {scale_param.shape[0]} does not match distribution size {distribution_size}"
                )
        else:
            # Handle scalar scale parameter
            if distribution_size != 1:
                raise ValueError(
                    f"Scale is a scalar but distribution size is {distribution_size}"
                )

        # Validate prior requirements
        param_types = list(self.parameter_types.values())
        has_rv = any(pt.is_randomvariable for pt in param_types)
        all_rv = all(pt.is_randomvariable for pt in param_types)
        if has_rv and not all_rv:
            raise NotImplementedError(
                "For simplicity, only priors are supported whose parameters are either all random variables or none."
            )

    def _set_distribution_state(self) -> DistributionState:
        """Setup distribution state based on parameter types."""
        distribution_state = DistributionState()
        parameter_types = self.parameter_types.values()

        all_rv = all(pt.is_randomvariable for pt in parameter_types)
        all_leaf = all(pt.is_array for pt in parameter_types)

        if all_rv:
            distribution_state.is_inner_prior = True
        elif all_leaf:
            distribution_state.is_leaf_prior = True

        return distribution_state

    def _validate_likelihood_related_parameters(self):
        """Validate likelihood-related parameters."""
        if self.distribution.responses is not None:
            raise ValueError(
                "Responses must not be provided for HalfCauchy as it cannot be a likelihood."
            )
        if self.distribution_state.is_likelihood:
            raise ValueError("HalfCauchy cannot be a likelihood.")


class HalfCauchy(Distribution):
    """HalfCauchy distribution implementation with fixed loc=0.0."""

    def __init__(
        self,
        rv_name: str,
        scale: Union[jnp.ndarray, Distribution],
        size: int,
    ):
        self.loc = jnp.zeros(size)

        if size != scale.size:
            raise ValueError(
                f"Scale parameter size {scale.size} does not match inferred size {size}"
            )

        super().__init__(rv_name, {"scale": scale}, size)

        # HalfCauchy specific
        self.realization_transformation = [TransformationFunctions.softplus]

        # Validate and setup distribution state using the validator
        self.validator = HalfCauchyDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI related
        self.node = self.setup_node()

        # Model building related
        if self._distribution_state.is_likelihood:
            raise ValueError("HalfCauchy cannot be a likelihood.")

    def setup_node(self) -> Node:
        if self._distribution_state.is_inner_prior:
            return self._setup_for_inner_prior()
        elif self._distribution_state.is_leaf_prior:
            return self._setup_for_leaf_prior()
        else:
            raise NotImplementedError(
                "HalfCauchy can only be used as inner or leaf prior."
            )

    def _setup_for_inner_prior(self) -> Node:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(self._compute_prior_log_pdf, in_axes=(0, 0)),
            local_dim=self.size,
            transformations=self.realization_transformation,
            parents=[self.parameters["scale"].node],
        )

    def _setup_for_leaf_prior(self) -> Node:
        return Node(
            name=self.rv_name,
            log_pdf=vmap(
                partial(
                    self._compute_prior_log_pdf,
                    scale=self.parameters["scale"],
                ),
                in_axes=(0,),
            ),
            local_dim=self.size,
            transformations=self.realization_transformation,
        )

    @staticmethod
    def _compute_prior_log_pdf(
        realizations: jnp.ndarray, scale: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the log PDF accounting for fixed loc=0.0."""
        log_pdf = tfd.HalfCauchy(loc=0.0, scale=scale).log_prob(realizations)
        #  -inverse_log_det_jacobian=forward_log_det_jacobian
        log_det_jacobian = -Softplus().inverse_log_det_jacobian(realizations)
        return log_pdf + log_det_jacobian

    def __add__(self, other):
        raise NotImplementedError("Addition operation is not supported for HalfCauchy.")

    def __rmatmul__(self, other):
        raise NotImplementedError(
            "Matrix multiplication is not supported for HalfCauchy."
        )


class ALDDistributionValidator:
    """
    Handles parameter validation and state management specifically for CustomALD distributions.

    The ALD distribution has four parameters:
      - loc: must be a LinearPredictor.
      - scale: must be a LinearPredictor.
      - tau: must be a constant (float) that lies in (0, 1).
      - c: must be a constant (float) and positive.

    In addition, since ALD is only allowed as a likelihood, responses must be provided
    and no explicit size is allowed.
    """

    def __init__(self, distribution):
        self.distribution = distribution
        self.parameter_types: Dict[str, ParameterType] = {}
        self.distribution_state: Optional[DistributionState] = None

    def validate_and_setup(self):
        # Set the type for each parameter.
        self.parameter_types = {
            name: self._set_parameter_type(name, param)
            for name, param in self.distribution.parameters.items()
        }
        self._validate_parameters()
        self.distribution_state = self._set_distribution_state()
        self._validate_likelihood_related_parameters()
        return self.distribution_state

    def _set_parameter_type(self, name: str, param) -> ParameterType:
        parameter_type = ParameterType()
        if name in ["loc", "scale"]:
            # These must be LinearPredictor objects.
            if hasattr(param, "node_type"):
                if param.node_type == "LinearPredictor":
                    parameter_type.is_linearpredictor = True
                    return parameter_type
                else:
                    raise ValueError(
                        f"{name} parameter must be a LinearPredictor for ALD."
                    )
            else:
                raise ValueError(f"{name} parameter must be a LinearPredictor for ALD.")
        elif name in ["tau", "c"]:
            # These parameters must be constants (float).
            if isinstance(param, float):
                return parameter_type
            else:
                raise ValueError(f"{name} parameter must be a float for ALD.")
        else:
            raise ValueError(f"Unknown parameter '{name}' for ALD.")

    def _validate_parameters(self) -> None:
        tau_val = self.distribution.parameters["tau"]
        c_val = self.distribution.parameters["c"]

        if not (0 < tau_val < 1):
            raise ValueError("`tau` must lie in (0, 1) for ALD.")
        if not (c_val > 0):
            raise ValueError("`c` must be positive for ALD.")

    def _set_distribution_state(self) -> DistributionState:
        # ALD is only supported as a likelihood.
        distribution_state = DistributionState()
        distribution_state.is_likelihood = True
        return distribution_state

    def _validate_likelihood_related_parameters(self):
        # Likelihood distributions must have responses and must not have an explicit size.
        if self.distribution.responses is None:
            raise ValueError("Responses must be provided for ALD (likelihood).")
        if self.distribution.size is not None:
            raise ValueError(
                "Size should not be provided for a likelihood distribution; it is implicitly determined by the responses."
            )


class ALD(Distribution):
    """
    Asymmetric Laplace Distribution (ALD) implementation as a likelihood.

    Parameters:
      - loc: a LinearPredictor for the location parameter.
      - scale: a LinearPredictor for the positive scale parameter.
      - tau: a float constant for the quantile level (must lie in (0, 1)).
      - c: a float constant for the tuning parameter (must be positive).
      - responses: observed responses.

    The log-PDF is defined using a partially applied function that fixes the constants tau and c.
    """

    def __init__(
        self,
        rv_name: str,
        loc: LinearPredictor,
        scale: LinearPredictor,
        tau: float,
        c: float,
        responses: jnp.ndarray,
    ):
        # Build the parameters dictionary.
        super().__init__(
            rv_name,
            {"loc": loc, "scale": scale, "tau": tau, "c": c},
            responses=responses,
        )

        # ALD-specific transformations:
        self.realization_transformation = [TransformationFunctions.identity]
        # Only loc and scale are predictors, so only these receive transformations.
        self.parameter_transforms = [
            TransformationFunctions.identity,  # for loc
            TransformationFunctions.softplus,  # for scale
        ]

        # Validate and set up distribution state using the validator.
        self.validator = ALDDistributionValidator(self)
        self._distribution_state = self.validator.validate_and_setup()
        self.parameter_types = self.validator.parameter_types

        # VI-related: set up the node.
        self.node = self.setup_node()

        # Model building: if the distribution is a likelihood, create a ModelDAG.
        if self._distribution_state.is_likelihood:
            self.model = ModelDAG(self.responses, self.node)

    def setup_node(self) -> Node:
        return self._setup_for_gamlss()

    def _setup_for_gamlss(self) -> Node:
        """
        Sets up a node with a GAMLSS log-PDF function.

        The log-PDF function is partially applied with the fixed (constant) tau and c parameters.
        Its signature after partial application is:
            (realizations, mask, loc, scale)
        """
        return Node(
            name=self.rv_name,
            log_pdf=vmap(
                partial(
                    self._compute_gamlss_log_pdf,
                    tau=self.parameters["tau"],
                    c=self.parameters["c"],
                ),
                in_axes=(None, None, 0, 0),
            ),
            transformations=self.parameter_transforms,
            parents=[
                self.parameters["loc"].node,
                self.parameters["scale"].node,
            ],
        )

    @staticmethod
    def _compute_gamlss_log_pdf(
        realizations: jnp.ndarray,
        mask: jnp.ndarray,
        loc: jnp.ndarray,
        scale: jnp.ndarray,
        tau: float,
        c: float,
    ) -> jnp.ndarray:
        """
        Computes the GAMLSS log-PDF for ALD.

        The CustomALD distribution defines the log density as:
            log f(x) = log(tau*(1-tau)) - log(scale) - ρ₍τ, c₎((x - loc)/scale)
        where ρ₍τ, c₎ is the modified check function.
        """
        distribution = CustomALD(loc=loc, scale=scale, tau=tau, c=c)
        log_pdf = distribution.log_prob(realizations)
        return jnp.sum(log_pdf * mask)

    def __add__(self, other):
        raise NotImplementedError("Addition operation is not supported for ALD.")

    def __rmatmul__(self, other):
        raise NotImplementedError("Matrix multiplication is not supported for ALD.")
