from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import vmap

from .model.model_builder import Node


@dataclass
class DesignMatrix:
    name: str
    matrix: jnp.ndarray
    size: int
    node_type: str = field(default_factory=lambda: "DesignMatrix")

    def __post_init__(self):
        self.node = Node(
            name=self.name,
            covariates=[self.matrix],
        )


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self):
        self.mean_: Optional[jnp.ndarray] = None
        self.scale_: Optional[jnp.ndarray] = None

    def fit(self, X: jnp.ndarray) -> None:
        """Compute mean and scale for later scaling."""
        self.mean_ = jnp.mean(X, axis=0)
        self.scale_ = jnp.std(X, axis=0) + 1e-8

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Perform standardization."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler needs to be fitted before transform")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Fit to data and perform standardization."""
        self.fit(X)
        return self.transform(X)


class BSplineBasis:
    """B-spline basis configuration and computation."""

    def __init__(
        self,
        degree: int,
        num_knots: int,
        user_knots: Optional[jnp.ndarray] = None,
        use_quantile: bool = False,
    ):
        """
        If user_knots is not None, those will be used directly.
        Otherwise, knots are generated automatically from data.
        """
        self.degree = degree
        self.num_knots = num_knots
        self.user_knots = user_knots
        self.use_quantile = use_quantile
        self._knots_used: Optional[jnp.ndarray] = None  # Will store the final knots

    @property
    def num_basis_functions(self) -> int:
        """Number of basis functions."""
        return self.num_knots + self.degree - 1

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Transform input data using B-spline basis functions.
        If user_knots is provided, it overrides automatic knot generation.
        """
        if self.user_knots is not None:
            knots = self.user_knots
        else:
            knots = self._generate_knots(X)

        self._knots_used = knots

        Z = self._evaluate_all_bases(knots, X)
        return Z.T

    def _generate_knots(self, X: jnp.ndarray) -> jnp.ndarray:
        """Generate B-spline knots from data."""
        x_min, x_max = jnp.min(X), jnp.max(X)

        if self.use_quantile:
            percentiles = jnp.linspace(
                100 / (self.num_knots + 1),
                100 * self.num_knots / (self.num_knots + 1),
                self.num_knots
            )
            interior_knots = jnp.percentile(X, percentiles)
            interior_knots = jnp.clip(interior_knots, x_min, x_max)

            boundary_lower = jnp.full((self.degree + 1,), x_min)
            boundary_upper = jnp.full((self.degree + 1,), x_max)

            knots = jnp.concatenate([boundary_lower, interior_knots, boundary_upper])
            knots = jnp.sort(knots)
        else:
            interior_knots, step = jnp.linspace(
                x_min, x_max, self.num_knots, retstep=True
            )
            step_lower = step_upper = step

            lower_knots = x_min - step_lower * jnp.arange(self.degree, 0, -1)
            upper_knots = x_max + step_upper * jnp.arange(1, self.degree + 1)

            knots = jnp.concatenate([lower_knots, interior_knots, upper_knots])

        return knots

    def _evaluate_all_bases(self, knots: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all B-spline basis functions."""
        return vmap(lambda i: self._evaluate_basis(knots, i, X))(
            jnp.arange(self.num_basis_functions) + self.degree
        )

    def _evaluate_basis(
        self, knots: jnp.ndarray, j: int, X: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate a single B-spline basis function using Cox-de Boor recursion."""

        def cox_de_boor(xi: float, jj: int, d: int) -> float:
            """Cox-de Boor recursion formula implementation."""
            if d == 0:
                return jnp.where((knots[jj] <= xi) & (xi < knots[jj + 1]), 1.0, 0.0)

            w1 = jnp.where(
                knots[jj] != knots[jj - d],
                (xi - knots[jj - d]) / (knots[jj] - knots[jj - d]),
                0.0,
            )
            w2 = jnp.where(
                knots[jj + 1] != knots[jj + 1 - d],
                (knots[jj + 1] - xi) / (knots[jj + 1] - knots[jj + 1 - d]),
                0.0,
            )

            return w1 * cox_de_boor(xi, jj - 1, d - 1) + w2 * cox_de_boor(xi, jj, d - 1)

        return vmap(lambda xi: cox_de_boor(xi, j, self.degree))(X)


class PSpline:
    """Handles P-spline basis transformations and penalty calculations."""

    def __init__(self, bspline_basis: jnp.ndarray, penalty_order: int = 2):
        """Initialize PSpline."""
        # bspline_basis is a (n_samples, n_basis) design matrix
        self.bspline_basis = bspline_basis
        self.penalty_order = penalty_order
        self.penalty_matrix = self._create_penalty_matrix()

    def __call__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply the sum-to-zero constraint via a QR-based approach
        and return (constrained_design_matrix, constrained_penalty).
        """
        constrained_matrix, constrained_penalty = self._apply_sum_zero_constraint(
            self.bspline_basis
        )
        return constrained_matrix, constrained_penalty

    def _create_penalty_matrix(self) -> jnp.ndarray:
        """Create difference penalty matrix."""
        n_basis = self.bspline_basis.shape[1]
        D = jnp.diff(jnp.eye(n_basis), n=self.penalty_order, axis=0)
        return D.T @ D

    def _apply_sum_zero_constraint(
        self, basis_matrix: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply sum-to-zero constraint using QR decomposition."""
        C = jnp.ones((1, basis_matrix.shape[0])) @ basis_matrix
        Q = jnp.linalg.qr(C.T, mode="complete")[0]
        Z = Q[:, C.shape[0] :]
        X_const = basis_matrix @ Z
        K_const = Z.T @ self.penalty_matrix @ Z
        return X_const, K_const


class DataPreparator:
    """Prepare data for model fitting.

    - If basis_transformation='pspline', a B-spline basis with penalty is created.
      The design matrix is automatically constrained to sum-to-zero (PSpline).
    - If basis_transformation='identity', the raw data is used (optionally with intercept).
    - If standardize=True, data is standardized before applying the basis transformation.

    If 'return_knots' True, the __call__ method
    will return the knots (user-supplied or automatically generated) as part of the result.
    """

    def __init__(
        self,
        name: str,
        data: jnp.ndarray,
        basis_transformation: str,
        intercept: bool,
        standardize: bool,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str
            Name for the resulting design matrix.
        data : jnp.ndarray
            Input array of shape (n_samples,) or (n_samples, n_features).
        basis_transformation : str
            Either 'pspline' or 'identity'.
        intercept : bool
            If True, add an intercept column (not valid with 'pspline').
        standardize : bool
            If True, standardize the data before transformation.
        **kwargs
            Additional configuration:
              - degree : int (for pspline)
              - num_knots : int (for pspline)
              - user_knots : Optional[jnp.ndarray]
              - return_knots : bool
            - use_quantile : bool (for pspline)
        """
        self.data = data
        self.basis_transformation = basis_transformation
        self.intercept = intercept
        self.standardize = standardize

        # Pop the special flags/parameters
        self.user_knots = kwargs.pop("user_knots", None)
        self.return_knots = kwargs.pop("return_knots", False)

        self.kwargs = kwargs

        self._validate_parameters()

        if self.standardize:
            self._standardize_data()

        # Build the design matrix
        if self.basis_transformation == "pspline":
            self._create_pspline_design_matrix(name)
        elif self.intercept:
            # 'identity' transformation with an intercept
            self._create_intercept_design_matrix(name)
        else:
            # 'identity' transformation with no intercept
            # we only do something if neither intercept nor pspline
            # basically pass data as-is
            self._create_identity_design_matrix(name)

    def _validate_parameters(self) -> None:
        valid_transformations = ["pspline", "identity"]
        if self.basis_transformation not in valid_transformations:
            raise ValueError(
                f"Invalid basis transformation: {self.basis_transformation}"
            )
        if self.basis_transformation == "pspline" and self.intercept:
            raise ValueError("Intercept not supported with P-spline basis")

    def _standardize_data(self) -> None:
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

    def _create_identity_design_matrix(self, name: str) -> None:
        # Just pass data as a design matrix
        data_2d = self.data.reshape(-1, 1) if self.data.ndim == 1 else self.data
        self.design_matrix = DesignMatrix(
            name=name, matrix=data_2d, size=data_2d.shape[1]
        )
        self.K = None

    def _create_intercept_design_matrix(self, name: str) -> None:
        data_2d = self.data.reshape(-1, 1) if self.data.ndim == 1 else self.data
        design_matrix = jnp.hstack([jnp.ones((data_2d.shape[0], 1)), data_2d])
        self.design_matrix = DesignMatrix(
            name=name,
            matrix=design_matrix,
            size=design_matrix.shape[1],
        )
        self.K = None

    def _create_pspline_design_matrix(self, name: str) -> None:
        # Build a B-spline with optional user_knots
        bspline = BSplineBasis(
            degree=self.kwargs["degree"],
            num_knots=self.kwargs["num_knots"],
            user_knots=self.user_knots,
            use_quantile=self.kwargs.get("use_quantile", False),
        )
        bspline_matrix = bspline.transform(self.data)
        self._knots_used = bspline._knots_used

        constrained_matrix, constrained_penalty = PSpline(bspline_matrix)()
        self.design_matrix = DesignMatrix(
            name=name,
            matrix=constrained_matrix,
            size=constrained_matrix.shape[1],
        )
        self.K = constrained_penalty

    def __call__(self):
        """
        Returns
        -------
        - If basis_transformation='pspline':
            (design_matrix, penalty_matrix) or
            (design_matrix, penalty_matrix, knots) if return_knots=True
        - If basis_transformation='identity' (or used intercept):
            design_matrix alone or
            (design_matrix, None) if return_knots=True
        """
        if self.basis_transformation == "pspline":
            if self.return_knots:
                # Return either the user-supplied knots (if provided)
                # or the automatically generated ones
                if self.user_knots is not None:
                    actual_knots = self.user_knots
                else:
                    actual_knots = self._knots_used

                return self.design_matrix, self.K, actual_knots
            else:
                return self.design_matrix, self.K
        else:
            # identity or intercept
            if self.return_knots:
                # We have no knots in this scenario, so return None
                return self.design_matrix, None
            else:
                return self.design_matrix
