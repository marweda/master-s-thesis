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


@dataclass
class BSplineBasis:
    """B-spline basis configuration and computation."""

    degree: int
    num_knots: int

    @property
    def num_basis_functions(self) -> int:
        """Number of basis functions."""
        return self.num_knots + self.degree - 1

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Transform input data using B-spline basis functions."""
        knots = self._generate_knots(X)
        Z = self._evaluate_all_bases(knots, X)
        return Z.T

    def _generate_knots(self, X: jnp.ndarray) -> jnp.ndarray:
        """Generate B-spline knots from data."""
        x_min, x_max = X.min(), X.max()
        interior_knots, step = jnp.linspace(x_min, x_max, self.num_knots, retstep=True)

        lower_knots = jnp.full(
            self.degree, x_min - step * jnp.arange(self.degree, 0, -1)
        )
        upper_knots = jnp.full(
            self.degree, x_max + step * jnp.arange(1, self.degree + 1)
        )

        return jnp.concatenate([lower_knots, interior_knots, upper_knots])

    def _evaluate_all_bases(self, knots: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all B-spline basis functions."""
        return vmap(lambda i: self._evaluate_basis(knots, i, X))(
            jnp.arange(self.num_basis_functions) + self.degree
        )

    def _evaluate_basis(
        self, knots: jnp.ndarray, j: int, X: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate B-spline basis function using Cox-de Boor recursion."""

        def cox_de_boor(xi: float, j: int, d: int) -> float:
            """Cox-de Boor recursion formula implementation."""
            if d == 0:
                return jnp.where((knots[j] <= xi) & (xi < knots[j + 1]), 1.0, 0.0)

            w1 = jnp.where(
                knots[j] != knots[j - d],
                (xi - knots[j - d]) / (knots[j] - knots[j - d]),
                0.0,
            )
            w2 = jnp.where(
                knots[j + 1] != knots[j + 1 - d],
                (knots[j + 1] - xi) / (knots[j + 1] - knots[j + 1 - d]),
                0.0,
            )

            return w1 * cox_de_boor(xi, j - 1, d - 1) + w2 * cox_de_boor(xi, j, d - 1)

        return vmap(lambda xi: cox_de_boor(xi, j, self.degree))(X)


class PSpline:
    """Handles P-spline basis transformations and penalty calculations."""

    def __init__(self, bspline_basis: jnp.ndarray, penalty_order: int = 2):
        """Initialize PSpline."""
        self.bspline_basis = bspline_basis
        self.penalty_order = penalty_order
        self.penalty_matrix = self._create_penalty_matrix()

    def __call__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Transform covariates and apply sum-to-zero constraint."""
        # Apply sum-to-zero constraint
        constrained_matrix, constrained_penalty = self._apply_sum_zero_constraint(
            self.bspline_basis
        )

        return constrained_matrix, constrained_penalty

    def _transform_covariates(self, covariate: jnp.ndarray) -> jnp.ndarray:
        """Transform covariates using P-spline basis."""
        return self.bspline_basis.transform(covariate)

    def _create_penalty_matrix(self) -> jnp.ndarray:
        """Create difference penalty matrix."""
        n_basis = self.bspline_basis.shape[1]
        D = jnp.diff(jnp.eye(n_basis), n=self.penalty_order, axis=0)
        return D.T @ D

    def _apply_sum_zero_constraint(
        self, basis_matrix: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply sum-to-zero constraint using QR decomposition."""
        # Create constraint matrix
        C = jnp.ones((1, basis_matrix.shape[0])) @ basis_matrix

        # Get null space basis
        Q = jnp.linalg.qr(C.T, mode="complete")[0]
        Z = Q[:, C.shape[0] :]

        # Transform basis and penalty
        X_const = basis_matrix @ Z
        K_const = Z.T @ self.penalty_matrix @ Z

        return X_const, K_const


class DataPreparator:
    """Prepare data for model fitting."""

    def __init__(
        self,
        name: str,
        data: jnp.ndarray,
        basis_transformation: str,
        intercept: bool,
        standardize: bool,
        **kwargs,
    ):
        self.data = data
        self.basis_transformation = basis_transformation
        self.intercept = intercept
        self.standardize = standardize
        self.kwargs = kwargs

        self._validate_parameters()

        if self.standardize:
            self._standardize_data()

        if self.basis_transformation == "pspline":
            self._create_pspline_design_matrix(name)
        elif self.intercept:
            self._create_intercept_design_matrix(name)

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        valid_transformations = ["pspline", "identity"]
        if self.basis_transformation not in valid_transformations:
            raise ValueError(
                f"Invalid basis transformation: {self.basis_transformation}"
            )
        if self.basis_transformation == "pspline" and self.intercept:
            raise ValueError("Intercept not supported with P-spline basis")

    def _standardize_data(self) -> None:
        """Standardize the data using StandardScaler."""
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

    def _create_intercept_design_matrix(self, name: str) -> None:
        """Create design matrix with an intercept column."""
        data_2d = self.data.reshape(-1, 1) if self.data.ndim == 1 else self.data
        design_matrix = jnp.hstack([jnp.ones((data_2d.shape[0], 1)), data_2d])
        self.design_matrix = DesignMatrix(
            name=name, matrix=design_matrix, size=design_matrix.shape[1]
        )

    def _create_pspline_design_matrix(self, name: str) -> None:
        """Create P-spline design matrix and penalty matrix."""
        bspline_basis = BSplineBasis(**self.kwargs).transform(self.data)
        constrained_design_matrix, constrained_penalty = PSpline(bspline_basis)()
        self.design_matrix = DesignMatrix(
            name=name,
            matrix=constrained_design_matrix,
            size=constrained_design_matrix.shape[1],
        )
        self.K = constrained_penalty

    def __call__(self):
        """Return design matrix and penalty matrix (if P-spline)."""
        if self.basis_transformation == "pspline":
            return self.design_matrix, self.K
        return self.design_matrix
