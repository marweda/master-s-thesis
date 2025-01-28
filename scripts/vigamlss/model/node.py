from typing import Optional, Callable

import jax.numpy as jnp


class Node:
    """Node in a DAG representing the model structure."""
    def __init__(
        self,
        name: Optional[str] = None,
        log_pdf: Optional[Callable] = None,
        local_dim: Optional[int] = None,
        transformations: list[Callable] = [],
        covariates: list[jnp.ndarray] = [],
        parents: list["Node"] = [],
        dp_markers: list[bool] = [],
    ):
        self.name = name
        self.log_pdf = log_pdf
        self.local_dim = local_dim
        self.transformations = transformations
        self.covariates = covariates
        self.parents = parents
        self.dp_markers = dp_markers
        self._idx: int = None
