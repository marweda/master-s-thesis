from typing import Callable, Type, Optional
from itertools import accumulate, chain

import jax
import jax.numpy as jnp
import networkx as nx
from jax.random import PRNGKey
from tensorflow_probability.substrates.jax.math import fill_triangular
from optax import GradientTransformation

from .node import Node
from ..svi.variational_distributions import VariationalDistribution
from ..svi.svi_core import core_svi_optimization
from ..svi.svi_utils.minibatching import prepare_mini_batching
from ..svi.svi_utils.misc_preperations import prepare_vi_dist, prepare_opt_state


class DAGSharedInfoProvider:
    """Provides shared information and utilities used across multiple components."""

    def __init__(self, graph, likelihood_node):
        self.graph = graph
        self.likelihood_node = likelihood_node

    def get_linear_predictor_nodes(self):
        """Returns list of linear predictor nodes."""
        return list(self.graph.predecessors(self.likelihood_node))

    def get_prior_nodes(self, linear_predictor_nodes):
        """Returns list of prior nodes."""
        return [
            node
            for lp_node in linear_predictor_nodes
            for node in list(nx.ancestors(self.graph, lp_node))
        ]

    def compute_non_none_covariates(self, covariates_per_lp):
        """Computes number of non-None covariates per LP node."""
        non_none_covariates_per_lp = [
            [
                cov
                for covariates in covariates_list
                for cov in (
                    covariates if isinstance(covariates, list) else [covariates]
                )
            ]
            for covariates_list in covariates_per_lp
        ]
        return [len(covariates) for covariates in non_none_covariates_per_lp]


class DAGIndexManager:
    """Manages node indices and special indices."""

    def __init__(self, shared_info: DAGSharedInfoProvider):
        self.shared_info = shared_info
        self.linear_predictor_nodes = None
        self.prior_nodes = None
        self.total_num_vi_nodes = None
        self.all_nodes = None
        self.vi_idxs = None
        self.lp_idxs = None
        self.mask_idx = None
        self.response_idx = None

    def assign_indices(self):
        """Assigns and manages all node indices."""
        self.linear_predictor_nodes = self.shared_info.get_linear_predictor_nodes()
        self.prior_nodes = self.shared_info.get_prior_nodes(self.linear_predictor_nodes)
        self.total_num_vi_nodes = len(self.prior_nodes)

        self.all_nodes = (
            self.prior_nodes
            + self.linear_predictor_nodes
            + [self.shared_info.likelihood_node]
        )

        for idx, node in enumerate(self.all_nodes):
            node._idx = idx

        self.vi_idxs = list(range(len(self.prior_nodes)))
        self.lp_idxs = list(
            range(
                len(self.prior_nodes),
                len(self.prior_nodes) + len(self.linear_predictor_nodes),
            )
        )

        # Set special indices
        likelihood_node_idx = len(self.prior_nodes) + len(self.linear_predictor_nodes)
        self.mask_idx = likelihood_node_idx
        self.response_idx = likelihood_node_idx + 1


class DAGMetadataCollector:
    """Collects and manages metadata from nodes."""

    def __init__(self, graph):
        self.graph = graph
        self.rv_names = []
        self.log_pdfs = []
        self.transformations = []
        self.num_vi_params = []
        self.covariates = []
        self.dp_markers = []
        self.dependencies = []

    def collect_metadata(self, sorted_nodes):
        """Collects all metadata from nodes."""
        self._collect_node_metadata(sorted_nodes)
        self._collect_dependencies(sorted_nodes)

    def _collect_node_metadata(self, sorted_nodes):
        """Collects individual node metadata."""
        for node in sorted_nodes:
            if node.name is not None:
                self.rv_names.append(node.name)

            if node.log_pdf is not None:
                self.log_pdfs.append(node.log_pdf)

            if node.transformations:
                self.transformations.append(node.transformations)

            if node.covariates:
                flattened_covariates = []
                for covariate in node.covariates:
                    if covariate is not None:
                        if isinstance(covariate, list):
                            flattened_covariates.extend(
                                [c for c in covariate if c is not None]
                            )
                        else:
                            flattened_covariates.append(covariate)
                if flattened_covariates:
                    self.covariates.append(flattened_covariates)

            if hasattr(node, "local_dim") and node.local_dim is not None:
                self.num_vi_params.append(node.local_dim)

            if hasattr(node, "dp_markers") and node.dp_markers:
                self.dp_markers.append(node.dp_markers)

    def _collect_dependencies(self, sorted_nodes):
        """Collects node dependencies."""
        for node in sorted_nodes:
            predecessor_nodes = list(self.graph.predecessors(node))
            predecessor_indices = [pred._idx for pred in predecessor_nodes]
            self.dependencies.append(predecessor_indices)


class DAGIndicesCreator:
    """Creates various indices used in the model."""

    def __init__(self, metadata_collector: DAGMetadataCollector):
        self.metadata_collector = metadata_collector
        self.split_indices = []
        self.dp_indices = []
        self.add_indices = []
        self.arg_indices = []
        self.bigX = None
        self._covariate_inbigX_positions = []
        self._vi_nodes_for_dp = []

    def create_all_indices(self, sorted_nodes, sorted_lp_nodes):
        """Creates all necessary indices."""
        self._assign_global_vi_slices()
        self._filter_vi_idxs_for_dp(sorted_lp_nodes)
        self._create_bigX_and_mapping()
        self._create_dp_indices()
        self._create_add_indices(sorted_lp_nodes)
        self._create_arg_indices(sorted_nodes)

    def _assign_global_vi_slices(self):
        """Assigns global VI parameter slices."""
        self.split_indices = list(
            accumulate(self.metadata_collector.num_vi_params[:-1])
        )

    def _filter_vi_idxs_for_dp(self, sorted_lp_nodes):
        """Filters VI indices for Dirichlet Process."""
        lp_parents_idxs = [
            parent._idx for node in sorted_lp_nodes for parent in node.parents
        ]

        dp_markers = [
            dp_marker for node in sorted_lp_nodes for dp_marker in node.dp_markers
        ]

        self._vi_nodes_for_dp = [
            idx for idx, dp_marker in zip(lp_parents_idxs, dp_markers) if dp_marker
        ]

    def _create_bigX_and_mapping(self):
        """Creates global design matrix and mapping."""
        current_col = 0
        flattened_covariates = []

        for covariate_list in self.metadata_collector.covariates:
            if isinstance(covariate_list, list):
                for covariate in covariate_list:
                    n_cols = covariate.shape[1]
                    self._covariate_inbigX_positions.append([current_col, n_cols])
                    current_col += n_cols
                    flattened_covariates.append(covariate)
            else:
                n_cols = covariate_list.shape[1]
                self._covariate_inbigX_positions.append([current_col, n_cols])
                current_col += n_cols
                flattened_covariates.append(covariate_list)

        self.bigX = jnp.hstack(flattened_covariates) if flattened_covariates else None

    def _create_dp_indices(self):
        """Creates Dirichlet Process indices."""
        self.dp_indices = [
            [vi_idx, covariate_positions]
            for vi_idx, covariate_positions in zip(
                self._vi_nodes_for_dp, self._covariate_inbigX_positions
            )
        ]

    def _create_add_indices(self, sorted_lp_nodes):
        """Creates additional indices."""
        lp_parents_idxs = [
            [parent._idx for parent in node.parents] for node in sorted_lp_nodes
        ]
        dp_markers = [node.dp_markers for node in sorted_lp_nodes]

        temp_vi_nodes_for_add = [
            [
                idx if not dp_marker else None
                for idx, dp_marker in zip(lp_parent_idxs, lp_dp_marker)
            ]
            for lp_parent_idxs, lp_dp_marker in zip(lp_parents_idxs, dp_markers)
        ]

        vi_indices_for_add = [
            [idx for idx in vi_idxs if idx is not None]
            for vi_idxs in temp_vi_nodes_for_add
        ]

        num_dp_terms_per_lp = [sum(dm) for dm in dp_markers]
        list_of_range_of_num_dp_terms = [
            list(range(sum(num_dp_terms_per_lp[:i]), sum(num_dp_terms_per_lp[: i + 1])))
            for i in range(len(num_dp_terms_per_lp))
        ]

        self.add_indices = [
            [vi_idx] + [range_list]
            for vi_idx, range_list in zip(
                vi_indices_for_add, list_of_range_of_num_dp_terms
            )
        ]

    def _create_arg_indices(self, sorted_nodes):
        """Creates argument indices."""
        response_idx = max(node._idx for node in sorted_nodes) + 1
        mask_idx = response_idx - 1

        self.arg_indices = []
        for vi_node_idx in [
            n._idx
            for n in sorted_nodes
            if n._idx < len(self.metadata_collector.num_vi_params)
        ]:
            self.arg_indices.append(
                [vi_node_idx] + self.metadata_collector.dependencies[vi_node_idx]
            )

        final_dependencies = self.metadata_collector.dependencies[-1]
        self.arg_indices.append([response_idx, mask_idx] + final_dependencies)


class ModelDAG:
    """ModelDAG builds and processes a directed acyclic graph of Nodes."""

    def __init__(self, responses: jnp.ndarray, likelihood_node: Node):
        self.responses = responses
        self._likelihood_node = likelihood_node
        self._graph = nx.DiGraph()
        self._build_graph()

        # Initialize components
        self._shared_info = DAGSharedInfoProvider(self._graph, self._likelihood_node)
        self._index_manager = DAGIndexManager(self._shared_info)
        self._metadata_collector = DAGMetadataCollector(self._graph)
        self._indices_creator = DAGIndicesCreator(self._metadata_collector)

        # Initialize sorting-related attributes
        self._sorted_nodes = []
        self._ascending_idx_sorted_nodes = []
        self._vi_nodes = []
        self._ascending_idx_sorted_vi_nodes = []
        self._lp_nodes = []
        self._ascending_idx_sorted_lp_nodes = []

        self._process_graph()

    def _build_graph(self):
        """Builds the underlying directed graph."""
        if self._likelihood_node is None:
            raise ValueError("Likelihood node must be set before building the graph")

        self._add_node(self._likelihood_node)
        self._build_edges()

        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("The constructed graph contains cycles and is not a DAG")

    def _add_node(self, node: Node):
        """Adds a single node to the graph."""
        self._graph.add_node(node)

    def _build_edges(self):
        """Builds directed edges in the graph."""
        nodes_to_process = [self._likelihood_node]
        processed_nodes = set()

        while nodes_to_process:
            current_node = nodes_to_process.pop(0)
            if current_node in processed_nodes:
                continue

            for parent in current_node.parents:
                self._graph.add_edge(parent, current_node)
                if parent not in processed_nodes:
                    nodes_to_process.append(parent)

            processed_nodes.add(current_node)

    def _process_graph(self):
        """Processes the graph structure and collects necessary information."""
        # Perform topological sort
        self._sorted_nodes = list(nx.topological_sort(self._graph))

        # Assign indices
        self._index_manager.assign_indices()

        # Sort nodes
        self._sort_nodes()

        # Collect metadata
        self._metadata_collector.collect_metadata(self._ascending_idx_sorted_nodes)

        # Create indices
        self._indices_creator.create_all_indices(
            self._ascending_idx_sorted_nodes, self._ascending_idx_sorted_lp_nodes
        )

        # Assign properties from components
        self.split_indices = self._nested_list_to_tuple(
            self._indices_creator.split_indices
        )
        self.dp_indices = self._nested_list_to_tuple(self._indices_creator.dp_indices)
        self.add_indices = self._nested_list_to_tuple(self._indices_creator.add_indices)
        self.arg_indices = self._nested_list_to_tuple(self._indices_creator.arg_indices)
        self.bigX = self._indices_creator.bigX

        self.rv_names = self._metadata_collector.rv_names
        self.log_pdfs = self._nested_list_to_tuple(self._metadata_collector.log_pdfs)
        self.transformations = self._nested_list_to_tuple(
            list(chain.from_iterable(self._metadata_collector.transformations))
        )
        self.num_vi_params = self._metadata_collector.num_vi_params
        self.covariates = self._metadata_collector.covariates
        self.dp_markers = self._metadata_collector.dp_markers
        self.dependencies = self._metadata_collector.dependencies

    def _sort_nodes(self):
        """Sorts nodes by their indices and types."""
        self._ascending_idx_sorted_nodes = sorted(
            self._index_manager.all_nodes, key=lambda node: node._idx
        )

        self._vi_nodes = [
            node
            for node in self._ascending_idx_sorted_nodes
            if node._idx in self._index_manager.vi_idxs
        ]
        self._ascending_idx_sorted_vi_nodes = self._vi_nodes

        self._lp_nodes = [
            node
            for node in self._ascending_idx_sorted_nodes
            if node._idx in self._index_manager.lp_idxs
        ]
        self._ascending_idx_sorted_lp_nodes = self._lp_nodes

    def _nested_list_to_tuple(self, data: list) -> tuple:
        """
        Recursively converts a nested list into a nested tuple with the same structure and elements.
        """
        if isinstance(data, list):
            return tuple(self._nested_list_to_tuple(item) for item in data)
        return data

    def _postprocess_results(self, final_carry, losses, svi_metadata) -> dict:
        """Postprocesses the results of SVI optimization."""
        final_vi_parameters, _, _, _, _ = final_carry
        loc_vi_parameters = final_vi_parameters[0]
        flattened_scale_vi_parameters = final_vi_parameters[1]
        unflattened_scale_vi_parameters = fill_triangular(flattened_scale_vi_parameters)
        num_iterations = len(losses)

        raw_loc_vi_parameters = tuple(jnp.split(loc_vi_parameters, self.split_indices, axis=0))
        num_loc_vi_groups = len(raw_loc_vi_parameters)
        transformed_loc_vi_parameters = jax.tree.map(
            lambda trans, x: trans(x),
            self.transformations[:num_loc_vi_groups],
            raw_loc_vi_parameters,
        )
        raw_loc_vi_parameters_dict = dict(
            zip(self.rv_names[:num_loc_vi_groups], raw_loc_vi_parameters)
        )
        transformed_loc_vi_parameters_dict = dict(
            zip(self.rv_names[:num_loc_vi_groups], transformed_loc_vi_parameters)
        )

        svi_metadata["num_iterations"] = num_iterations

        results_dict = {
            "raw_loc_vi_parameters": raw_loc_vi_parameters_dict,
            "transformed_loc_vi_parameters": transformed_loc_vi_parameters_dict,
            "scale_vi_matrix": unflattened_scale_vi_parameters,
            "losses": losses,
            "svi_metadata": svi_metadata,
        }
        return results_dict

    def run_svi_optimization(
        self,
        optimizer: GradientTransformation,
        vi_dist: Type[VariationalDistribution],
        vi_sample_size: int,
        epochs: int,
        mb_size: Optional[int],
        lr: float,
        max_norm: Optional[float],
        clip_min_max_enabled: bool,
        prng_key: PRNGKey,
    ) -> dict:
        """
        Runs SVI optimization for the model.

        Return dict keys and values are:
        - raw_loc_vi_parameters: dictionary of raw VI parameters
        - transformed_loc_vi_parameters: dictionary of transformed VI parameters
        - scale_vi_matrix: unflattened scale VI matrix
        - losses: list of losses per iteration
        - svi_metadata: dictionary of metadata used in SVI optimization
        """
        mb_prngkey, svi_prngkey = jax.random.split(prng_key)

        vi_dist = vi_dist(self.total_num_vi_params)
        init_vi_parameters, vi_sample_func, vi_log_pdf_func = prepare_vi_dist(
            vi_dist, vi_sample_size
        )

        if mb_size is None:
            mb_size = self.bigX.shape[0]  # full batch

        mb_pointers, masks, responses_padded, design_matrix_padded = (
            prepare_mini_batching(
                self.responses, self.bigX, epochs, mb_size, mb_prngkey
            )
        )

        init_opt_state, prepared_optimizer = prepare_opt_state(
            sgd_method=optimizer,
            lr=lr,
            init_vi_parameters=init_vi_parameters,
            max_norm=max_norm,
            clip_min_max_enabled=clip_min_max_enabled,
        )

        final_carry, losses = core_svi_optimization(
            responses_padded=responses_padded,
            design_matrix_padded=design_matrix_padded,
            mb_pointers=mb_pointers,
            masks=masks,
            joint_log_pdfs_funcs=self.log_pdfs,
            transformations=self.transformations,
            vi_sample_func=vi_sample_func,
            vi_log_pdf_func=vi_log_pdf_func,
            optimizer=prepared_optimizer,
            init_opt_state=init_opt_state,
            init_vi_parameters=init_vi_parameters,
            prng_key=svi_prngkey,
            split_idxs=self.split_indices,
            dp_idxs=self.dp_indices,
            add_idxs=self.add_indices,
            arg_idxs=self.arg_indices,
        )

        svi_metadata = {
            "mb_size": mb_size,
            "lr": lr,
            "max_norm": max_norm,
            "clip_min_max_enabled": clip_min_max_enabled,
            "epochs": epochs,
            "vi_dist": vi_dist.name,
        }

        return self._postprocess_results(final_carry, losses, svi_metadata)

    @property
    def total_num_vi_params(self) -> int:
        """Returns the sum of all VI parameter dimensions collected from nodes."""
        return sum(self.num_vi_params)
