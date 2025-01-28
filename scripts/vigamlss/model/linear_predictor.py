from .node import Node


class LinearPredictor:
    def __init__(self, term_l, term_r, operation: str):
        self.term_l = term_l
        self.term_r = term_r
        self.operation = operation
        self.term_lclass_type = term_l.node_type
        self.term_rclass_type = term_r.node_type
        self.term_l_node = term_l.node
        self.term_r_node = term_r.node
        # Internal states
        self.node_type = "LinearPredictor"
        # Node related
        self.node = self.setup_lp_node()

    def setup_lp_node(self):
        name_l = self.term_l_node.name
        name_r = self.term_r_node.name
        name = f"{name_l} {self.operation} {name_r}"

        # Handle covariates - always use lists
        covariates = self.term_l_node.covariates + self.term_r_node.covariates

        # Handle parents
        if self.term_lclass_type == "LinearPredictor":
            lp_l_parents = self.term_l_node.parents
            dp_markers_l = self.term_l_node.dp_markers
        elif self.term_lclass_type == "Distribution":
            lp_l_parents = [self.term_l_node]
            dp_markers_l = [self.operation == "@"]
        else:
            lp_l_parents = []
            dp_markers_l = []

        if self.term_rclass_type == "LinearPredictor":
            lp_r_parents = self.term_r_node.parents
            dp_markers_r = self.term_r_node.dp_markers
        elif self.term_rclass_type == "Distribution":
            lp_r_parents = [self.term_r_node]
            dp_markers_r = [self.operation == "@"]
        else:
            lp_r_parents = []
            dp_markers_r = []

        parents = lp_l_parents + lp_r_parents
        dp_markers = dp_markers_l + dp_markers_r

        return Node(
            name=name,
            covariates=covariates,
            parents=parents,
            dp_markers=dp_markers,
        )

    @staticmethod
    def _add_validations(other):
        other_class_type = other.node_type
        if other_class_type == "DesignMatrix":
            raise ValueError("DesignMatrix cannot be added to LinearPredictor.")

    def __add__(self, other):
        return LinearPredictor(self, other, "+")
