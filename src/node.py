class TreeNode:
    def __init__(self, query="", answer="", parent=None, num_children=5):
        self.query = query  # Current action
        self.answer = answer  # Current state extra info
        self.num_children = (
            num_children if parent is None else parent.num_children - 1
        )  # Number of possible actions
        self.state = f"Original question: {query}\n"  # Current state
        self.parent = parent  # Parent node
        self.children = {}  # Child nodes mapping action to node
        self.visit_count = 0  # Number of times node was visited
        self.value = 0  # Node value
        self.depth = 0 if parent is None else parent.depth + 1  # Depth in tree
        self.is_fully_expanded = False  # Whether all children are expanded
        self.is_terminal = self.depth >= 3  # Whether node is terminal

    def append_children(self, sub_query: str, sub_answer: str):
        node = TreeNode(sub_query, sub_answer, self, self.num_children - 1)
        node.update_state_from_parent()
        self.children.update({sub_query: node})
        return self

    def update_state_from_parent(self):
        if self.parent is not None:
            self.state = self.parent.state + f"Intermediate answer: {self.answer}\n"
        else:
            self.state = f"Original question: {self.query}\n"

    def update_value(self, value):
        self.value = value

    def to_dict(self):
        return {
            "query": self.query,
            "answer": self.answer,
            "visit_count": self.visit_count,
            "value": self.value,
            "children": list(self.children.keys()),
            "depth": self.depth,
            "is_fully_expanded": self.is_fully_expanded,
            "is_terminal": self.is_terminal,
        }
