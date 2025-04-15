from typing import Dict, List
from .node import Node
from .llm import get_next_node

class Agent:
    def __init__(
        self,
        nodes: List[Node],
        start_node_id: str,
        end_node_id: str,
        model: str,
        api_key: str,
        user_message: str
    ):
        # Validate mandatory fields

        if not nodes or not isinstance(nodes, list):
            raise ValueError("The 'nodes' field is required and must be a non-empty list.")
        
        # For string fields, also check that they are not empty after strip()
        if not start_node_id or not start_node_id.strip():
            raise ValueError("The 'start_node_id' field is required and cannot be empty.")
        if not end_node_id or not end_node_id.strip():
            raise ValueError("The 'end_node_id' field is required and cannot be empty.")
        if not model or not model.strip():
            raise ValueError("The 'model' field is required and cannot be empty.")
        if not api_key or not api_key.strip():
            raise ValueError("The 'api_key' field is required and cannot be empty.")
        if not user_message or not user_message.strip():
            raise ValueError("The 'user_message' field is required and cannot be empty.")

        # Validate that all node IDs are unique
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node IDs found. All node IDs must be unique.")

        self.nodes: Dict[str, Node] = {node.id: node for node in nodes}
        if start_node_id not in self.nodes:
            raise ValueError(f"The start node '{start_node_id}' does not exist.")
        if end_node_id not in self.nodes:
            raise ValueError(f"The end node '{end_node_id}' does not exist.")
        if model not in ["gemini-2.0-flash", "gpt-4o"]:
            raise ValueError(f"Model '{model}' is not allowed. Use 'gemini-2.0-flash' or 'gpt-4o'.")

        self.model = model
        self.current_id: str = start_node_id
        self.end_node_id = end_node_id
        self.history: List[str] = []
        self.context: dict = {}
        self.api_key = api_key
        self.user_message = user_message

    def run(self, steps: int = 100):
        for _ in range(steps):
            if self.current_id is None:
                break
            current_node = self.nodes[self.current_id]
            self.history.append(self.current_id)

            explicit_next = current_node.execute(self)
            if explicit_next is not None:
                self.current_id = explicit_next
            else:
                if not current_node.children:
                    self.current_id = None
                    break

                # Include end_node_id even without description
                filtered = [
                    (child_id, self.nodes[child_id].description)
                    for child_id in current_node.children
                    if (self.nodes[child_id].description not in (None, "")) or (child_id == self.end_node_id)
                ]
                if filtered:
                    if len(filtered) == 1:
                        self.current_id = filtered[0][0]
                    else:
                        filtered_ids, filtered_descriptions = zip(*filtered)
                        self.current_id = get_next_node(
                            list(filtered_ids),
                            list(filtered_descriptions),
                            model=self.model,
                            api_key=self.api_key,
                            user_message=self.user_message,
                            extra_context=self.context.get("context", "")
                        )
                else:
                    self.current_id = self.end_node_id