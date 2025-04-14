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
        self.nodes: Dict[str, Node] = {node.id: node for node in nodes}
        if start_node_id not in self.nodes:
            raise ValueError(f"El nodo inicial '{start_node_id}' no existe.")
        if end_node_id not in self.nodes:
            raise ValueError(f"El nodo final '{end_node_id}' no existe.")
        if model not in ["gemini-2.0-flash", "gpt-4o"]:
            raise ValueError(f"Modelo '{model}' no permitido. Usa 'gemini-2.0-flash' o 'gpt-4o'.")

        self.model = model
        self.current_id: str = start_node_id
        self.end_node_id = end_node_id
        self.history: List[str] = []
        self.context: dict = {}
        self.api_key = api_key
        
        # Store the user_message to pass along to the LLM
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
                            extra_context=self.context.get("context", "")  # Pass additional context if available
                        )
                else:
                    self.current_id = self.end_node_id