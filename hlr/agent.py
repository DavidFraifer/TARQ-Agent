from typing import Dict, List, Optional
from .node import Node


class Agent:
    def __init__(self, nodes: List[Node], start_node_id: str):
        self.nodes: Dict[str, Node] = {node.id: node for node in nodes}
        if start_node_id not in self.nodes:
            raise ValueError(f"El nodo inicial '{start_node_id}' no existe.")
        self.current_id: Optional[str] = start_node_id
        self.history: List[str] = []
        self.context: Dict[str, any] = {}

    def run(self, steps: int = 100):
        for _ in range(steps):
            if self.current_id is None:
                break
            self.history.append(self.current_id)
            current_node = self.nodes[self.current_id]
            self.current_id = current_node.execute(self)
        else:
            print("Se alcanzó el límite de pasos.")
