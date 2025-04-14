from typing import Callable, Optional
from .agent import Agent  # para anotaciones circulares (o usa string tipo "Agent")


class Node:
    def __init__(self, node_id: str, default_next: Optional[str], func: Callable[['Agent'], Optional[str]]):
        self.id = node_id
        self.default_next = default_next
        self.func = func

    def execute(self, agent: 'Agent') -> Optional[str]:
        next_node = self.func(agent)
        return next_node if next_node is not None else self.default_next
