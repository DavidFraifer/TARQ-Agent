from typing import Callable, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent 

class Node:
    def __init__(
        self,
        node_id: str,
        children: Optional[List[str]],
        func: Optional[Callable[['Agent'], Optional[str]]] = None,
        description: Optional[str] = None
    ):
        self.id = node_id
        self.children = children or []
        self.func = func
        self.description = description 

    def execute(self, agent: 'Agent') -> Optional[str]:
        if self.func is None:
            return None
        return self.func(agent)