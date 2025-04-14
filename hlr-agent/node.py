from typing import Callable, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent  # solo se importa para comprobación de tipos

class Node:
    def __init__(
        self,
        node_id: str,
        children: Optional[List[str]],
        func: Optional[Callable[['Agent'], Optional[str]]] = None,
        description: Optional[str] = None
    ):
        self.id = node_id
        self.children = children or []  # Si children es None, se asigna una lista vacía
        self.func = func
        self.description = description  # Puede ser None

    def execute(self, agent: 'Agent') -> Optional[str]:
        if self.func is None:
            return None
        return self.func(agent)