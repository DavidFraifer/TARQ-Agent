from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..c_layers.C0.graph import Graph

class Tool:
    def __init__(
        self,
        name: str,
        func: Callable[['Graph'], Optional[str]],
        description: str,
        children: Optional[list] = None
    ):
        if not name or not name.strip():
            raise ValueError("Tool name is required and cannot be empty.")
        if not description or not description.strip():
            raise ValueError("Tool description is required and cannot be empty.")
        if not callable(func):
            raise ValueError("Tool function must be callable.")
            
        self.name = name.strip()
        self.func = func
        self.description = description.strip()
        self.children = children
        
    def __repr__(self):
        return f"Tool(name='{self.name}', description='{self.description}')"
