"""Tool class for defining custom tools in HLR"""

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..c_layers.C0.graph import Graph

class Tool:
    """
    Represents a tool that can be used in the HLR workflow.
    
    Args:
        name: Unique identifier for the tool
        func: Function to execute when this tool is used
        description: Description for LLM routing decisions
        children: List of tool names this tool can transition to (None = can go to any tool)
    """
    
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
        self.children = children  # None means can connect to any tool
        
    def __repr__(self):
        return f"Tool(name='{self.name}', description='{self.description}')"
