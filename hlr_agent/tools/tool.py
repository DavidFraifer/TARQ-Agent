from typing import Callable, Optional, Dict, Any
import asyncio

class Tool:
    """Tool class for backward compatibility - now works with simplified architecture"""
    def __init__(self, name: str, func: Callable[[str], Optional[str]], description: str, children: Optional[list] = None):
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


class ToolContainer:
    """Container for managing tools in the simplified architecture"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def add_tool(self, name: str, func: Callable):
        """Add a tool to the container"""
        if not name or not callable(func):
            raise ValueError("Tool name and callable function required")
        self.tools[name.strip()] = func
        
    def remove_tool(self, name: str):
        """Remove a tool from the container"""
        if name in self.tools:
            del self.tools[name]
            
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self.tools
        
    async def execute_tool(self, name: str, context: str = "", task_id: str = None) -> str:
        """Execute a tool with the given context"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        tool_func = self.tools[name]
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                # Try to pass task_id if the function accepts it
                try:
                    result = await tool_func(context, task_id=task_id)
                except TypeError:
                    # Fallback if function doesn't accept task_id
                    result = await tool_func(context)
            else:
                # Try to pass task_id if the function accepts it
                try:
                    result = tool_func(context, task_id=task_id)
                except TypeError:
                    # Fallback if function doesn't accept task_id
                    result = tool_func(context)
            
            return str(result) if result is not None else "Tool completed successfully"
            
        except Exception as e:
            raise Exception(f"Tool '{name}' execution failed: {str(e)}")
    
    def get_tool_names(self) -> list:
        """Get list of available tool names"""
        return list(self.tools.keys())
