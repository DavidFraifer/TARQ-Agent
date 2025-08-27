from typing import Callable, Optional, Dict, Any
import asyncio

class Tool:
    """Tool class for backward compatibility - now works with simplified architecture"""
    def __init__(self, name: str, func: Callable[[str], Optional[str]], description: str):
        if not name or not name.strip():
            raise ValueError("Tool name is required and cannot be empty.")
        if not description or not description.strip():
            raise ValueError("Tool description is required and cannot be empty.")
        if not callable(func):
            raise ValueError("Tool function must be callable.")
            
        self.name = name.strip()
        self.func = func
        self.description = description.strip()
        
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
        
    async def execute_tool(self, name: str, context: str = "", task_id: str = None, task_memory = None) -> str:
        """Execute a tool with the given context"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        tool_func = self.tools[name]
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                try:
                    result = await tool_func(context, task_id=task_id, task_memory=task_memory)
                except TypeError:
                    try:
                        result = await tool_func(context, task_id=task_id)
                    except TypeError:
                        result = await tool_func(context)
            else:
                try:
                    result = tool_func(context, task_id=task_id, task_memory=task_memory)
                except TypeError:
                    try:
                        result = tool_func(context, task_id=task_id)
                    except TypeError:
                        result = tool_func(context)
            
            return str(result) if result is not None else "Tool completed successfully"
            
        except Exception as e:
            raise Exception(f"Tool '{name}' execution failed: {str(e)}")
