from typing import Callable, Optional, Dict, Any
from ..utils import report_error, raise_error
import asyncio

class Tool:
    """Tool class for backward compatibility - now works with simplified architecture"""
    def __init__(self, name: str, func: Callable[[str], Optional[str]], description: str):
        if not name or not name.strip():
            raise_error("VAL-001", context={"field": "name", "value": name})
        if not description or not description.strip():
            raise_error("VAL-001", context={"field": "description", "value": description})
        if not callable(func):
            raise_error("VAL-002", context={"field": "func", "value_type": type(func).__name__})
            
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
            raise_error("TL-002", context={"name": name, "func_type": type(func).__name__})
        self.tools[name.strip()] = func
        
    async def execute_tool(self, name: str, context: str = "", task_id: str = None, task_memory = None, light_llm: str = "gpt-4o-mini", heavy_llm: str = "gpt-4o", agent_id: str = None, validation_mode: bool = False) -> str:
        """Execute a tool with the given context"""
        if name not in self.tools:
            raise_error("TL-003", context={"tool_name": name, "available_tools": list(self.tools.keys())})
        
        tool_func = self.tools[name]
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                try:
                    # Try with full parameters first
                    result = await tool_func(context, task_id=task_id, task_memory=task_memory, light_llm=light_llm, heavy_llm=heavy_llm, agent_id=agent_id, validation_mode=validation_mode)
                except TypeError:
                    try:
                        result = await tool_func(context, task_id=task_id, task_memory=task_memory, light_llm=light_llm, heavy_llm=heavy_llm, agent_id=agent_id)
                    except TypeError:
                        try:
                            result = await tool_func(context, task_id=task_id, task_memory=task_memory, agent_id=agent_id)
                        except TypeError:
                            try:
                                result = await tool_func(context, task_id=task_id)
                            except TypeError:
                                result = await tool_func(context)
            else:
                try:
                    # Try with full parameters first
                    result = tool_func(context, task_id=task_id, task_memory=task_memory, light_llm=light_llm, heavy_llm=heavy_llm, agent_id=agent_id, validation_mode=validation_mode)
                except TypeError:
                    try:
                        result = tool_func(context, task_id=task_id, task_memory=task_memory, light_llm=light_llm, heavy_llm=heavy_llm, agent_id=agent_id)
                    except TypeError:
                        try:
                            result = tool_func(context, task_id=task_id, task_memory=task_memory, agent_id=agent_id)
                        except TypeError:
                            try:
                                result = tool_func(context, task_id=task_id)
                            except TypeError:
                                result = tool_func(context)
            
            return str(result) if result is not None else "Tool completed successfully"
            
        except Exception as e:
            raise_error("TL-004", context={"tool_name": name, "error": str(e), "error_type": type(e).__name__})
