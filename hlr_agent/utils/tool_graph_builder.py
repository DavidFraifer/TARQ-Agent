"""Tool Graph Builder - Automatically constructs graphs from tools"""

from typing import List, Union
from ..c_layers.C0.node import Node
from ..c_layers.C0.graph import Graph
from ..tools.tool import Tool
from ..tools.internal_tools import get_internal_tool

def _func_output(graph):
    """Default output function for completing tasks."""
    # Output function now runs silently
    pass

class ToolGraphBuilder:
    """Builds workflow graphs automatically from tool specifications."""
    
    @staticmethod
    def build_graph(tools: List[Union[str, Tool]], light_llm: str, heavy_llm: str) -> Graph:
        """Build a graph from a list of tools with dual LLM setup."""
        if not tools:
            raise ValueError("At least one tool must be specified")
        
        # Convert tool specifications to Tool objects
        tool_objects = []
        for tool in tools:
            if isinstance(tool, str):
                tool_objects.append(get_internal_tool(tool))
            elif isinstance(tool, Tool):
                tool_objects.append(tool)
            else:
                raise ValueError(f"Invalid tool specification: {tool}. Must be string or Tool object.")
        
        # Validate unique names
        tool_names = [tool.name for tool in tool_objects]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique")
        
        # Build nodes
        nodes = [
            # Input node - can connect to any tool or directly to output
            Node("Input", children=tool_names + ["Output"])
        ]
        
        # Tool nodes - each tool can connect to any other tool or output
        for tool in tool_objects:
            # Simplified: all tools can connect to all other tools + output
            all_other_tools = [t.name for t in tool_objects if t.name != tool.name]
            tool_children = tool.children if tool.children else all_other_tools
            tool_children = tool_children + ["Output"]
            
            nodes.append(Node(
                node_id=tool.name,
                children=tool_children,
                func=tool.func,
                description=tool.description
            ))
        
        # Output node - terminal node
        nodes.append(Node(
            node_id="Output",
            children=None,
            func=_func_output,
            description="Complete the task and show results"
        ))
        
        # Create and return graph
        return Graph(
            nodes=nodes,
            start_node_id="Input",
            end_node_id="Output", 
            light_llm=light_llm,
            heavy_llm=heavy_llm
        )
