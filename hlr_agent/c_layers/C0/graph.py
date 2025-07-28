from typing import Dict, List, Optional
import copy
from .node import Node
from ...internal.llm import get_next_node_async

class Graph:
    SUPPORTED_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gpt-4o"]
    
    def __init__(self, nodes: List[Node], start_node_id: str, end_node_id: str, light_llm: str, heavy_llm: str):
        self._validate_inputs(nodes, start_node_id, end_node_id, light_llm, heavy_llm)

        # Build node dictionary and validate IDs
        self.nodes: Dict[str, Node] = {node.id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("Duplicate node IDs found. All node IDs must be unique.")
        
        # Validate node existence
        for node_id in [start_node_id, end_node_id]:
            if node_id not in self.nodes:
                raise ValueError(f"Node '{node_id}' does not exist.")

        # Store both LLM models
        self.light_llm = light_llm  # For routing and basic decisions
        self.heavy_llm = heavy_llm  # For complex tool execution
        self.start_node_id = start_node_id 
        self.end_node_id = end_node_id
        
        # Runtime state
        self.current_id: Optional[str] = start_node_id
        self.history: List[str] = []
        self.context: dict = {"route": f"NODE HISTORY: {start_node_id}", "info": ""}
        self.user_message: Optional[str] = None
        self._memory_ref = None
        
        # Token tracking
        self.total_tokens_used = 0
        self.logger = None
        self.current_task_id = None
    
    def _validate_inputs(self, nodes, start_node_id, end_node_id, light_llm, heavy_llm):
        """Validate constructor inputs."""
        if not nodes or not isinstance(nodes, list):
            raise ValueError("'nodes' must be a non-empty list.")
        
        for field, value in [("start_node_id", start_node_id), ("end_node_id", end_node_id), 
                           ("light_llm", light_llm), ("heavy_llm", heavy_llm)]:
            if not value or not value.strip():
                raise ValueError(f"'{field}' is required and cannot be empty.")
        
        for model, name in [(light_llm, "light_llm"), (heavy_llm, "heavy_llm")]:
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(f"{name} '{model}' not supported. Use: {self.SUPPORTED_MODELS}")
    
    def reset(self):
        """Reset the graph to its initial state for a new run."""
        self.current_id = self.start_node_id
        self.history = []
        self.context = {"route": f"NODE HISTORY: {self.start_node_id}", "info": self.context.get("info", "")}
        self.user_message = None
        self.total_tokens_used = 0
    
    def set_logger(self, logger, task_id: str):
        """Set logger for token tracking."""
        self.logger = logger
        self.current_task_id = task_id
    
    def add_tokens(self, tokens: int):
        """Add tokens to the total count and logger."""
        self.total_tokens_used += tokens
        if self.logger and self.current_task_id:
            self.logger.add_tokens(self.current_task_id, tokens)
        
    def copy(self):
        """Create a deep copy of the graph for independent task execution."""
        nodes_copy = [
            Node(node_id=node.id, children=node.children.copy() if node.children else None, 
                 func=node.func, description=node.description)
            for node in self.nodes.values()
        ]
        
        graph_copy = Graph(nodes=nodes_copy, start_node_id=self.start_node_id, 
                          end_node_id=self.end_node_id, light_llm=self.light_llm, heavy_llm=self.heavy_llm)
        graph_copy.context = copy.deepcopy(self.context)
        # Reset token tracking for new copy
        graph_copy.total_tokens_used = 0
        return graph_copy
        
    async def run(self, user_message: str, steps: int = 100):
        """Run the graph workflow with the given user message."""
        if not user_message or not user_message.strip():
            raise ValueError("'user_message' is required and cannot be empty.")

        self.reset()
        self.user_message = user_message
        self.context["route"] = f"NODE HISTORY: {self.current_id}"
        
        for _ in range(steps):
            if self.current_id is None:
                break

            current_node = self.nodes[self.current_id]
            self.history.append(self.current_id)
            
            # Execute current node function
            explicit_next = current_node.execute(self)
            
            # Determine next node
            next_node_id = await self._get_next_node(current_node, explicit_next)
            
            # Update state
            if next_node_id:
                self.context["route"] += f" -> {next_node_id}"
                self.current_id = next_node_id
            else:
                self.current_id = None
                
    async def _get_next_node(self, current_node, explicit_next):
        """Determine the next node to transition to."""
        # If node explicitly returns next node
        if explicit_next is not None:
            if explicit_next not in self.nodes:
                print(f"Warning: Node '{current_node.id}' returned invalid node ID '{explicit_next}'. Ending run.")
                return None
            return explicit_next
        
        # If no children, end execution
        if not current_node.children:
            return None
        
        # Filter valid children (have descriptions or are end node)
        valid_children = [
            (child_id, self.nodes[child_id].description)
            for child_id in current_node.children
            if self.nodes[child_id].description or child_id == self.end_node_id
        ]
        
        if not valid_children:
            print(f"Warning: No valid children for node '{current_node.id}', defaulting to end node.")
            return self.end_node_id
        
        if len(valid_children) == 1:
            return valid_children[0][0]
        
        # Use LLM to choose from multiple options
        return await self._llm_choose_node(valid_children)
    
    async def _llm_choose_node(self, valid_children):
        """Use LLM to choose from multiple valid children."""
        children_ids, children_descriptions = zip(*valid_children)
        
        # Prepare context for LLM
        route_str = self.context.get("route", "")
        memory_str = self._memory_ref.get() if self._memory_ref else ""
        llm_context = f"Route: {route_str}\nInfo: {memory_str}" if route_str or memory_str else ""
        
        try:
            chosen_id, tokens = await get_next_node_async(
                list(children_ids), list(children_descriptions), 
                self.light_llm, self.user_message, llm_context
            )
            
            # Track tokens
            self.add_tokens(tokens)
            
            if chosen_id not in children_ids:
                print(f"Warning: LLM returned invalid node ID '{chosen_id}'. Options were: {children_ids}. Defaulting to end node.")
                return self.end_node_id
            
            return chosen_id
            
        except Exception as e:
            print(f"Warning: LLM error: {e}. Defaulting to end node.")
            return self.end_node_id
