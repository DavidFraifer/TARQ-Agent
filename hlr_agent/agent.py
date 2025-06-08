from typing import Dict, List, Optional
from .node import Node
from .llm import get_next_node

class Agent:
    def __init__(
        self,
        nodes: List[Node],
        start_node_id: str,
        end_node_id: str,
        model: str,
        api_key: str,
    ):
        if not nodes or not isinstance(nodes, list):
            raise ValueError("The 'nodes' field is required and must be a non-empty list.")
        if not start_node_id or not start_node_id.strip():
            raise ValueError("The 'start_node_id' field is required and cannot be empty.")
        if not end_node_id or not end_node_id.strip():
            raise ValueError("The 'end_node_id' field is required and cannot be empty.")
        if not model or not model.strip():
            raise ValueError("The 'model' field is required and cannot be empty.")
        if not api_key or not api_key.strip():
            raise ValueError("The 'api_key' field is required and cannot be empty.")

        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node IDs found. All node IDs must be unique.")

        self.nodes: Dict[str, Node] = {node.id: node for node in nodes}
        if start_node_id not in self.nodes:
            raise ValueError(f"The start node '{start_node_id}' does not exist.")
        if end_node_id not in self.nodes:
            raise ValueError(f"The end node '{end_node_id}' does not exist.")
        if model not in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gpt-4o"]:
            raise ValueError(f"Model '{model}' is not allowed. Use 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-flash-8b' or 'gpt-4o'.")

        self.model = model
        self.current_id: Optional[str] = start_node_id
        self.end_node_id = end_node_id
        self.history: List[str] = []
        self.context: dict = {}
        self.api_key = api_key
        self.user_message: Optional[str] = None
        self.context["route"] = f"NODE HISTORY: {start_node_id}"
        self.context["info"] = ""
        

    def run(self, user_message: str, steps: int = 100):
        if not user_message or not user_message.strip():
            raise ValueError("The 'user_message' field is required for run() and cannot be empty.")
        self.user_message = user_message
        self.context["route"] = f"NODE HISTORY: {self.current_id}"

        for _ in range(steps):
            if self.current_id is None:
                break

            current_node_id_for_step = self.current_id
            current_node = self.nodes[current_node_id_for_step]
            self.history.append(current_node_id_for_step)
            explicit_next = current_node.execute(self)
            next_node_id: Optional[str] = None

            if explicit_next is not None:
                if explicit_next not in self.nodes:
                    print(f"Warning: Node '{current_node_id_for_step}' returned non-existent node ID '{explicit_next}'. Ending run.")
                    next_node_id = None
                else:
                    next_node_id = explicit_next
            else:
                if not current_node.children:
                    next_node_id = None
                else:
                    filtered = [
                        (child_id, self.nodes[child_id].description)
                        for child_id in current_node.children
                        if (self.nodes[child_id].description not in (None, "")) or (child_id == self.end_node_id)
                    ]

                    if not filtered:
                        print(f"Warning: No valid children found for node '{current_node_id_for_step}', defaulting to end node.")
                        next_node_id = self.end_node_id
                    elif len(filtered) == 1:
                        next_node_id = filtered[0][0]
                    else:
                        filtered_ids, filtered_descriptions = zip(*filtered)
                        if self.user_message is None:
                            raise ValueError("Internal error: user_message is None during LLM call.")

                        route_str = self.context.get("route", "Route info unavailable")
                        info_str = self.context.get("info", "No additional info")
                        llm_context = f"Current Route: {route_str}\nAdditional Info: {info_str}"

                        llm_chosen_id = get_next_node(
                            list(filtered_ids),
                            list(filtered_descriptions),
                            model=self.model,
                            api_key=self.api_key,
                            user_message=self.user_message,
                            extra_context=llm_context
                        )

                        if llm_chosen_id not in [fid for fid, _ in filtered]:
                            print(f"Warning: LLM returned invalid node ID '{llm_chosen_id}' not in filtered options {filtered_ids}. Defaulting to end node.")
                            next_node_id = self.end_node_id
                        else:
                            next_node_id = llm_chosen_id

            if next_node_id is not None:
                self.context["route"] += f" -> {next_node_id}"
                self.current_id = next_node_id
            else:
                self.current_id = None
