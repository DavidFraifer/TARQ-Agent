from typing import List, Union, Optional
from .tools import Tool
from .core.orchestrator import Orchestrator
from .config import configure_api_keys
from .utils.logger import TARQLogger
from .utils.console import console
import uuid
import time
import os
from datetime import datetime

class Agent:
    SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    
    def __init__(self, tools: List[Union[str, Tool]], light_llm: str, heavy_llm: str, agent_id: Optional[str] = None, disable_delegation: bool = False, context: Optional[List[str]] = None):
        configure_api_keys()
        
        # Generate agent ID if not provided
        self.agent_id = agent_id or f"agent-{str(uuid.uuid4())[:6]}"
        
        for model, name in [(light_llm, "light_llm"), (heavy_llm, "heavy_llm")]:
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(f"Unsupported {name}: {model}. Supported: {self.SUPPORTED_MODELS}")
        
        self.tools = tools
        self.light_llm = light_llm
        self.heavy_llm = heavy_llm
        self.logger = TARQLogger()  # Always enable logging for token tracking
        
        # Initialize RAG engine only if context documents are provided
        self.rag = None
        if context:
            console.info("RAG", "Context documents provided, initializing RAG engine...", agent_id=self.agent_id)
            try:
                from .rag import RAGEngine
                self.rag = RAGEngine()
                
                # Only ingest documents if RAG engine was successfully initialized
                if self.rag.is_enabled():
                    console.info("RAG", "Ingesting context documents...", agent_id=self.agent_id)
                    for doc_path in context:
                        self.rag.ingest_file(doc_path)
                else:
                    console.warning("RAG", "RAG engine failed to initialize, continuing without context", agent_id=self.agent_id)
                    self.rag = None
            except ImportError as e:
                console.warning("RAG", f"Failed to import RAG engine: {e}", agent_id=self.agent_id)
                self.rag = None
        
        self.orchestrator = Orchestrator(logger=self.logger, light_llm=light_llm, heavy_llm=heavy_llm, agent_id=self.agent_id, disable_delegation=disable_delegation, rag_engine=self.rag)
        self.running = False
        self.last_called = None  # Track when the agent was last used

        self._configure_tools()
    
    def _configure_tools(self):
        """Configure which tools are available to the orchestrator"""
        if not self.tools:
            raise ValueError("Tools list cannot be empty. Please provide at least one tool.")
        
        self.orchestrator.tools.tools.clear()
        from .tools.internal_tools import internal_tools
        
        for tool in self.tools:
            if isinstance(tool, Tool):
                # Tool wrapper with name and function - pass description
                self.orchestrator.add_tool(tool.name, tool.func, tool.description)
            elif isinstance(tool, str) and tool in internal_tools:
                # Built-in tool by name - no description needed
                self.orchestrator.add_tool(tool, internal_tools[tool])
            elif callable(tool):
                # Raw function - use function name as tool name, no description
                tool_name = getattr(tool, '__name__', str(tool))
                self.orchestrator.add_tool(tool_name, tool)
            else:
                console.warning(f"Tool '{tool}' is not supported. Use Tool() wrapper, built-in tool name, or raw function", agent_id=self.agent_id)
    
    def add_tool(self, name: str, func):
        """Add a tool to the agent"""
        self.orchestrator.add_tool(name, func)
        
    def start(self):
        if not self.running:
            self.orchestrator.start()
            self.running = True
            available_tools = list(self.orchestrator.tools.tools.keys())
            console.system("Agent started", f"ID: {self.agent_id} | Available tools: {', '.join(available_tools)}", agent_id=self.agent_id)
        
    def stop(self, task_id: Optional[str] = None):
        """Stop a specific task or the entire agent
        
        Args:
            task_id (Optional[str]): If provided, stops the specific task. 
                                   If None or empty, stops the entire agent.
        """
        if task_id:
            # Stop specific task
            if self.logger and task_id in self.logger.active_tasks:
                # Mark task as stopped and log completion
                console.info("Task stopped", f"Task {task_id} manually stopped", task_id=task_id, agent_id=self.agent_id)
                self.logger.complete_task(task_id, "stopped", 0.0)
                console.success("Task termination", f"Task {task_id} has been stopped", agent_id=self.agent_id)
            else:
                # Task not found or logging not enabled
                if not self.logger:
                    console.warning("Task stop failed", "Logging not enabled - cannot track specific tasks", agent_id=self.agent_id)
                else:
                    console.warning("Task stop failed", f"Task {task_id} not found in active tasks", agent_id=self.agent_id)
        else:
            # Stop entire agent
            if self.running:
                # Stop all active tasks first
                if self.logger and self.logger.active_tasks:
                    active_task_ids = list(self.logger.active_tasks.keys())
                    console.info("Agent shutdown", f"Stopping {len(active_task_ids)} active tasks", agent_id=self.agent_id)
                    
                    for active_task_id in active_task_ids:
                        console.debug("Task cleanup", f"Stopping task {active_task_id}", task_id=active_task_id, agent_id=self.agent_id)
                        self.logger.complete_task(active_task_id, "stopped", 0.0)
                
                # Stop the orchestrator
                self.orchestrator.stop()
                self.running = False
                console.system("Agent stopped", f"ID: {self.agent_id} - All tasks terminated", agent_id=self.agent_id)
            else:
                console.warning("Agent stop", f"Agent {self.agent_id} is not currently running", agent_id=self.agent_id)
    
    def run(self, message: str):
        """Run a task with the agent"""
        if not self.running:
            raise RuntimeError(f"Agent {self.agent_id} must be started before running tasks. Call agent.start() first.")
        
        self.last_called = datetime.now()
        self.orchestrator.receive_message(message)
        
    def info(self) -> dict:
        """Get comprehensive information about the agent's current state"""
        # Basic agent information
        info = {
            "agent_id": self.agent_id,
            "running": self.running,
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "creation_time": datetime.now().isoformat(),  # Could be tracked if needed
            "models": {
                "light_llm": self.light_llm,
                "heavy_llm": self.heavy_llm
            },
            "logging_enabled": self.logger is not None
        }
        
        # Tool information
        info["tools"] = {
            "available_tools": self.get_available_tools(),
            "tool_count": len(self.get_available_tools())
        }
        
        # Current tasks information
        if self.logger and hasattr(self.logger, 'active_tasks'):
            active_tasks = self.logger.active_tasks
            info["current_tasks"] = {
                "active_count": len(active_tasks),
                "tasks": []
            }
            
            for task_id, task_data in active_tasks.items():
                task_info = {
                    "task_id": task_id,
                    "message": task_data.get("message", "")[:100] + ("..." if len(task_data.get("message", "")) > 100 else ""),
                    "start_time": task_data.get("start_datetime", ""),
                    "status": task_data.get("status", "unknown"),
                    "tokens_used": task_data.get("tokens_used", 0),
                    "llm_calls": task_data.get("llm_calls", 0)
                }
                info["current_tasks"]["tasks"].append(task_info)
        else:
            info["current_tasks"] = {
                "active_count": 0,
                "tasks": [],
                "note": "Logging not enabled - cannot track active tasks"
            }
        
        # Queue information
        if hasattr(self.orchestrator, 'message_queue'):
            queue_size = self.orchestrator.message_queue.qsize()
            info["message_queue"] = {
                "pending_messages": queue_size,
                "queue_empty": queue_size == 0
            }
        
        # Team information
        if hasattr(self.orchestrator, 'team') and getattr(self.orchestrator, 'team'):
            team = self.orchestrator.team
            info["team"] = {
                "is_team_member": True,
                "team_name": getattr(team, 'name', 'Unknown'),
                "team_id": getattr(team, 'team_id', 'Unknown'),
                "team_running": getattr(team, 'running', False),
                "team_agent_count": len(getattr(team, 'agents', {}))
            }
            
            # Add team members info
            if hasattr(team, 'agents'):
                team_members = []
                for member_name, member_agent in team.agents.items():
                    member_info = {
                        "name": member_name,
                        "agent_id": getattr(member_agent, 'agent_id', 'Unknown'),
                        "running": getattr(member_agent, 'running', False),
                        "tools": getattr(member_agent, 'get_available_tools', lambda: [])()
                    }
                    team_members.append(member_info)
                info["team"]["members"] = team_members
        else:
            info["team"] = {
                "is_team_member": False,
                "note": "Agent is not part of a team"
            }
        
        # RAG information
        if self.rag:
            info["rag"] = {
                "enabled": self.rag.is_enabled() if hasattr(self.rag, 'is_enabled') else True,
                "context_available": True
            }
        else:
            info["rag"] = {
                "enabled": False,
                "context_available": False
            }
            
        return info
        
    def get_agent_id(self) -> str:
        """Get the agent ID"""
        return self.agent_id
        
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.orchestrator.tools.tools.keys())
    
    @staticmethod
    def get_all_internal_tools() -> List[str]:
        """Get list of all internal tools that can be used"""
        from .tools.internal_tools import internal_tools
        return list(internal_tools.keys())
    
    def get_log_stats(self) -> dict:
        """Get logging statistics"""
        if self.logger:
            return self.logger.get_log_stats()
        return {"logging": "disabled"}
    
    def __str__(self) -> str:
        return f"Agent(id='{self.agent_id}', tools={len(self.tools)}, running={self.running})"
    
    def __repr__(self) -> str:
        return self.__str__()
