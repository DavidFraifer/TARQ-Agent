from typing import List, Union, Optional
from .tools import Tool
from .core.orchestrator import Orchestrator
from .config import configure_api_keys
from .utils.logger import TARQLogger
from .utils.console import console
import uuid
import time

class Agent:
    SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    
    def __init__(self, tools: List[Union[str, Tool]], light_llm: str = "gpt-5-nano", heavy_llm: str = "gpt-5-nano", enable_logging: bool = False, agent_id: Optional[str] = None):
        configure_api_keys()
        
        # Generate agent ID if not provided
        self.agent_id = agent_id or f"agent-{str(uuid.uuid4())[:6]}"
        
        for model, name in [(light_llm, "light_llm"), (heavy_llm, "heavy_llm")]:
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(f"Unsupported {name}: {model}. Supported: {self.SUPPORTED_MODELS}")
        
        self.tools = tools
        self.light_llm = light_llm
        self.heavy_llm = heavy_llm
    # role removed; tools describe agent capabilities
        self.logger = TARQLogger() if enable_logging else None
        self.orchestrator = Orchestrator(logger=self.logger, light_llm=light_llm, heavy_llm=heavy_llm, agent_id=self.agent_id)
        self.running = False

        self._configure_tools()
    
    def _configure_tools(self):
        """Configure which tools are available to the orchestrator"""
        if not self.tools:
            raise ValueError("Tools list cannot be empty. Please provide at least one tool.")
        
        self.orchestrator.tools.tools.clear()
        from .tools.internal_tools import internal_tools
        
        for tool in self.tools:
            if isinstance(tool, Tool):
                self.orchestrator.add_tool(tool.name, tool.func)
            elif isinstance(tool, str) and tool in internal_tools:
                self.orchestrator.add_tool(tool, internal_tools[tool])
            else:
                console.warning(f"Tool '{tool}' not found in internal tools")
    
    def add_tool(self, name: str, func):
        """Add a tool to the agent"""
        self.orchestrator.add_tool(name, func)
        
    def start(self):
        if not self.running:
            self.orchestrator.start()
            self.running = True
            available_tools = list(self.orchestrator.tools.tools.keys())
            console.system("Agent started", f"ID: {self.agent_id} | Available tools: {', '.join(available_tools)}")
        
    def stop(self):
        if self.running:
            self.orchestrator.stop()
            self.running = False
    
    def run(self, message: str):
        """Run a task with the agent"""
        if not self.running:
            raise RuntimeError(f"Agent {self.agent_id} must be started before running tasks. Call agent.start() first.")
        
        self.orchestrator.receive_message(message)
        
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
