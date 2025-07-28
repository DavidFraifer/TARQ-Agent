from typing import List, Union
from .tools import Tool, ToolGraphBuilder
from .c_layers.C1 import C1
from .config import configure_api_keys
from .utils.logger import TaskLogger

class Agent:
    SUPPORTED_MODELS = ["gpt-4o", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    
    def __init__(self, tools: List[Union[str, Tool]], light_llm: str, heavy_llm: str, enable_logging: bool = False):
        configure_api_keys()
        
        # Validate models
        for model, name in [(light_llm, "light_llm"), (heavy_llm, "heavy_llm")]:
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(f"Unsupported {name}: {model}. Supported: {self.SUPPORTED_MODELS}")
        
        self.tools = tools
        self.light_llm = light_llm
        self.heavy_llm = heavy_llm
        self.enable_logging = enable_logging
        self.logger = TaskLogger() if enable_logging else None
        self.graph = ToolGraphBuilder.build_graph(tools, light_llm, heavy_llm)
        self.c1 = C1(self.graph)
        self.running = False
        
        # Pass logger to C1 if enabled
        if self.logger:
            self.c1.set_logger(self.logger)
        
    def start(self):
        print("[Agent] Starting...")
        if not self.running:
            self.c1.start()
            self.running = True
        
    def stop(self):
        if self.running:
            self.c1.stop()
            self.running = False
    
    def run(self, message: str):
        if not self.running:
            raise RuntimeError("Agent must be started before running tasks. Call agent.start() first.")
        
        self.c1.receive_message(message)
        
    def get_available_tools(self) -> List[str]:
        return [tool.name if isinstance(tool, Tool) else tool for tool in self.tools]
    
    def get_log_stats(self) -> dict:
        """Get logging statistics if logging is enabled."""
        if self.logger:
            return self.logger.get_log_stats()
        return {"logging": "disabled"}
    
    def enable_task_logging(self, log_file: str = "hlr_tasks.log"):
        """Enable task logging with optional custom log file."""
        if not self.enable_logging:
            self.enable_logging = True
            self.logger = TaskLogger(log_file)
            if self.c1:
                self.c1.set_logger(self.logger)
            print(f"📊 [Agent] Logging enabled: {log_file}")
    
    def disable_task_logging(self):
        """Disable task logging."""
        if self.enable_logging:
            self.enable_logging = False
            if self.c1:
                self.c1.set_logger(None)
            self.logger = None
            print("📊 [Agent] Logging disabled")
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
