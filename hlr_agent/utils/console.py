# console.py - Professional Console Output System
import time
import threading
from typing import Optional
from enum import Enum

class Colors:
    """ANSI color codes for console output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

class LogLevel(Enum):
    """Log levels for different types of messages"""
    INFO = "INFO"
    SUCCESS = "SUCCESS" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    TASK = "TASK"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"

class ProfessionalConsole:
    """Professional console output with colors and animations"""
    
    def __init__(self, enable_colors: bool = True):
        self.enable_colors = enable_colors
        self._lock = threading.Lock()
        
        # Define color schemes for different log levels
        self.level_colors = {
            LogLevel.INFO: Colors.CYAN,
            LogLevel.SUCCESS: Colors.GREEN,
            LogLevel.WARNING: Colors.YELLOW,
            LogLevel.ERROR: Colors.RED,
            LogLevel.DEBUG: Colors.BRIGHT_BLACK,
            LogLevel.TASK: Colors.BLUE,
            LogLevel.SYSTEM: Colors.MAGENTA,
            LogLevel.TOOL: Colors.YELLOW
        }
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled"""
        if not self.enable_colors:
            return text
        return f"{color}{text}{Colors.RESET}"
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        return time.strftime("%H:%M:%S")
    
    def _format_message(self, level: LogLevel, message: str, details: Optional[str] = None, task_id: Optional[str] = None) -> str:
        """Format a message with timestamp and level"""
        timestamp = self._colorize(self._get_timestamp(), Colors.DIM)
        level_color = self.level_colors.get(level, Colors.WHITE)
        level_text = self._colorize(f"[{level.value:^7}]", level_color + Colors.BOLD)
        
        # Add task ID if provided
        if task_id:
            task_text = self._colorize(f"[{task_id}]", Colors.BRIGHT_BLUE)
            formatted = f"{timestamp} {level_text} {task_text} {message}"
        else:
            formatted = f"{timestamp} {level_text} {message}"
        
        if details:
            details_colored = self._colorize(details, Colors.DIM)
            formatted += f" {details_colored}"
            
        return formatted
    
    def print(self, level: LogLevel, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print a formatted message"""
        with self._lock:
            formatted = self._format_message(level, message, details, task_id)
            print(formatted)
    
    def info(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print info message"""
        self.print(LogLevel.INFO, message, details, task_id)
    
    def success(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print success message"""
        self.print(LogLevel.SUCCESS, message, details, task_id)
    
    def warning(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print warning message"""
        self.print(LogLevel.WARNING, message, details, task_id)
    
    def error(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print error message"""
        self.print(LogLevel.ERROR, message, details, task_id)
    
    def debug(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print debug message"""
        self.print(LogLevel.DEBUG, message, details, task_id)
    
    def task(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print task-related message"""
        self.print(LogLevel.TASK, message, details, task_id)
    
    def system(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print system message"""
        self.print(LogLevel.SYSTEM, message, details, task_id)
    
    def tool(self, message: str, details: Optional[str] = None, task_id: Optional[str] = None):
        """Print tool-related message"""
        self.print(LogLevel.TOOL, message, details, task_id)
    
    def task_summary(self, task_id: str, duration: float, tokens: dict, status: str, final_message: str = None, computational_time: float = None):
        """Print a formatted task summary"""
        status_color = Colors.GREEN if status == "completed" else Colors.YELLOW if status == "incomplete" else Colors.RED
        status_text = self._colorize(status.upper(), status_color + Colors.BOLD)
        
        # Format token information
        total_tokens = tokens.get('tokens_used', 0)
        input_tokens = tokens.get('input_tokens', 0)
        output_tokens = tokens.get('output_tokens', 0)
        llm_calls = tokens.get('llm_calls', 0)
        
        tokens_info = f"Tokens: {total_tokens} (input: {input_tokens}, output: {output_tokens})"
        calls_info = f"LLM Calls: {llm_calls}"
        
        # Format timing information with both total and computational time
        if computational_time is not None:
            timing_info = f"Duration: {duration:.2f}s (compute: {computational_time:.2f}s)"
        else:
            timing_info = f"Duration: {duration:.2f}s"
        
        # Combine task status and final message in one line
        if final_message:
            if status == "completed":
                self.success(f"Task {task_id} COMPLETED - {final_message}")
                self.task(f"Task {task_id} COMPLETED - Closing task")
            else:
                self.warning(f"Task {task_id} {status.upper()} - {final_message}")
                self.task(f"Task {task_id} {status.upper()} - Closing task")
        else:
            self.task(f"Task {task_id} {status_text}")
        
        self.info(timing_info, f"{tokens_info} | {calls_info}")

# Global console instance
console = ProfessionalConsole()
