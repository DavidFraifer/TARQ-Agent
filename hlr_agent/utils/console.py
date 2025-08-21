# console.py - Professional Console Output System
import sys
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
        self.spinner_active = False
        self.spinner_thread = None
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
        
        # Spinner characters for loading animation
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_index = 0
    
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
            if self.spinner_active:
                self._clear_spinner_line()
            
            formatted = self._format_message(level, message, details, task_id)
            print(formatted)
            
            if self.spinner_active:
                self._show_spinner_line()
    
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
    
    def start_spinner(self, message: str = "Processing"):
        """Start animated spinner"""
        if self.spinner_active:
            return
            
        self.spinner_active = True
        self.spinner_message = message
        self.spinner_thread = threading.Thread(target=self._spinner_worker, daemon=True)
        self.spinner_thread.start()
    
    def stop_spinner(self, final_message: Optional[str] = None):
        """Stop animated spinner"""
        if not self.spinner_active:
            return
            
        self.spinner_active = False
        if self.spinner_thread:
            self.spinner_thread.join()
        
        with self._lock:
            self._clear_spinner_line()
            if final_message:
                self.success(final_message)
    
    def _spinner_worker(self):
        """Worker thread for spinner animation"""
        while self.spinner_active:
            with self._lock:
                self._show_spinner_line()
            time.sleep(0.1)
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
    
    def _show_spinner_line(self):
        """Show the spinner line"""
        if not self.spinner_active:
            return
            
        spinner_char = self.spinner_chars[self.spinner_index]
        timestamp = self._colorize(self._get_timestamp(), Colors.DIM)
        spinner_colored = self._colorize(spinner_char, Colors.CYAN + Colors.BOLD)
        message_colored = self._colorize(self.spinner_message, Colors.CYAN)
        
        line = f"{timestamp} {spinner_colored} {message_colored}..."
        print(f"\r{line}", end="", flush=True)
    
    def _clear_spinner_line(self):
        """Clear the current spinner line"""
        print(f"\r{' ' * 80}\r", end="", flush=True)
    
    def section_header(self, title: str):
        """Print a section header"""
        border = "=" * 60
        border_colored = self._colorize(border, Colors.BLUE + Colors.BOLD)
        title_colored = self._colorize(f" {title} ", Colors.WHITE + Colors.BOLD + Colors.BG_BLUE)
        
        print()
        print(border_colored)
        print(title_colored)
        print(border_colored)
        print()
    
    def task_summary(self, task_id: str, duration: float, tokens: dict, status: str, final_message: str = None):
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
    
    def progress_bar(self, current: int, total: int, width: int = 40, message: str = "Progress"):
        """Display a progress bar"""
        if total == 0:
            percentage = 100
            filled = width
        else:
            percentage = (current / total) * 100
            filled = int(width * current / total)
        
        bar = "█" * filled + "░" * (width - filled)
        bar_colored = self._colorize(bar, Colors.GREEN if percentage == 100 else Colors.CYAN)
        percentage_colored = self._colorize(f"{percentage:5.1f}%", Colors.BOLD)
        
        with self._lock:
            if self.spinner_active:
                self._clear_spinner_line()
            
            print(f"\r{message}: {bar_colored} {percentage_colored} ({current}/{total})", end="", flush=True)
            
            if current >= total:
                print()  # New line when complete

# Global console instance
console = ProfessionalConsole()
