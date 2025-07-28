"""Simple task logger for HLR Agent"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class TaskLogger:
    """Simple logger for tracking task execution metrics."""
    
    def __init__(self, log_file: str = "hlr_tasks.log"):
        self.log_file = Path(log_file)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
    def start_task(self, task_id: str, message: str, is_periodic: bool = False, frequency_seconds: int = 0, has_completion_condition: bool = False):
        """Start tracking a new task."""
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "message": message[:100] + "..." if len(message) > 100 else message,
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "iterations": 0,
            "tokens_used": 0,
            "status": "running",
            "is_periodic": is_periodic,
            "frequency_seconds": frequency_seconds,
            "has_completion_condition": has_completion_condition
        }
    
    def increment_iteration(self, task_id: str):
        """Increment iteration count for a task."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["iterations"] += 1
    
    def add_tokens(self, task_id: str, tokens: int):
        """Add token count for a task."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["tokens_used"] += tokens
    
    def complete_task(self, task_id: str, status: str = "completed"):
        """Mark a task as completed and log it."""
        if task_id not in self.active_tasks:
            return
        
        task_data = self.active_tasks[task_id]
        task_data["status"] = status
        task_data["end_time"] = time.time()
        task_data["end_datetime"] = datetime.now().isoformat()
        task_data["duration_seconds"] = round(task_data["end_time"] - task_data["start_time"], 2)
        
        # Write to log file
        self._write_log_entry(task_data)
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        # Print summary
        self._print_task_summary(task_data)
    
    def _write_log_entry(self, task_data: Dict[str, Any]):
        """Write a single log entry to the file."""
        try:
            # Create log entry without internal timestamps
            log_entry = {
                "task_id": task_data["task_id"],
                "message": task_data["message"],
                "start_time": task_data["start_datetime"],
                "end_time": task_data["end_datetime"],
                "duration_seconds": task_data["duration_seconds"],
                "iterations": task_data["iterations"],
                "tokens_used": task_data["tokens_used"],
                "status": task_data["status"],
                "is_periodic": task_data.get("is_periodic", False),
                "frequency_seconds": task_data.get("frequency_seconds", 0),
                "has_completion_condition": task_data.get("has_completion_condition", False)
            }
            
            # Append to log file
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"⚠️ [Logger] Failed to write log entry: {e}")
    
    def _print_task_summary(self, task_data: Dict[str, Any]):
        """Print a concise task summary (disabled for cleaner output)."""
        pass    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from the log file."""
        if not self.log_file.exists():
            return {"total_tasks": 0, "total_tokens": 0, "total_iterations": 0, "log_file": str(self.log_file)}
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if not lines:
                return {"total_tasks": 0, "total_tokens": 0, "total_iterations": 0, "log_file": str(self.log_file)}
            
            total_tasks = 0
            total_duration = 0
            total_tokens = 0
            total_iterations = 0
            completed_tasks = 0
            periodic_tasks = 0
            tasks_with_completion_condition = 0
            
            for line in lines:  
                try:
                    entry = json.loads(line)
                    total_tasks += 1
                    total_duration += entry.get("duration_seconds", 0)
                    total_tokens += entry.get("tokens_used", 0)
                    total_iterations += entry.get("iterations", 0)
                    
                    if entry.get("status") == "completed":
                        completed_tasks += 1
                    if entry.get("is_periodic", False):
                        periodic_tasks += 1
                    if entry.get("has_completion_condition", False):
                        tasks_with_completion_condition += 1
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ [Logger] Skipping malformed log line: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ [Logger] Error processing log line: {e}")
                    continue
            
            if total_tasks == 0:
                return {"total_tasks": 0, "total_tokens": 0, "total_iterations": 0, "log_file": str(self.log_file)}
            
            avg_duration = round(total_duration / total_tasks, 2)
            avg_tokens = round(total_tokens / total_tasks, 1)
            avg_iterations = round(total_iterations / total_tasks, 1)
            
            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "periodic_tasks": periodic_tasks,
                "tasks_with_completion_condition": tasks_with_completion_condition,
                "total_tokens": total_tokens,
                "total_iterations": total_iterations,
                "total_duration": round(total_duration, 2),
                "avg_duration": avg_duration,
                "avg_tokens": avg_tokens,
                "avg_iterations": avg_iterations,
                "log_file": str(self.log_file)
            }
            
        except Exception as e:
            print(f"⚠️ [Logger] Error reading log stats: {e}")
            return {"error": str(e), "total_tasks": 0, "total_tokens": 0, "log_file": str(self.log_file)}
