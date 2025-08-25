import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class HLRLogger:
    """Hierarchical LLM Router Logger - Simplified for new architecture"""
    
    def __init__(self, log_file: str = "hlr_tasks.log"):
        self.log_file = Path(log_file)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
    def start_task(self, task_id: str, message: str, is_periodic: bool = False, frequency_seconds: int = 0, has_completion_condition: bool = False):
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "message": message[:100] + "..." if len(message) > 100 else message,
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "iterations": 0,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
            "status": "running",
            "is_periodic": is_periodic,
            "frequency_seconds": frequency_seconds,
            "has_completion_condition": has_completion_condition
        }
    
    def add_tokens(self, task_id: str, token_info: dict):
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["tokens_used"] += token_info.get("total_tokens", 0)
            self.active_tasks[task_id]["input_tokens"] += token_info.get("input_tokens", 0)
            self.active_tasks[task_id]["output_tokens"] += token_info.get("output_tokens", 0)
            self.active_tasks[task_id]["llm_calls"] += 1
    
    def complete_task(self, task_id: str, status: str = "completed", computational_time: float = None):
        if task_id not in self.active_tasks:
            return
        
        task_data = self.active_tasks[task_id]
        task_data["status"] = status
        task_data["end_time"] = time.time()
        task_data["end_datetime"] = datetime.now().isoformat()
        task_data["duration_seconds"] = round(task_data["end_time"] - task_data["start_time"], 2)
        
        # Add computational time if provided
        if computational_time is not None:
            task_data["computation_seconds"] = round(computational_time, 2)
        
        # Write to log file
        self._write_log_entry(task_data)
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        

    def _write_log_entry(self, task_data: Dict[str, Any]):
        try:
            log_entry = {
                "task_id": task_data["task_id"],
                "message": task_data["message"],
                "start_time": task_data["start_datetime"],
                "end_time": task_data["end_datetime"],
                "duration_seconds": task_data["duration_seconds"],
                "computation_seconds": task_data.get("computation_seconds", task_data["duration_seconds"]),
                "iterations": task_data["iterations"],
                "tokens_used": task_data["tokens_used"],
                "input_tokens": task_data.get("input_tokens", 0),
                "output_tokens": task_data.get("output_tokens", 0),
                "llm_calls": task_data.get("llm_calls", 0),
                "status": task_data["status"],
                "is_periodic": task_data.get("is_periodic", False),
                "frequency_seconds": task_data.get("frequency_seconds", 0),
                "has_completion_condition": task_data.get("has_completion_condition", False)
            }
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"⚠️ [Logger] Failed to write log entry: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        if not self.log_file.exists():
            return self._empty_stats()
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if not lines:
                return self._empty_stats()
            
            stats = {"total_tasks": 0, "total_duration": 0, "total_tokens": 0, "total_input_tokens": 0, 
                    "total_output_tokens": 0, "total_llm_calls": 0, "total_iterations": 0, 
                    "completed_tasks": 0, "periodic_tasks": 0, "tasks_with_completion_condition": 0}
            
            for line in lines:  
                try:
                    entry = json.loads(line)
                    stats["total_tasks"] += 1
                    stats["total_duration"] += entry.get("duration_seconds", 0)
                    stats["total_tokens"] += entry.get("tokens_used", 0)
                    stats["total_input_tokens"] += entry.get("input_tokens", 0)
                    stats["total_output_tokens"] += entry.get("output_tokens", 0)
                    stats["total_llm_calls"] += entry.get("llm_calls", 0)
                    stats["total_iterations"] += entry.get("iterations", 0)
                    
                    if entry.get("status") == "completed":
                        stats["completed_tasks"] += 1
                    if entry.get("is_periodic", False):
                        stats["periodic_tasks"] += 1
                    if entry.get("has_completion_condition", False):
                        stats["tasks_with_completion_condition"] += 1
                        
                except (json.JSONDecodeError, Exception) as e:
                    print(f"⚠️ [Logger] Skipping malformed log line: {e}")
                    continue
            
            if stats["total_tasks"] == 0:
                return self._empty_stats()
            
            # Add averages
            stats.update({
                "avg_duration": round(stats["total_duration"] / stats["total_tasks"], 2),
                "avg_tokens": round(stats["total_tokens"] / stats["total_tasks"], 1),
                "avg_iterations": round(stats["total_iterations"] / stats["total_tasks"], 1),
                "total_duration": round(stats["total_duration"], 2),
                "log_file": str(self.log_file)
            })
            
            return stats
            
        except Exception as e:
            print(f"⚠️ [Logger] Error reading log stats: {e}")
            return {"error": str(e), "total_tasks": 0, "total_tokens": 0, "log_file": str(self.log_file)}
    
    def _empty_stats(self) -> Dict[str, Any]:
        return {"total_tasks": 0, "total_tokens": 0, "total_iterations": 0, "log_file": str(self.log_file)}
