from typing import Dict, Optional, List
from .TaskMemory import TaskMemory

class AgentMemory:
    """Agent-level memory that manages multiple TaskMemory instances."""
    
    def __init__(self, agent_id: str, max_tasks: int = 100):
        self.agent_id = agent_id
        self.max_tasks = max_tasks
        self._task_memories: Dict[str, TaskMemory] = {}
        self._task_order: List[str] = []
    
    def create_task_memory(self, task_id: str, max_lines: int = 50) -> TaskMemory:
        """Create a new TaskMemory and add it to the agent's memory."""
        task_memory = TaskMemory(task_id, max_lines)
        self.add_task_memory(task_memory)
        return task_memory
    
    def add_task_memory(self, task_memory: TaskMemory) -> None:
        """Add an existing TaskMemory to the agent's memory."""
        task_id = task_memory.name
        
        if task_id in self._task_memories:
            self._task_memories[task_id] = task_memory
        else:
            self._task_memories[task_id] = task_memory
            self._task_order.append(task_id)
            self._cleanup_if_needed()
    
    def get_task_memory(self, task_id: str) -> Optional[TaskMemory]:
        """Get a TaskMemory by its ID."""
        return self._task_memories.get(task_id)
    
    def get_all_task_ids(self) -> List[str]:
        """Get all task IDs in the agent memory."""
        return list(self._task_memories.keys())
    
    def get_recent_tasks(self, count: int = 10) -> List[TaskMemory]:
        """Get the most recent TaskMemory instances."""
        recent_ids = self._task_order[-count:] if count > 0 else self._task_order
        return [self._task_memories[task_id] for task_id in recent_ids if task_id in self._task_memories]
    
    def get_task_count(self) -> int:
        """Get the number of tasks in memory."""
        return len(self._task_memories)
    
    def get_summary(self) -> str:
        """Get a summary of all tasks in memory."""
        if not self._task_memories:
            return f"Agent {self.agent_id}: No tasks in memory"
        
        lines = [
            f"Agent {self.agent_id} Memory Summary:",
            f"Total tasks: {len(self._task_memories)}",
            "Recent tasks:"
        ]
        
        for task_memory in self.get_recent_tasks(5):
            lines.append(f"  - {task_memory.name}: {task_memory.get_line_count()} entries")
        
        return "\n".join(lines)
    
    def _cleanup_if_needed(self) -> None:
        """Remove oldest tasks to maintain max_tasks limit."""
        while len(self._task_memories) > self.max_tasks and self._task_order:
            oldest_task_id = self._task_order.pop(0)
            self._task_memories.pop(oldest_task_id, None)
    
    def __len__(self) -> int:
        return len(self._task_memories)
    
    def __contains__(self, task_id: str) -> bool:
        return task_id in self._task_memories
    
    def __str__(self) -> str:
        return self.get_summary()
