from asyncio import create_task, gather, CancelledError, sleep
from ...internal.llm import is_task_complete

class C0:
    def __init__(self, graph):
        self.graph = graph
        self._tasks = []
        self._should_stop = False
        self.logger = None
        self.current_task_id = None
        
    def set_logger(self, logger, task_id: str):
        """Set the logger and current task ID for tracking."""
        self.logger = logger
        self.current_task_id = task_id
        
    async def run_iteration(self, message: str, memory=None, frequency: int = 0, max_iterations: int = -1):
        """Run graph iterations at specified frequency until max_iterations reached."""
        
        # Set memory reference for graph
        self.graph._memory_ref = memory
        
        if frequency <= 0:
            # Run once if frequency is 0 or negative
            try:
                await self.graph.run(message)
                # Count single execution as 1 iteration for logging consistency
                if self.logger and self.current_task_id:
                    self.logger.increment_iteration(self.current_task_id)
            except Exception as e:
                print(f"⚠️ Single execution failed: {e}")
            return

        iteration_count = 0
        
        while not self._should_stop and (max_iterations == -1 or iteration_count < max_iterations):
            # Create and run task for this iteration
            task = create_task(self.graph.run(message))
            self._tasks.append(task)
            
            try:
                await task
                # Count iteration ONLY after successful execution
                iteration_count += 1
                print(f"[C0] ✅ Completed periodic iteration {iteration_count}")
                
                # Log successful iteration if logger is available
                if self.logger and self.current_task_id:
                    self.logger.increment_iteration(self.current_task_id)
                    
            except CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Iteration failed: {e}")
                # Don't count failed iterations
                continue
            # Check termination condition only for periodic tasks with completion criteria
            # One-time tasks and periodic tasks without completion conditions don't need checking
            should_check_completion = (
                frequency > 0 and  # Periodic task
                getattr(self.graph, '_has_completion_condition', False)  # Has completion condition
            )
            
            if should_check_completion:
                try:
                    memory_content = self.graph._memory_ref.get() if self.graph._memory_ref else ""
                    complete, tokens = await is_task_complete(user_message=message, graph_result=memory_content, model=self.graph.heavy_llm)
                    
                    # Track completion check tokens
                    if self.logger and self.current_task_id:
                        self.logger.add_tokens(self.current_task_id, tokens)
                    
                    if complete:
                        break
                        
                except Exception as e:
                    print(f"Warning: Error in condition evaluation: {e}, continuing normally.")
            
            if max_iterations != -1 and iteration_count >= max_iterations:
                break
                
            try:
                await sleep(frequency)
            except CancelledError:
                break
        
        await self._cancel_tasks()
    
    async def _cancel_tasks(self):
        """Cancel all running tasks."""
        self._should_stop = True
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
