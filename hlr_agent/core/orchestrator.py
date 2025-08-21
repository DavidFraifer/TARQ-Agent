# orchestrator.py - HLR Task Orchestrator
from ..utils.logger import HLRLogger
from ..utils.console import console, LogLevel
from ..tools.tool import ToolContainer
from ..tools.internal_tools import internal_tools
from ..memory.AgentMemory import AgentMemory
from ..internal.llm import llm_completion_async
from typing import Dict, Any, Optional, List
import asyncio
import json
import threading
import queue
import time

class Orchestrator:
    def __init__(self, logger: Optional[HLRLogger] = None, light_llm: str = "gemini-2.5-flash-lite", heavy_llm: str = "gemini-2.5-flash-lite"):
        self.logger = logger
        self.light_llm = light_llm
        self.heavy_llm = heavy_llm
        self.tools = ToolContainer()
        self.message_queue = queue.Queue()
        self.scheduler_thread = None
        self.running = False
        self.agent_memory = AgentMemory(f"Orchestrator-Agent-{id(self)}", max_tasks=50)
        
        # Add internal tools
        for tool_name, tool_func in internal_tools.items():
            self.tools.add_tool(tool_name, tool_func)
        
    def add_tool(self, name: str, func):
        self.tools.add_tool(name, func)
    
    def set_logger(self, logger):
        self.logger = logger
    
    def start(self):
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
            self.scheduler_thread.start()
    
    def stop(self):
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
    
    def receive_message(self, message):
        """Add a message to the processing queue"""
        return self.message_queue.put(message) if self.running else False
    
    def _scheduler_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        running_tasks = set()
        
        async def scheduler_loop():
            nonlocal running_tasks
            while self.running:
                try:
                    message = self.message_queue.get(timeout=0.1)
                    task = asyncio.create_task(self._process_message_async(message))
                    running_tasks.add(task)
                    running_tasks -= {t for t in running_tasks if t.done()}
                    self.message_queue.task_done()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                except Exception:
                    await asyncio.sleep(0.1)
        
        try:
            loop.run_until_complete(scheduler_loop())
            if running_tasks:
                loop.run_until_complete(asyncio.gather(*running_tasks, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()
    
    async def _process_message_async(self, message):
        start_time = time.time()
        task_id = f"{int(start_time * 1000) % 100000000:08d}"
        task_memory = self.agent_memory.create_task_memory(f"Task-{task_id}")
        
        if self.logger:
            self.logger.start_task(task_id, message)
        
        try:
            result = await self.handle_message(message, task_memory, task_id)
            
            # For periodic tasks, use the computation time instead of total elapsed time
            if result.get("computation_time") is not None:
                task_duration = result["computation_time"]
            else:
                task_duration = time.time() - start_time
            
            # Get token information for summary
            token_info = {}
            if self.logger and task_id in self.logger.active_tasks:
                task_data = self.logger.active_tasks[task_id]
                token_info = {
                    'tokens_used': task_data.get('tokens_used', 0),
                    'input_tokens': task_data.get('input_tokens', 0),
                    'output_tokens': task_data.get('output_tokens', 0),
                    'llm_calls': task_data.get('llm_calls', 0)
                }
            
            status = "completed" if result.get("completed", True) else "incomplete"
            final_message = result.get('final_message', "Task execution finished" if status == "completed" else "Task incomplete")
            
            # Display task summary (this will show the completion status and message)
            console.task_summary(task_id, task_duration, token_info, status, final_message)
            
            if self.logger:
                self.logger.complete_task(task_id, status)
                
        except Exception as e:
            console.error("Task execution failed", str(e), task_id=task_id)
            task_memory.set(f"ERROR: {str(e)}")
            if self.logger:
                self.logger.complete_task(task_id, "error")
    
    async def handle_message(self, message: str, task_memory, task_id: str = "default") -> Dict[str, Any]:
        try:
            # Step 1: Initial Analysis
            analysis = await self.llm_analyze_task(message, task_memory, task_id)
            
            frequency_minutes = analysis.get("frequency_minutes", 0)
            
            if frequency_minutes > 0:
                return await self._handle_periodic_execution(message, task_memory, task_id, analysis, frequency_minutes)
            
            # Non-periodic tasks with feedback loop
            current_message = message
            iteration = 1
            max_iterations = 3
            
            while iteration <= max_iterations:
                if iteration > 1:
                    console.info(f"Starting feedback iteration #{iteration}", task_id=task_id)
                    # Re-analyze for feedback iterations
                    analysis = await self.llm_analyze_task(current_message, task_memory, task_id)
                
                tools_to_execute = analysis.get("action_tools", analysis.get("tools", []))
                if not tools_to_execute:
                    return {
                        "completed": True,
                        "final_message": f"Task completed - no tools needed (iteration {iteration})"
                    }
                
                # Step 2: Tool Execution
                if tools_to_execute:
                    console.info("Executing tools", f"Tools: {', '.join(tools_to_execute)}", task_id=task_id)
                await self.execute_tools_sequentially(tools_to_execute, task_memory, f"{task_id}-iter{iteration}")
                
                # Step 3: Validation
                console.debug("Validating task completion", task_id=task_id)
                validation = await self.llm_validate_completion(message, task_memory, task_id)
                
                if validation.get("completed", False):
                    if iteration > 1:
                        console.success(f"Task completed after {iteration} iterations", task_id=task_id)
                    return {"completed": True, "final_message": validation.get("final_message", "Task completed")}
                
                continue_message = validation.get("continue_message", "").strip()
                if not continue_message or continue_message == current_message:
                    return {"completed": False, "final_message": f"Task incomplete after {iteration} iteration(s)"}
                
                console.info("Applying feedback", continue_message[:80] + ("..." if len(continue_message) > 80 else ""), task_id=task_id)
                current_message = continue_message
                iteration += 1
            
            return {"completed": False, "final_message": f"Task incomplete after {max_iterations} iterations"}
            
        except Exception as e:
            return {"completed": True, "final_message": f"Task failed: {str(e)}"}
    
    async def llm_analyze_task(self, message: str, task_memory, task_id: str):
        memory_content = task_memory.get() if task_memory else "No previous context"
        available_tools = list(self.tools.tools.keys())
        
        analysis_prompt = f"""Task: "{message}"
Context: {memory_content}
Tools: {available_tools}

JSON: {{"monitoring_tools": ["tool"], "action_tools": ["tool1","tool2"], "frequency_minutes": int, "partial_task": "description"}}

Examples:
- Send email → {{"monitoring_tools": [], "action_tools": ["gmail"], "frequency_minutes": 0, "partial_task": null}}
- Check emails until admin arrives, then upload → {{"monitoring_tools": ["gmail"], "action_tools": ["sheets"], "frequency_minutes": 60, "partial_task": "Check gmail for admin"}}
- Analyze emails and send summary → {{"monitoring_tools": [], "action_tools": ["gmail", "sheets", "gmail"], "frequency_minutes": 0, "partial_task": null}}

Rules: Tools in order, can repeat. frequency=0 for one-time, >0 for repeating."""
        
        try:
            response, token_info = await llm_completion_async(
                model=self.heavy_llm, 
                prompt=analysis_prompt,
                temperature=0.0,  # Maximum determinism
                max_tokens=100,  
                response_format="json"
            )
            
            if self.logger:
                self.logger.add_tokens(task_id, token_info)
            
            result = json.loads(response)
            return result
            
        except Exception as e:
            return {"monitoring_tools": [], "action_tools": [], "frequency_minutes": 0, "partial_task": None}
    
    async def execute_tools_sequentially(self, tools: List[str], task_memory, task_id: str):
        for tool_name in tools:
            if tool_name in self.tools.tools:
                try:
                    context = task_memory.get() if task_memory else ""
                    result = await self.tools.execute_tool(tool_name, context)
                    task_memory.set(f"Tool {tool_name} result: {result}")
                except Exception as e:
                    console.error(f"Tool '{tool_name}' execution failed", str(e), task_id=task_id)
    
    def _clean_json_response(self, response: str) -> str:
        """Extract clean JSON from LLM response"""
        response_clean = response.strip()
        if response_clean.startswith('{') and '}' in response_clean:
            json_end = response_clean.rfind('}') + 1
            response_clean = response_clean[:json_end]
        return response_clean
    
    async def llm_validate_completion(self, original_message: str, task_memory, task_id: str):
        memory_content = task_memory.get() if task_memory else "No memory available"
        
        validation_prompt = f"""Request: {original_message}
Results: {memory_content}

JSON: {{"completed": true/false, "final_message": "what was accomplished", "continue_message": "next step if incomplete"}}"""
        
        try:
            response, token_info = await llm_completion_async(
                model=self.light_llm, prompt=validation_prompt, temperature=0.0, max_tokens=100, response_format="json"  # Increased tokens for more descriptive messages
            )
            
            if self.logger:
                self.logger.add_tokens(task_id, token_info)
            
            result = json.loads(self._clean_json_response(response))
            return {
                "completed": result.get("completed", False),
                "final_message": result.get("final_message", "Task execution completed"),
                "continue_message": result.get("continue_message", "")
            }
            
        except Exception:
            return {"completed": True, "final_message": "Task execution completed successfully", "continue_message": ""}
        

    async def llm_validate_partial_task(self, partial_task: str, task_memory, task_id: str):
        """Validate if the monitoring condition for a partial task is met"""
        memory_content = task_memory.get() if task_memory else "No memory available"
        clean_partial_task = partial_task.replace('"', "'").replace('\n', ' ')[:100]
        
        validation_prompt = f"""Monitor: {clean_partial_task}
Data: {memory_content}

JSON: {{"condition_met": true/false}}"""
        
        try:
            response, token_info = await llm_completion_async(
                model=self.light_llm,
                prompt=validation_prompt,
                temperature=0.0,   # Zero temperature for maximum consistency
                max_tokens=20,     # Reduced significantly since we only need true/false
                response_format="json"
            )
            
            if self.logger:
                self.logger.add_tokens(task_id, token_info)
            
            result = json.loads(self._clean_json_response(response))
            return {
                "condition_met": result.get("condition_met", False),
                "message": "Condition checked"  # Simple static message
            }
            
        except Exception:
            return {"condition_met": False, "message": "Condition checked"}

    async def _handle_periodic_execution(self, message: str, task_memory, task_id: str, analysis: dict, frequency_minutes: int):
        """Handle periodic task execution with partial task support"""
        partial_task = analysis.get("partial_task")
        has_partial_task = partial_task is not None
        
        if has_partial_task:
            console.info("Starting conditional periodic task", f"Frequency: every {frequency_minutes} minutes", task_id=task_id)
            # Limit monitoring phase message to 60 characters
            monitoring_msg = partial_task[:60] + "..." if len(partial_task) > 60 else partial_task
            console.info("Monitoring phase", monitoring_msg, task_id=task_id)
        else:
            console.info("Starting periodic task", f"Frequency: every {frequency_minutes} minutes", task_id=task_id)
        
        frequency_seconds = frequency_minutes / 60
        iteration = 1
        condition_met = False
        total_computation_time = 0  # Track only computation time, not sleep time
        
        while self.running:
            try:
                iteration_start = time.time()  # Start timing this iteration
                console.info(f"Periodic check #{iteration}", time.strftime('%H:%M:%S'), task_id=task_id)
                
                if has_partial_task and not condition_met:
                    # Phase 1: Execute only monitoring tools
                    monitoring_tools = analysis.get("monitoring_tools", [])
                    if monitoring_tools:
                        await self.execute_tools_sequentially(monitoring_tools, task_memory, f"{task_id}-{iteration}")
                    
                    # Check if monitoring condition is met
                    monitoring_validation = await self.llm_validate_partial_task(partial_task, task_memory, task_id)
                    
                    if monitoring_validation.get("condition_met", False):
                        console.success("Monitoring condition met", monitoring_validation.get('message', 'Proceeding with action phase'), task_id=task_id)
                        condition_met = True
                        
                        # Phase 2: Execute action tools (no re-analysis needed!)
                        action_tools = analysis.get("action_tools", [])
                        if action_tools:
                            console.info("Executing action tools", f"Tools: {', '.join(action_tools)}", task_id=task_id)
                            await self.execute_tools_sequentially(action_tools, task_memory, f"{task_id}-{iteration}")
                        
                        # Validate final completion with feedback loop
                        final_validation = await self.llm_validate_completion(message, task_memory, task_id)
                        
                        if final_validation.get("completed", False):
                            iteration_time = time.time() - iteration_start
                            total_computation_time += iteration_time
                            # Update the logger with actual computation time
                            if self.logger and task_id in self.logger.active_tasks:
                                self.logger.active_tasks[task_id]["duration_seconds"] = round(total_computation_time, 2)
                            return {
                                "completed": True,
                                "final_message": final_validation.get('final_message', 'Conditional task completed'),
                                "computation_time": total_computation_time
                            }
                        
                        # Check for continue_message and apply feedback
                        continue_message = final_validation.get("continue_message", "").strip()
                        if continue_message and continue_message != message:
                            console.info("Applying feedback for conditional task", continue_message[:80] + ("..." if len(continue_message) > 80 else ""), task_id=task_id)
                            # Re-analyze with the feedback message to get new action tools
                            feedback_analysis = await self.llm_analyze_task(continue_message, task_memory, task_id)
                            feedback_tools = feedback_analysis.get("action_tools", feedback_analysis.get("tools", []))
                            if feedback_tools:
                                await self.execute_tools_sequentially(feedback_tools, task_memory, f"{task_id}-{iteration}-feedback")
                                # Re-validate after feedback execution
                                final_validation = await self.llm_validate_completion(message, task_memory, task_id)
                                if final_validation.get("completed", False):
                                    iteration_time = time.time() - iteration_start
                                    total_computation_time += iteration_time
                                    # Update the logger with actual computation time
                                    if self.logger and task_id in self.logger.active_tasks:
                                        self.logger.active_tasks[task_id]["duration_seconds"] = round(total_computation_time, 2)
                                    return {
                                        "completed": True,
                                        "final_message": final_validation.get('final_message', 'Conditional task completed with feedback'),
                                        "computation_time": total_computation_time
                                    }
                    else:
                        console.debug("Monitoring condition not met", "Continuing periodic checks", task_id=task_id)
                else:
                    # No partial task OR condition already met - execute action tools
                    action_tools = analysis.get("action_tools", analysis.get("tools", []))
                    await self.execute_tools_sequentially(action_tools, task_memory, f"{task_id}-{iteration}")
                    
                    # Validate completion with feedback loop
                    validation = await self.llm_validate_completion(message, task_memory, task_id)
                    
                    if validation.get("completed", False):
                        iteration_time = time.time() - iteration_start
                        total_computation_time += iteration_time
                        # Update the logger with actual computation time
                        if self.logger and task_id in self.logger.active_tasks:
                            self.logger.active_tasks[task_id]["duration_seconds"] = round(total_computation_time, 2)
                        return {
                            "completed": True,
                            "final_message": validation.get('final_message', 'Periodic task completed'),
                            "computation_time": total_computation_time
                        }
                    
                    # Check for continue_message and apply feedback
                    continue_message = validation.get("continue_message", "").strip()
                    if continue_message and continue_message != message:
                        console.info("Applying feedback for periodic task", continue_message[:80] + ("..." if len(continue_message) > 80 else ""), task_id=task_id)
                        # Re-analyze with the feedback message to get new action tools
                        feedback_analysis = await self.llm_analyze_task(continue_message, task_memory, task_id)
                        feedback_tools = feedback_analysis.get("action_tools", feedback_analysis.get("tools", []))
                        if feedback_tools:
                            await self.execute_tools_sequentially(feedback_tools, task_memory, f"{task_id}-{iteration}-feedback")
                            # Re-validate after feedback execution
                            validation = await self.llm_validate_completion(message, task_memory, task_id)
                            if validation.get("completed", False):
                                iteration_time = time.time() - iteration_start
                                total_computation_time += iteration_time
                                # Update the logger with actual computation time
                                if self.logger and task_id in self.logger.active_tasks:
                                    self.logger.active_tasks[task_id]["duration_seconds"] = round(total_computation_time, 2)
                                return {
                                    "completed": True,
                                    "final_message": validation.get('final_message', 'Periodic task completed with feedback'),
                                    "computation_time": total_computation_time
                                }
                
                # Add this iteration's computation time (excluding sleep)
                iteration_time = time.time() - iteration_start
                total_computation_time += iteration_time
                
                console.debug(f"Periodic check #{iteration} completed", "Continuing to next iteration", task_id=task_id)
                iteration += 1
                
                # Wait for next execution (this time is NOT counted in computation time)
                console.info(f"Next check scheduled", f"in {frequency_minutes} minutes", task_id=task_id)
                await asyncio.sleep(frequency_seconds)
                
            except Exception as e:
                # Add computation time even for failed iterations
                iteration_time = time.time() - iteration_start
                total_computation_time += iteration_time
                console.error(f"Periodic task iteration #{iteration} failed", str(e), task_id=task_id)
                await asyncio.sleep(frequency_seconds)  # Still wait before next attempt
                iteration += 1
        
        # If we exit the loop (agent stopped), return stopped status with computation time
        if self.logger and task_id in self.logger.active_tasks:
            self.logger.active_tasks[task_id]["duration_seconds"] = round(total_computation_time, 2)
        console.warning("Periodic task stopped by system", task_id=task_id)
        return {
            "completed": True,
            "final_message": "Periodic task stopped",
            "computation_time": total_computation_time
        }
