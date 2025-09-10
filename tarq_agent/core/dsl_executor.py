from ..utils.console import console
from ..internal.llm import llm_completion_async
from typing import Dict, Any, List
import asyncio, json, time

class DSLExecutor:
    """Handles execution of parsed DSL flow structures"""
    
    def __init__(self, tools, logger, light_llm, heavy_llm, agent_id, validation_mode):
        self.tools, self.logger = tools, logger
        self.light_llm, self.heavy_llm = light_llm, heavy_llm
        self.agent_id, self.validation_mode = agent_id, validation_mode
        self.wait_times, self.running = {}, True

    def _normalize_llm_result(self, llm_result):
        """Normalize various possible return shapes from llm_completion_async."""
        response, token_info = "", {}
        if isinstance(llm_result, tuple) and len(llm_result) >= 1:
            response = llm_result[0] or ""
            token_info = llm_result[1] if len(llm_result) > 1 and isinstance(llm_result[1], dict) else {}
        else:
            response = llm_result if llm_result is not None else ""
        
        token_info = token_info or {}
        return response, {
            'input_tokens': token_info.get('input_tokens', token_info.get('input', 0)),
            'output_tokens': token_info.get('output_tokens', token_info.get('output', 0)),
            'total_tokens': token_info.get('total_tokens', token_info.get('tokens', 0)),
            'llm_calls': token_info.get('llm_calls', token_info.get('calls', 1))
        }

    async def execute_dsl_flow(self, flow: List, task_memory, task_id: str, original_message: str, 
                              parent_fetch_result: str = "", skip_validation: bool = False) -> Dict[str, Any]:
        """Execute the DSL flow structure"""
        execution_start = time.time()
        
        # If a delegate already redirected this task, stop executing here
        if hasattr(self, '_redirects') and task_id in getattr(self, '_redirects', {}):
            info = self._redirects.get(task_id, {})
            target = info.get('target_name', 'unknown')
            console.info(f"Task redirected", f"Forwarded to {target}", task_id=task_id, agent_id=self.agent_id)
            return {"completed": False, "final_message": f"Task redirected to {target}"}
        
        last_fetch_result = parent_fetch_result
        
        try:
            for step in flow:
                if not isinstance(step, list) or len(step) < 1:
                    continue
                
                command = step[0]
                
                if command == "F":  # Fetch
                    tool = step[1]
                    console.info(f"Fetching data", f"Tool: {tool}", task_id=task_id, agent_id=self.agent_id)
                    context = original_message
                    try:
                        result = await self.tools.execute_tool(tool, context, task_id=task_id, 
                                                             task_memory=task_memory, light_llm=self.light_llm, 
                                                             heavy_llm=self.heavy_llm, agent_id=self.agent_id, 
                                                             validation_mode=self.validation_mode)
                        last_fetch_result = result
                        self._track_tool_tokens(tool, task_id)
                    except Exception as e:
                        console.error(f"Tool execution failed", f"Tool: {tool}, Error: {str(e)}", task_id=task_id, agent_id=self.agent_id)
                        last_fetch_result = f"Error executing {tool}: {str(e)}"
                
                elif command == "A":  # Action
                    tool = step[1]
                    console.info(f"Executing action", f"Tool: {tool}", task_id=task_id, agent_id=self.agent_id)
                    context = original_message
                    try:
                        result = await self.tools.execute_tool(tool, context, task_id=task_id, 
                                                             task_memory=task_memory, light_llm=self.light_llm, 
                                                             heavy_llm=self.heavy_llm, agent_id=self.agent_id, 
                                                             validation_mode=self.validation_mode)
                        self._track_tool_tokens(tool, task_id)
                    except Exception as e:
                        console.error(f"Tool execution failed", f"Tool: {tool}, Error: {str(e)}", task_id=task_id, agent_id=self.agent_id)
                
                elif command == "IF":  # Conditional
                    condition = step[1]
                    then_steps = step[2] if len(step) > 2 else []
                    else_steps = step[3] if len(step) > 3 else []
                    
                    condition_result, condition_time = await self._evaluate_dsl_condition(condition, last_fetch_result, task_memory, task_id, original_message)
                    
                    if condition_result:
                        console.info(f"✅ Condition met: {condition}", f"Executing THEN steps (Time: {condition_time:.2f}s)", task_id=task_id, agent_id=self.agent_id)
                        sub_result = await self.execute_dsl_flow(then_steps, task_memory, task_id, original_message, last_fetch_result, skip_validation)
                        if sub_result.get("stopped_via_command", False):
                            return sub_result
                        if not skip_validation and sub_result.get("completed", False):
                            return sub_result
                    else:
                        if else_steps:
                            console.info(f"❌ Condition not met: {condition}", f"Executing ELSE steps (Time: {condition_time:.2f}s)", task_id=task_id, agent_id=self.agent_id)
                            sub_result = await self.execute_dsl_flow(else_steps, task_memory, task_id, original_message, last_fetch_result, skip_validation)
                            if sub_result.get("stopped_via_command", False):
                                return sub_result
                            if not skip_validation and sub_result.get("completed", False):
                                return sub_result
                        else:
                            console.debug(f"❌ Condition not met: {condition}", f"No else clause (Time: {condition_time:.2f}s)", task_id=task_id, agent_id=self.agent_id)
                
                elif command == "WHILE":  # Loop with condition
                    condition = step[1]
                    body_steps = step[2] if len(step) > 2 else []
                    iteration = 1
                    
                    console.info(f"Starting WHILE loop", f"Condition: {condition}", task_id=task_id, agent_id=self.agent_id)
                    
                    while self.running and iteration <= 100:  # Safety limit
                        condition_result, condition_time = await self._evaluate_dsl_condition(condition, last_fetch_result, task_memory, task_id, original_message)
                        
                        if not condition_result:
                            console.info(f"WHILE loop ended", f"Condition false after {iteration-1} iterations", task_id=task_id, agent_id=self.agent_id)
                            break
                        
                        console.info(f"WHILE iteration #{iteration}", f"Executing body steps", task_id=task_id, agent_id=self.agent_id)
                        sub_result = await self.execute_dsl_flow(body_steps, task_memory, task_id, original_message, last_fetch_result, skip_validation=True)
                        
                        if sub_result.get("completed", False) and sub_result.get("stopped_via_command", False):
                            return sub_result
                        
                        if condition.lower() not in ["true", "1", "always"] and sub_result.get("completed", False):
                            return sub_result
                        
                        iteration += 1
                
                elif command == "WAIT":  # Wait/Sleep
                    minutes = step[1] if len(step) > 1 else 1
                    seconds = minutes / 60  # Fast-simulate minutes as seconds for examples/tests
                    console.info(f"Waiting", f"{minutes} minutes ({seconds}s)", task_id=task_id, agent_id=self.agent_id)
                    
                    wait_start = time.time()
                    await asyncio.sleep(seconds)
                    wait_duration = time.time() - wait_start
                    
                    if task_id not in self.wait_times:
                        self.wait_times[task_id] = 0.0
                    self.wait_times[task_id] += wait_duration
                
                elif command == "STOP":  # Stop execution
                    console.info(f"STOP command reached", "Completing task", task_id=task_id, agent_id=self.agent_id)
                    
                    validation_start = time.time()
                    validation = await self._validate_completion(original_message, task_memory, task_id)
                    validation_time = time.time() - validation_start
                    console.debug("STOP validation completed", f"Time: {validation_time:.2f}s", task_id=task_id, agent_id=self.agent_id)
                    
                    execution_time = time.time() - execution_start
                    if validation.get("completed", True):
                        return {
                            "completed": True,
                            "stopped_via_command": True,
                            "status": validation.get("status", "success"),
                            "final_message": validation.get("final_message", "Task completed via STOP command"),
                            "execution_time": execution_time
                        }
                    else:
                        # Handle incomplete task with re-analysis if needed
                        continue_message = validation.get("continue_message", "")
                        if continue_message:
                            console.info("Task incomplete", "Re-analyzing with additional context", task_id=task_id, agent_id=self.agent_id)
                            # This would require access to llm_analyze_task from orchestrator
                            # For now, return partial completion
                        
                        return {
                            "completed": False,
                            "stopped_via_command": True,
                            "status": validation.get("status", "warning"),
                            "final_message": validation.get("final_message", "Task partially completed via STOP command"),
                            "execution_time": execution_time
                        }
            
            if skip_validation:
                execution_time = time.time() - execution_start
                return {
                    "completed": True,
                    "status": "success",
                    "final_message": "Flow executed without validation",
                    "execution_time": execution_time
                }
            
            validation_start = time.time()
            validation = await self._validate_completion(original_message, task_memory, task_id)
            validation_time = time.time() - validation_start
            console.debug("DSL flow validation completed", f"Time: {validation_time:.2f}s", task_id=task_id, agent_id=self.agent_id)
            
            execution_time = time.time() - execution_start
            return {
                "completed": validation.get("completed", True),
                "status": validation.get("status", "success"),
                "final_message": validation.get("final_message", "DSL flow completed"),
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - execution_start
            return {
                "completed": True,
                "status": "error",
                "final_message": f"DSL flow execution failed: {str(e)}",
                "execution_time": execution_time
            }

    def _track_tool_tokens(self, tool, task_id):
        """Track token usage for websearch tool"""
        if tool == "websearch" and hasattr(self.tools.tools["websearch"], '_last_token_info'):
            token_info = getattr(self.tools.tools["websearch"], '_last_token_info')
            if token_info:
                for _ in range(token_info.get("llm_calls", 0)):
                    self.logger.add_tokens(task_id, {
                        "input_tokens": token_info.get("input_tokens", 0) // token_info.get("llm_calls", 1),
                        "output_tokens": token_info.get("output_tokens", 0) // token_info.get("llm_calls", 1),
                        "total_tokens": token_info.get("tokens_used", 0) // token_info.get("llm_calls", 1)
                    }, self.light_llm)

    async def _evaluate_dsl_condition(self, condition: str, last_result: str, task_memory, task_id: str, original_message: str = "") -> tuple[bool, float]:
        """Evaluate DSL condition against task memory and last result"""
        condition_start = time.time()
        
        condition_lower = condition.lower().strip()
        if condition_lower in ["true", "1", "always"]:
            return True, time.time() - condition_start
        elif condition_lower in ["false", "0", "never"]:
            return False, time.time() - condition_start
        
        memory_content = task_memory.get() if task_memory else ""
        memory_lines = memory_content.splitlines() if memory_content else []
        memory_content_lite = "\n".join(memory_lines[-10:])  # Always use last 20 lines
        
        # Use same prompt structure for all conditions
        prompt = (
            f"Task: {original_message}\n"
            f"Question: {condition}\n"
            f"Memory: {memory_content_lite}\n\n"
            f"Answer with JSON: {{\"met\": true/false}}"
            f"In case of several possible interpretations, focus on the most recent one based on the memory."
        )
        try:
            llm_result = await llm_completion_async(
                model=self.light_llm, prompt=prompt, temperature=0.0, max_tokens=50, response_format="json")
            response, norm_token_info = self._normalize_llm_result(llm_result)
            try:
                self.logger.add_tokens(task_id, norm_token_info, self.light_llm)
            except Exception:
                pass
            result = json.loads(response.strip())
            is_met = result.get("met", False)
        except Exception as e:
            is_met = False
            
        return is_met, time.time() - condition_start

    async def _validate_completion(self, original_message: str, task_memory, task_id: str):
        """Validate task completion using LLM"""
        memory_content = task_memory.get() if task_memory else ""
        mem_lines = memory_content.splitlines() if memory_content else []
        memory_content_lite = "\n".join(mem_lines[-20:])  
        
        validation_prompt = (
            f"Task: {original_message}\n"
            f"Results:\n{memory_content_lite}\n\n"
            f"Return JSON only. Analyze the task completion and provide:\n"
            f"- status: 'success' (task completed successfully), 'warning' (completed with issues), or 'error' (failed)\n"
            f"- If done, set final_message (leave continue_message empty)\n"
            f"- If not done, set continue_message (leave final_message empty)\n\n"
            f"{{\"status\": \"success\", \"final_message\": \"\", \"continue_message\": \"\"}}"
        )
        
        try:
            llm_result = await llm_completion_async(
                model=self.heavy_llm, prompt=validation_prompt, temperature=0.1, max_tokens=500, response_format="json")
            response, norm_token_info = self._normalize_llm_result(llm_result)
            self.logger.add_tokens(task_id, norm_token_info, self.heavy_llm)

            result = json.loads(response.strip())
            status = result.get("status", "success").strip()
            final_message = result.get("final_message", "").strip()
            continue_message = result.get("continue_message", "").strip()
            
            if status not in ["success", "warning", "error"]:
                status = "success"
            
            completed = bool(final_message and not continue_message)
            return {
                "completed": completed,
                "status": status,
                "final_message": final_message if final_message else "Task execution completed",
                "continue_message": continue_message
            }
            
        except Exception as e:
            console.error("LLM validation failed", str(e), task_id=task_id, agent_id=self.agent_id)
            return {"completed": True, "status": "error", "final_message": "Task execution completed with validation error"}
