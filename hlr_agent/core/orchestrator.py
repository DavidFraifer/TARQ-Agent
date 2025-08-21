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
        
        console.task(f"Task {task_id} ADDED", task_id=task_id)
        
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
            # Step 1: Initial Analysis with timing
            analysis_start = time.time()
            analysis = await self.llm_analyze_task(message, task_memory, task_id)
            analysis_time = time.time() - analysis_start
            console.debug("Task analysis completed", f"Time: {analysis_time:.2f}s", task_id=task_id)
            
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
                    # Re-analyze for feedback iterations with timing
                    analysis_start = time.time()
                    analysis = await self.llm_analyze_task(current_message, task_memory, task_id)
                    analysis_time = time.time() - analysis_start
                    console.debug(f"Re-analysis completed (iter #{iteration})", f"Time: {analysis_time:.2f}s", task_id=task_id)
                
                tools_to_execute = analysis.get("tools", [])
                
                # 🌟 BRILLIANT CONDITIONAL LOGIC 🌟
                conditional_logic = analysis.get("conditional_logic", {})
                if conditional_logic.get("enabled", False):
                    console.info("Conditional execution detected", "Evaluating conditions", task_id=task_id)
                    return await self._execute_conditional_logic(analysis, task_memory, task_id, message)
                
                if not tools_to_execute:
                    return {
                        "completed": True,
                        "final_message": f"Task completed - no tools needed (iteration {iteration})"
                    }
                
                # Step 2: Tool Execution
                if tools_to_execute:
                    console.info("Executing tools", f"Tools: {', '.join(tools_to_execute)}", task_id=task_id)
                await self.execute_tools_sequentially(tools_to_execute, task_memory, f"{task_id}-iter{iteration}")
                
                # Step 3: Validation with timing
                validation_start = time.time()
                console.debug("Validating task completion", task_id=task_id)
                validation = await self.llm_validate_completion(message, task_memory, task_id)
                validation_time = time.time() - validation_start
                console.debug("Task validation completed", f"Time: {validation_time:.2f}s", task_id=task_id)
                
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
        # Limit memory context to save tokens
        memory_content = task_memory.get() if task_memory else "No context"
        if len(memory_content) > 200:  # Limit context length
            memory_content = memory_content[-200:]  # Keep last 200 chars
        available_tools = list(self.tools.tools.keys())
        
        analysis_prompt = f"""Task: "{message}"
Context:{memory_content}
Tools:{available_tools}

OUTPUT ONLY valid JSON matching this schema:
{{"tools":["tool1"...],"frequency_minutes":int,"conditional_logic":{{"enabled":true|false,"conditions":[{{"if":"<condition>","then":["tool1"...],"else":["toolB"...}}]}}}}

Rules: first of "tools" gathers data; use "else" for natural if/else; return empty arrays if none; frequency:0 unless explicit interval; no extra fields or text.
"""
        try:
            response, token_info = await llm_completion_async(
                model=self.heavy_llm, 
                prompt=analysis_prompt,
                temperature=0.0,  # Maximum determinism
                max_tokens=80,  # Reduced from 100 (conservative)  
                response_format="json"
            )
            
            if self.logger:
                self.logger.add_tokens(task_id, token_info)
            
            result = json.loads(response)
            return result
            
        except Exception as e:
            return {"tools": [], "frequency_minutes": 0, "conditional_logic": {"enabled": False, "conditions": []}}

    async def _execute_conditional_logic(self, analysis: dict, task_memory, task_id: str, original_message: str) -> dict:
        """🌟 NATURAL: Execute data gathering tool → Evaluate conditions with if-else logic"""
        try:
            # Step 1: Execute first tool to gather data (usually monitoring tool like gmail)
            tools = analysis.get("tools", [])
            if tools:
                data_tool = tools[0]  # First tool is for gathering data
                console.info("Gathering data for conditions", f"Tool: {data_tool}", task_id=task_id)
                await self.execute_tools_sequentially([data_tool], task_memory, f"{task_id}-data")
            
            # Step 2: Get data for condition evaluation
            monitoring_data = task_memory.get() if task_memory else ""
            
            # Step 3: Evaluate conditions with natural if-else logic
            conditions = analysis.get("conditional_logic", {}).get("conditions", [])
            
            for condition in conditions:
                condition_text = condition.get("if", "")
                then_tools = condition.get("then", [])
                else_tools = condition.get("else", [])
                
                # Evaluate condition and execute appropriate tools
                condition_result, condition_time = await self._evaluate_single_condition(condition_text, monitoring_data, task_id)
                if condition_result:
                    # IF condition met
                    console.info(f"✅ Condition met: {condition_text}", f"Executing: {', '.join(then_tools)} (Time: {condition_time:.2f}s)", task_id=task_id)
                    await self.execute_tools_sequentially(then_tools, task_memory, f"{task_id}-then")
                    execution_summary = f"IF executed: {condition_text} → {', '.join(then_tools)}"
                else:
                    # ELSE condition - execute else tools if provided
                    if else_tools:
                        console.info(f"❌ Condition not met: {condition_text}", f"Executing ELSE: {', '.join(else_tools)} (Time: {condition_time:.2f}s)", task_id=task_id)
                        await self.execute_tools_sequentially(else_tools, task_memory, f"{task_id}-else")
                        execution_summary = f"ELSE executed: NOT {condition_text} → {', '.join(else_tools)}"
                    else:
                        console.debug(f"❌ Condition not met: {condition_text}", f"No else clause (Time: {condition_time:.2f}s)", task_id=task_id)
                        continue  # Try next condition
                
                # Validate completion after execution (common for both IF and ELSE) with timing
                validation_start = time.time()
                validation = await self.llm_validate_completion(original_message, task_memory, task_id)
                validation_time = time.time() - validation_start
                console.debug("Conditional validation completed", f"Time: {validation_time:.2f}s", task_id=task_id)
                return {
                    "completed": validation.get("completed", True),
                    "final_message": validation.get("final_message", execution_summary)
                }
            
            # No conditions executed
            return {
                "completed": True, 
                "final_message": "No conditions or else clauses were executed"
            }
            
        except Exception as e:
            return {
                "completed": True, 
                "final_message": f"Conditional execution failed: {str(e)}"
            }

    async def _evaluate_single_condition(self, condition: str, data: str, task_id: str) -> tuple[bool, float]:
        """Smart condition evaluation using lightweight LLM call - returns (result, timing)"""
        prompt = f"""Data: {data}

Check if this condition is met based on the data: {condition}
Return JSON only.
JSON: {{"met": true/false}}"""
        
        condition_start = time.time()
        try:
            response, _ = await llm_completion_async(
                model=self.light_llm,
                prompt=prompt,
                temperature=0.0,
                max_tokens=25,  # Increased to ensure complete JSON response
                response_format="json"
            )
            
            result = json.loads(self._clean_json_response(response))
            is_met = result.get("met", False)
            
        except Exception as e:
            is_met = False
            
        condition_time = time.time() - condition_start
        return is_met, condition_time
    
    
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
        # Limit memory for validation to save tokens
        memory_content = task_memory.get() if task_memory else "No results"
        if len(memory_content) > 300:  # Limit results length
            memory_content = memory_content[-300:]  # Keep last 300 chars
        
        validation_prompt = f"""Task: {original_message}
Results: {memory_content}

JSON: {{"completed": true/false, "final_message": "summary", "continue_message": "next step"}}"""
        
        try:
            response, token_info = await llm_completion_async(
                model=self.light_llm, prompt=validation_prompt, temperature=0.0, max_tokens=80, response_format="json"  # Reduced from 100
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
        

    async def _handle_periodic_execution(self, message: str, task_memory, task_id: str, analysis: dict, frequency_minutes: int):
        """🌟 SIMPLIFIED: Clean periodic execution using conditional logic"""
        console.info("Starting periodic task", f"Frequency: every {frequency_minutes} minutes", task_id=task_id)
        
        frequency_seconds = frequency_minutes * 60  # Convert minutes to seconds
        iteration = 1
        total_computation_time = 0
        
        while self.running:
            try:
                iteration_start = time.time()
                console.info(f"Periodic check #{iteration}", time.strftime('%H:%M:%S'), task_id=task_id)
                
                # Use conditional logic if enabled, otherwise use tools
                conditional_logic = analysis.get("conditional_logic", {})
                if conditional_logic.get("enabled", False):
                    result = await self._execute_conditional_logic(analysis, task_memory, task_id, message)
                else:
                    # Simple tools execution
                    tools = analysis.get("tools", [])
                    if tools:
                        await self.execute_tools_sequentially(tools, task_memory, f"{task_id}-{iteration}")
                    
                    # Validation with timing for periodic tasks
                    validation_start = time.time()
                    validation = await self.llm_validate_completion(message, task_memory, task_id)
                    validation_time = time.time() - validation_start
                    console.debug(f"Periodic validation #{iteration}", f"Time: {validation_time:.2f}s", task_id=task_id)
                    result = {
                        "completed": validation.get("completed", False),
                        "final_message": validation.get("final_message", "Periodic iteration completed")
                    }
                
                iteration_time = time.time() - iteration_start
                total_computation_time += iteration_time
                
                # If task completed, return success
                if result.get("completed", False):
                    if self.logger and task_id in self.logger.active_tasks:
                        self.logger.active_tasks[task_id]["duration_seconds"] = round(total_computation_time, 2)
                    return {
                        "completed": True,
                        "final_message": result.get("final_message", "Periodic task completed"),
                        "computation_time": total_computation_time
                    }
                
                console.debug(f"Periodic check #{iteration} completed", "Continuing to next iteration", task_id=task_id)
                iteration += 1
                
                console.info("Next check scheduled", f"in {frequency_minutes} minutes", task_id=task_id)
                await asyncio.sleep(frequency_seconds)
                
            except Exception as e:
                iteration_time = time.time() - iteration_start
                total_computation_time += iteration_time
                console.error(f"Periodic iteration #{iteration} failed", str(e), task_id=task_id)
                await asyncio.sleep(frequency_seconds)
                iteration += 1
        
        # Agent stopped
        if self.logger and task_id in self.logger.active_tasks:
            self.logger.active_tasks[task_id]["duration_seconds"] = round(total_computation_time, 2)
        return {
            "completed": True,
            "final_message": "Periodic task stopped",
            "computation_time": total_computation_time
        }
