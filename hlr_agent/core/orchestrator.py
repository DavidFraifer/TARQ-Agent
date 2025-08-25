# orchestrator.py - HLR Task Orchestrator
from ..utils.logger import HLRLogger
from ..utils.console import console
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
        
        # Initialize wait time tracking
        self.wait_times = {}
        self.wait_times[task_id] = 0.0
        
        # Truncate message for logging display
        display_message = message[:60] + "..." if len(message) > 60 else message
        console.task(f"Task {task_id} ADDED - {display_message}", task_id=task_id)
        
        if self.logger:
            self.logger.start_task(task_id, message)
        
        try:
            result = await self.handle_message(message, task_memory, task_id)
            
            # Always use actual elapsed time for consistency
            task_duration = time.time() - start_time
            
            # Calculate computational time (total time - wait time)
            total_wait_time = self.wait_times.get(task_id, 0.0)
            computational_time = task_duration - total_wait_time
            
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
            
            # Display task summary with both timing types
            console.task_summary(task_id, task_duration, token_info, status, final_message, computational_time)
            
            # Display additional timing breakdown only if there was wait time
            if total_wait_time > 0:
                console.info(f"Timing breakdown", f"Total: {task_duration:.2f}s | Computational: {computational_time:.2f}s | Wait: {total_wait_time:.2f}s", task_id=task_id)
            
            if self.logger:
                self.logger.complete_task(task_id, status, computational_time)
                
        except Exception as e:
            console.error("Task execution failed", str(e), task_id=task_id)
            task_memory.set(f"ERROR: {str(e)}")
            if self.logger:
                self.logger.complete_task(task_id, "error", computational_time)
    
    async def handle_message(self, message: str, task_memory, task_id: str = "default") -> Dict[str, Any]:
        try:
            # Step 1: Initial Analysis with timing
            analysis_start = time.time()
            analysis = await self.llm_analyze_task(message, task_memory, task_id)
            analysis_time = time.time() - analysis_start
            
            # Print timing info
            console.debug("Task analysis completed", f"Time: {analysis_time:.2f}s", task_id=task_id)
            
            # Execute the DSL flow
            flow = analysis.get("flow", [])
            if flow:
                return await self._execute_dsl_flow(flow, task_memory, task_id, message)
            else:
                # Fallback: no flow provided, return simple completion
                return {
                    "completed": True,
                    "final_message": "Task completed successfully"
                }
            
        except Exception as e:
            return {"completed": True, "final_message": f"Task failed: {str(e)}"}
    
    async def llm_analyze_task(self, message: str, task_memory, task_id: str):
        # Limit memory context to save tokens
        memory_content = task_memory.get() if task_memory else ""
        if len(memory_content) > 150:
            memory_content = memory_content[-150:]
        available_tools = list(self.tools.tools.keys())
        
        analysis_prompt = f"""Task: "{message}"
Context: {memory_content}
Tools: {available_tools}

DSL Commands:
W N=wait N min, F TOOL=fetch, A TOOL=action, IF/ELSE/ENDIF=conditional, WHILE/ENDWHILE=loop, STOP=complete

Examples:

"Check mail for reports, if from admin create ticket, if from support update sheet"
F gmail
IF contains "admin@google.com" in sender
  A jira
ELSEIF contains "support@google.com" in sender
  A sheets
ELSE
  A slack
ENDIF

"Watch gmail every hour for a report then upload it to sheets"
WHILE TRUE
  F gmail
  IF contains "report" in subject
    A sheets
    STOP
  ENDIF
  W 60
ENDWHILE

Output DSL only:"""
        try:
            response, token_info = await llm_completion_async(
                model=self.heavy_llm, 
                prompt=analysis_prompt,
                temperature=0.0,  # Maximum determinism
                max_tokens=100,  # Reduced from 150
                response_format=None  
            )
            
            if self.logger:
                self.logger.add_tokens(task_id, token_info)
            
            # Parse text DSL instead of JSON
            flow = self._parse_text_dsl(response)
            result = {"flow": flow}
            
            return result
            
        except Exception as e:
            return {"flow": []}

    def _parse_text_dsl(self, text: str) -> List:
        """Parse text DSL into array format for execution"""
        lines = [line.rstrip() for line in text.strip().split('\n') if line.strip()]
        flow = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            i += 1
            
            if line.strip().startswith('W '):
                # Wait command: W 15 -> ["WAIT", 15]
                minutes = int(line.strip().split()[1])
                flow.append(["WAIT", minutes])
                
            elif line.strip().startswith('F '):
                # Fetch command: F gmail -> ["F", "gmail"]
                tool = line.strip().split()[1]
                flow.append(["F", tool])
                
            elif line.strip().startswith('A '):
                # Action command: A jira -> ["A", "jira"]
                parts = line.strip().split()
                tool = parts[1]
                flow.append(["A", tool])
                
            elif line.strip() == 'STOP':
                # Stop command: STOP -> ["STOP"]
                flow.append(["STOP"])
                    
            elif line.strip().startswith('IF '):
                # Conditional: IF condition -> parse block
                condition = line.strip()[3:].strip()
                then_block, else_block, i = self._parse_conditional_block(lines, i, 2)
                flow.append(["IF", condition, then_block, else_block])
                
            elif line.strip().startswith('WHILE '):
                # While loop: WHILE condition -> parse block
                condition = line.strip()[6:].strip()
                body_block, i = self._parse_while_block(lines, i)
                flow.append(["WHILE", condition, body_block])
        
        return flow
    
    def _parse_conditional_block(self, lines: List[str], start_idx: int, expected_indent: int = 2):
        """Parse IF/ELSEIF/ELSE/ENDIF block"""
        then_block = []
        else_block = []
        current_block = then_block
        i = start_idx
        
        while i < len(lines):
            line = lines[i]  # Don't strip here - we need to check indentation
            i += 1
            
            if line.strip() == 'ENDIF':
                break
            elif line.strip().startswith('ELSEIF ') or line.strip() == 'ELSE':
                current_block = else_block
                if line.strip().startswith('ELSEIF '):
                    # Convert ELSEIF to nested IF in else block
                    condition = line.strip()[7:].strip()
                    nested_then, nested_else, i = self._parse_conditional_block(lines, i, expected_indent)
                    else_block.append(["IF", condition, nested_then, nested_else])
                    break
                continue
            elif line.startswith(' ' * expected_indent):
                # Indented line - part of current block
                line_content = line[expected_indent:]  # Remove expected indentation
                
                if line_content.startswith('IF '):
                    # Nested conditional - use deeper indentation
                    condition = line_content[3:].strip()
                    nested_then, nested_else, new_i = self._parse_conditional_block(lines, i, expected_indent + 2)
                    current_block.append(["IF", condition, nested_then, nested_else])
                    i = new_i  # Update index to skip processed lines
                elif line_content.startswith('A '):
                    # Simple parsing - just take tool name
                    parts = line_content.split()
                    tool = parts[1]
                    current_block.append(["A", tool])
                elif line_content.startswith('F '):
                    tool = line_content.split()[1]
                    current_block.append(["F", tool])
                elif line_content.startswith('W '):
                    minutes = int(line_content.split()[1])
                    current_block.append(["WAIT", minutes])
                elif line_content.strip() == 'STOP':
                    current_block.append(["STOP"])
        
        return then_block, else_block, i
    
    def _parse_while_block(self, lines: List[str], start_idx: int):
        """Parse WHILE/ENDWHILE block"""
        body_block = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i]  # Don't strip - need to check indentation
            i += 1
            
            if line.strip() == 'ENDWHILE':
                break
            elif line.startswith('  '):
                # Indented line - part of while body
                line_content = line[2:]  # Remove indentation
                if line_content.startswith('A '):
                    parts = line_content.split()
                    tool = parts[1]
                    body_block.append(["A", tool])
                elif line_content.startswith('F '):
                    tool = line_content.split()[1]
                    body_block.append(["F", tool])
                elif line_content.startswith('W '):
                    minutes = int(line_content.split()[1])
                    body_block.append(["WAIT", minutes])
                elif line_content.strip() == 'STOP':
                    body_block.append(["STOP"])
                elif line_content.startswith('IF '):
                    # Nested conditional in while loop
                    condition = line_content[3:].strip()
                    then_block, else_block, i = self._parse_conditional_block(lines, i, 4)
                    body_block.append(["IF", condition, then_block, else_block])
                    i -= 1  # Adjust for the increment at loop start
        
        return body_block, i

    def _contains_infinite_loop(self, flow: List) -> bool:
        """Check if flow contains WHILE true/True/1/always"""
        for step in flow:
            if isinstance(step, list) and len(step) >= 2:
                if step[0] == "WHILE" and step[1].lower() in ["true", "1", "always"]:
                    return True
                # Check nested flows
                if step[0] in ["IF", "WHILE"] and len(step) > 2:
                    for sub_flow in step[2:]:
                        if isinstance(sub_flow, list) and self._contains_infinite_loop(sub_flow):
                            return True
        return False

    async def _execute_dsl_flow(self, flow: List, task_memory, task_id: str, original_message: str, parent_fetch_result: str = "", skip_validation: bool = False) -> Dict[str, Any]:
        """Execute the new DSL flow structure"""
        execution_start = time.time()
        last_fetch_result = parent_fetch_result
        has_infinite_loop = self._contains_infinite_loop(flow) and not skip_validation
        
        try:
            for step in flow:
                if not isinstance(step, list) or len(step) < 1:
                    continue
                
                command = step[0]
                
                if command == "F":  # Fetch
                    tool = step[1]
                    console.info(f"Fetching data", f"Tool: {tool}", task_id=task_id)
                    context = task_memory.get() if task_memory else ""
                    result = await self.tools.execute_tool(tool, context, task_id=task_id)
                    last_fetch_result = result
                    task_memory.set(f"Fetch {tool}: {result}")
                
                elif command == "A":  # Action
                    tool = step[1]
                    console.info(f"Executing action", f"Tool: {tool}", task_id=task_id)
                    context = task_memory.get() if task_memory else ""
                    result = await self.tools.execute_tool(tool, context, task_id=task_id)
                    task_memory.set(f"Action {tool}: {result}")
                
                elif command == "IF":  # Conditional
                    condition = step[1]
                    then_steps = step[2] if len(step) > 2 else []
                    else_steps = step[3] if len(step) > 3 else []
                    
                    # Evaluate condition against last fetch result
                    condition_result, condition_time = await self._evaluate_dsl_condition(condition, last_fetch_result, task_id)
                    
                    if condition_result:
                        console.info(f"✅ Condition met: {condition}", f"Executing THEN steps (Time: {condition_time:.2f}s)", task_id=task_id)
                        sub_result = await self._execute_dsl_flow(then_steps, task_memory, task_id, original_message, last_fetch_result)
                        if sub_result.get("completed", False):
                            return sub_result
                    else:
                        if else_steps:
                            console.info(f"❌ Condition not met: {condition}", f"Executing ELSE steps (Time: {condition_time:.2f}s)", task_id=task_id)
                            sub_result = await self._execute_dsl_flow(else_steps, task_memory, task_id, original_message, last_fetch_result)
                            if sub_result.get("completed", False):
                                return sub_result
                        else:
                            console.debug(f"❌ Condition not met: {condition}", f"No else clause (Time: {condition_time:.2f}s)", task_id=task_id)
                
                elif command == "WHILE":  # Loop with condition
                    condition = step[1]
                    body_steps = step[2] if len(step) > 2 else []
                    iteration = 1
                    
                    console.info(f"Starting WHILE loop", f"Condition: {condition}", task_id=task_id)
                    
                    while self.running and iteration <= 100:  # Safety limit
                        condition_result, condition_time = await self._evaluate_dsl_condition(condition, last_fetch_result, task_id)
                        
                        if not condition_result:
                            console.info(f"WHILE loop ended", f"Condition false after {iteration-1} iterations", task_id=task_id)
                            break
                        
                        console.info(f"WHILE iteration #{iteration}", f"Executing body steps", task_id=task_id)
                        sub_result = await self._execute_dsl_flow(body_steps, task_memory, task_id, original_message, last_fetch_result, skip_validation=True)
                        
                        # Check if STOP command was executed (always exit on STOP, even for infinite loops)
                        if sub_result.get("completed", False) and sub_result.get("stopped_via_command", False):
                            return sub_result
                        
                        # For finite loops, exit if sub-result is "completed"
                        if condition.lower() not in ["true", "1", "always"] and sub_result.get("completed", False):
                            return sub_result
                        
                        iteration += 1
                
                # Remove unused FOR loop command - not in DSL spec
                
                elif command == "WAIT":  # Wait/Sleep
                    minutes = step[1] if len(step) > 1 else 1
                    seconds = minutes / 60  # Convert minutes to seconds
                    console.info(f"Waiting", f"{minutes} minutes ({seconds}s)", task_id=task_id)
                    
                    # Track wait time for computational timing
                    wait_start = time.time()
                    await asyncio.sleep(seconds)
                    wait_duration = time.time() - wait_start
                    
                    # Add to total wait time for this task
                    if task_id not in self.wait_times:
                        self.wait_times[task_id] = 0.0
                    self.wait_times[task_id] += wait_duration
                
                elif command == "STOP":  # Stop execution
                    console.info(f"STOP command reached", "Completing task", task_id=task_id)
                    
                    # Run validation to get proper final message
                    validation_start = time.time()
                    validation = await self.llm_validate_completion(original_message, task_memory, task_id)
                    validation_time = time.time() - validation_start
                    console.debug("STOP validation completed", f"Time: {validation_time:.2f}s", task_id=task_id)
                    
                    # Check if task is actually complete or needs more work
                    if validation.get("completed", True):
                        execution_time = time.time() - execution_start
                        return {
                            "completed": True,
                            "stopped_via_command": True,  # Flag to indicate STOP command was used
                            "final_message": validation.get("final_message", "Task completed via STOP command"),
                            "execution_time": execution_time
                        }
                    else:
                        # Task is not complete, check for continue_message for feedback loop
                        continue_message = validation.get("continue_message", "")
                        if continue_message:
                            console.info("Task incomplete", "Re-analyzing with additional context", task_id=task_id)
                            
                            # Create extended task description with continue guidance
                            extended_task = f"{original_message}\n\nPrevious progress context: {continue_message}"
                            
                            # Re-analyze the task with the additional context
                            analysis_start = time.time()
                            new_analysis = await self.llm_analyze_task(extended_task, task_memory, task_id)
                            analysis_time = time.time() - analysis_start
                            console.debug("Re-analysis completed", f"Time: {analysis_time:.2f}s", task_id=task_id)
                            
                            if new_analysis and new_analysis.get("flow"):
                                console.debug("Re-analysis generated new plan", "Continuing with new flow", task_id=task_id)
                                # Continue with new analysis - restart the loop
                                return await self._execute_dsl_flow(new_analysis["flow"], task_memory, task_id, original_message, "", skip_validation)
                            else:
                                console.warning("Re-analysis failed", "Completing with partial results", task_id=task_id)
                        
                        # If no continue_message or re-analysis failed, complete with current state
                        execution_time = time.time() - execution_start
                        return {
                            "completed": False,
                            "stopped_via_command": True,
                            "final_message": validation.get("final_message", "Task partially completed via STOP command"),
                            "execution_time": execution_time
                        }
            
            # For infinite loops, never exit - keep running indefinitely
            if has_infinite_loop:
                execution_time = time.time() - execution_start
                return {
                    "completed": False,
                    "final_message": "Periodic task stopped unexpectedly",
                    "execution_time": execution_time
                }
            
            # Skip validation if requested
            if skip_validation:
                execution_time = time.time() - execution_start
                return {
                    "completed": True,
                    "final_message": "Flow executed without validation",
                    "execution_time": execution_time
                }
            
            # Validate completion for finite tasks
            validation_start = time.time()
            validation = await self.llm_validate_completion(original_message, task_memory, task_id)
            validation_time = time.time() - validation_start
            console.debug("DSL flow validation completed", f"Time: {validation_time:.2f}s", task_id=task_id)
            
            execution_time = time.time() - execution_start
            return {
                "completed": validation.get("completed", True),
                "final_message": validation.get("final_message", "DSL flow completed"),
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - execution_start
            return {
                "completed": True,
                "final_message": f"DSL flow execution failed: {str(e)}",
                "execution_time": execution_time
            }

    async def _evaluate_dsl_condition(self, condition: str, last_result: str, task_id: str) -> tuple[bool, float]:
        """Evaluate DSL condition against last_result"""
        condition_start = time.time()
        
        # Optimize obvious conditions without LLM call
        condition_lower = condition.lower().strip()
        if condition_lower in ["true", "1", "always"]:
            condition_time = time.time() - condition_start
            return True, condition_time
        elif condition_lower in ["false", "0", "never"]:
            condition_time = time.time() - condition_start
            return False, condition_time
        
        # Use LLM for complex conditions
        prompt = f"""Data: {last_result}
Condition: {condition}
JSON: {{"met": true/false}}"""
        
        try:
            response, token_info = await llm_completion_async(
                model=self.light_llm,
                prompt=prompt,
                temperature=0.0,
                max_tokens=25,
                response_format="json"
            )
            
            if self.logger:
                self.logger.add_tokens(task_id, token_info)
            
            result = json.loads(response.strip())
            is_met = result.get("met", False)
            
        except Exception:
            is_met = False
            
        condition_time = time.time() - condition_start
        return is_met, condition_time

    async def llm_validate_completion(self, original_message: str, task_memory, task_id: str):
        # Limit memory for validation to save tokens
        memory_content = task_memory.get() if task_memory else ""
        if len(memory_content) > 200:
            memory_content = memory_content[-200:]
        
        validation_prompt = f"""Task: {original_message}
Results: {memory_content}

If DONE: final_message only. If NOT DONE: continue_message only.
JSON: {{"final_message": "", "continue_message": ""}}"""
        
        try:
            response, token_info = await llm_completion_async(
                model=self.light_llm, prompt=validation_prompt, temperature=0.0, max_tokens=60, response_format="json"
            )
            
            if self.logger:
                self.logger.add_tokens(task_id, token_info)
            
            result = json.loads(response.strip())
            final_message = result.get("final_message", "").strip()
            continue_message = result.get("continue_message", "").strip()
            
            # Determine completion based on which message is populated
            completed = bool(final_message and not continue_message)
            
            return {
                "completed": completed,
                "final_message": final_message if final_message else "Task execution completed",
                "continue_message": continue_message
            }
            
        except Exception:
            return {"completed": True, "final_message": "Task execution completed successfully"}
