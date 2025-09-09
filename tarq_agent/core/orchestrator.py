from ..utils.logger import TARQLogger
from ..utils.console import console
from ..tools.tool import ToolContainer
from ..tools.internal_tools import internal_tools
from ..memory.AgentMemory import AgentMemory
from ..internal.llm import llm_completion_async
from typing import Dict, Any, List
import asyncio
import json
import threading
import queue
import time
import uuid
import re

class Orchestrator:
    def __init__(self, light_llm: str, heavy_llm: str, logger: TARQLogger, agent_id: str = "unknown", disable_delegation: bool = False, rag_engine=None, validation_mode: bool = False):
        self.logger = logger
        self.light_llm = light_llm
        self.heavy_llm = heavy_llm
        self.rag_engine = rag_engine
        self.agent_id = agent_id
        self.disable_delegation = disable_delegation
        self.validation_mode = validation_mode
        self.tools = ToolContainer()
        self.tool_descriptions = {}  # Store descriptions for custom tools
        self.message_queue = queue.Queue()
        self.scheduler_thread = None
        self.running = False
        self.agent_memory = AgentMemory(f"Orchestrator-Agent-{agent_id}", max_tasks=50)
        self._redirects = {}

        # Add internal tools
        for tool_name, tool_func in internal_tools.items():
            self.tools.add_tool(tool_name, tool_func)
        
    def add_tool(self, name: str, func, description: str = None):
        self.tools.add_tool(name, func)
        if description:
            self.tool_descriptions[name] = description

    def _normalize_llm_result(self, llm_result):
        """Normalize various possible return shapes from llm_completion_async.

        Returns: (response_str, normalized_token_info_dict)
        """
        # Default empty values
        response = ""
        token_info = {}

        if isinstance(llm_result, tuple) and len(llm_result) >= 1:
            response = llm_result[0] or ""
            token_info = llm_result[1] if len(llm_result) > 1 and isinstance(llm_result[1], dict) else {}
        else:
            response = llm_result if llm_result is not None else ""
            token_info = {}

        token_info = token_info or {}
        norm_token_info = {
            'input_tokens': token_info.get('input_tokens', token_info.get('input', 0)),
            'output_tokens': token_info.get('output_tokens', token_info.get('output', 0)),
            'total_tokens': token_info.get('total_tokens', token_info.get('tokens', 0)),
            'llm_calls': token_info.get('llm_calls', token_info.get('calls', 1))
        }

        return response, norm_token_info
    
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
        task_id = f"task-{str(uuid.uuid4())[:8]}"
        task_memory = self.agent_memory.create_task_memory(f"Task-{task_id}")
        # Detect forwarded/delegated envelope messages and extract payload
        is_forwarded = False
        origin_agent = None
        origin_task = None
        payload = message
        if isinstance(message, dict) and message.get('_forwarded'):
            is_forwarded = True
            origin_agent = message.get('origin_agent')
            origin_task = message.get('origin_task_id')
            payload = message.get('payload', '')

        # Initialize wait time tracking
        self.wait_times = {}
        self.wait_times[task_id] = 0.0

        # Truncate message for logging display
        display_message = payload[:60] + "..." if isinstance(payload, str) and len(payload) > 60 else payload
        console.task(f"Task {task_id} ADDED [Agent: {self.agent_id}] - {display_message}", task_id=task_id, agent_id=self.agent_id)
        
        self.logger.start_task(task_id, message, self.agent_id)

        # If this orchestrator is part of a team and the task was NOT forwarded, start the delegation worker in background
        delegate_task = None
        try:
            if hasattr(self, 'team') and getattr(self, 'team') and not is_forwarded and not self.disable_delegation:
                delegate_task = asyncio.create_task(self._delegate_worker(payload, task_id))
        except Exception:
            delegate_task = None

        try:
            # Step 1: Initial Analysis with timing (pre-parse the DSL)
            analysis_start = time.time()
            analysis = await self.llm_analyze_task(payload, task_memory, task_id)
            analysis_time = time.time() - analysis_start
            
            # Debug: Print the LLM analysis result
            print("LLM Analysis Result:", analysis)
            # If a delegate redirected this task while analysis was running, stop here and suppress analysis output
            # Wait for delegate to finish if it is still running to ensure we honor its decision
            if delegate_task is not None and not delegate_task.done():
                try:
                    await delegate_task
                except Exception:
                    # delegate failure is non-fatal
                    pass

            if task_id in getattr(self, '_redirects', {}):
                info = self._redirects.get(task_id, {})
                target = info.get('target_name', 'unknown')
                console.debug("Task analysis suppressed due to redirect", f"Forwarded to {target}", task_id=task_id, agent_id=self.agent_id)
                result = {"completed": False, "final_message": f"Task redirected to {target}"}
            else:
                # Check if this is a direct answer (no DSL needed)
                if "direct_answer" in analysis:
                    console.debug("Direct answer provided", "Skipping DSL execution to save tokens", task_id=task_id, agent_id=self.agent_id)
                    result = {"completed": True, "final_message": analysis["direct_answer"]}
                else:
                    # Print timing info
                    console.debug("Task analysis completed", f"Time: {analysis_time:.2f}s", task_id=task_id, agent_id=self.agent_id)

                    # Execute the DSL flow
                    flow = analysis.get("flow", [])
                    if flow:
                        result = await self._execute_dsl_flow(flow, task_memory, task_id, payload)
                    else:
                        # Fallback: no flow provided, return simple completion
                        result = {"completed": True, "final_message": "Task completed successfully"}
            
            # Always use actual elapsed time for consistency
            task_duration = time.time() - start_time
            
            # Calculate computational time (total time - wait time)
            total_wait_time = self.wait_times.get(task_id, 0.0)
            computational_time = task_duration - total_wait_time
            
            # Get token information for summary
            task_data = self.logger.active_tasks.get(task_id, {})
            token_info = {
                'tokens_used': task_data.get('tokens_used', 0),
                'input_tokens': task_data.get('input_tokens', 0),
                'output_tokens': task_data.get('output_tokens', 0),
                'llm_calls': task_data.get('llm_calls', 0),
                'total_cost': task_data.get('total_cost', 0.0)
            }
            
            status = "completed" if result.get("completed", True) else "incomplete"
            final_message = result.get('final_message', "Task execution finished" if status == "completed" else "Task incomplete")
            
            # Display task summary with both timing types
            console.task_summary(task_id, task_duration, token_info, status, final_message, computational_time, agent_id=self.agent_id)
            
            self.logger.complete_task(task_id, status, computational_time)
        except Exception as e:
            console.error("Task execution failed", str(e), task_id=task_id, agent_id=self.agent_id)
            task_memory.set(f"ERROR: {str(e)}")
            # Ensure computational_time is defined on error path
            elapsed = time.time() - start_time
            total_wait_time = self.wait_times.get(task_id, 0.0)
            comp_time = elapsed - total_wait_time
            self.logger.complete_task(task_id, "error", comp_time)
    
    async def llm_analyze_task(self, message: str, task_memory, task_id: str):
        # Limit memory context to save tokens
        memory_content = task_memory.get() if task_memory else ""

        # Get RAG context if available
        rag_context = ""
        if self.rag_engine and self.rag_engine.is_enabled():
            rag_context = self.rag_engine.get_context(message, max_length=500)
            if rag_context:
                rag_context = f"\nKnowledge Base:\n{rag_context}\n"

        available_tools = list(self.tools.tools.keys())
        
        # Build tool descriptions - include descriptions only for custom tools
        tools_info = []
        for tool_name in available_tools:
            if tool_name in self.tool_descriptions:
                # Custom tool with description
                tools_info.append(f"{tool_name}: {self.tool_descriptions[tool_name]}")
            else:
                # Internal tool without description
                tools_info.append(tool_name)
        
        tools_display = "[" + ", ".join(tools_info) + "]"

        analysis_prompt = f"""
Task: "{message}"
Context: {memory_content}{rag_context}
Tools: {tools_display}

DSL:
W N=wait N min | F TOOL=fetch (before IF/WHILE) | A TOOL=action | IF/ELSEIF/ELSE/ENDIF=conditions | WHILE/ENDWHILE=loops | STOP=complete

Examples:
"Look for amazon revenue 2024 'Annual Report' and upload to sheets"
A websearch
A sheets


"Watch gmail every hour for a report then upload it to sheets, in case the email is not a report notify me in slack"
WHILE TRUE
    F gmail
    IF (subject contains "report")
        A sheets
        STOP
    ELSE
        A slack
    ENDIF
    W 60
ENDWHILE

"Monitor emails until finding messages from BOTH admin and support, then stop"
WHILE TRUE
    F gmail
    IF (sender="admin@google.com")
        A jira
    ELSEIF (sender="support@google.com")
        A sheets
    ENDIF
    IF (found emails from both admin and support in memory)
        STOP
    ENDIF
    W 15
ENDWHILE

Rules:
- STOP only inside IF/ELSEIF if task says "stop immediately"
- For "stop when BOTH X and Y": do actions separately, then check both before STOP
- Final completion check must be in its own IF
- Memory = action history → use to verify

If user needs only a normal answer (no tool use):  
Answer: [response]
"""
        
        try:
            llm_result = await llm_completion_async(
                model=self.heavy_llm,
                prompt=analysis_prompt,
                temperature=0.0,  # Maximum determinism for DSL parsing
                max_tokens=100,   # Reduced from 150 for efficiency
                response_format=None,
            )

            response, norm_token_info = self._normalize_llm_result(llm_result)
            try:
                self.logger.add_tokens(task_id, norm_token_info, self.heavy_llm)
            except Exception:
                pass

            # Check if this is a direct answer instead of DSL
            if response.strip().startswith("Answer:"):
                direct_answer = response.strip()[7:].strip()  # Remove "Answer:" prefix
                return {"direct_answer": direct_answer}

            flow = self._parse_text_dsl(response)
            
            # DEBUG: Show the structure more clearly
            def print_flow_structure(flow_item, indent=0):
                spaces = "  " * indent
                if isinstance(flow_item, list) and len(flow_item) > 0:
                    if flow_item[0] == "WHILE":
                        print(f"{spaces}WHILE {flow_item[1]}")
                        for sub_item in flow_item[2]:
                            print_flow_structure(sub_item, indent + 1)
                        print(f"{spaces}ENDWHILE")
                    elif flow_item[0] == "IF":
                        print(f"{spaces}IF {flow_item[1]}")
                        for sub_item in flow_item[2]:
                            print_flow_structure(sub_item, indent + 1)
                        if len(flow_item) > 3 and flow_item[3]:
                            print(f"{spaces}ELSE")
                            for sub_item in flow_item[3]:
                                print_flow_structure(sub_item, indent + 1)
                        print(f"{spaces}ENDIF")
                    else:
                        print(f"{spaces}{' '.join(str(x) for x in flow_item)}")
                else:
                    print(f"{spaces}{flow_item}")
            

            return {"flow": flow}

        except Exception:
            # Return empty flow on any error, not direct answer
            return {"flow": []}

    def _parse_text_dsl(self, text: str) -> List:
        """Parse text DSL into array format for execution"""
        lines = [line.rstrip() for line in text.strip().split('\n') if line.strip()]
        flow = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            i += 1
            
            # Basic commands
            simple = self._parse_simple_command(line)
            if simple:
                flow.append(simple)
                continue
                    
            if line.strip().upper().startswith('IF'):
                # Conditional: IF condition or IF (condition) -> parse block
                condition = line.strip()[2:].strip()
                # Remove surrounding parentheses if present
                if condition.startswith('(') and condition.endswith(')'):
                    condition = condition[1:-1].strip()
                then_block, else_block, i = self._parse_conditional_block(lines, i, 2)
                flow.append(["IF", condition, then_block, else_block])
                
            elif line.strip().startswith('WHILE '):
                # While loop: WHILE condition -> parse block
                condition = line.strip()[6:].strip()
                body_block, i = self._parse_while_block(lines, i)
                flow.append(["WHILE", condition, body_block])
        
        return flow

    def _parse_simple_command(self, line: str):
        """Parse a single-line basic command (A/F/W/STOP). Returns list or None."""
        s = line.strip()
        if s.startswith('W '):
            minutes = int(s.split()[1])
            return ["WAIT", minutes]
        if s.startswith('F '):
            tool = s.split()[1]
            return ["F", tool]
        if s.startswith('A '):
            parts = s.split()
            tool = parts[1]
            return ["A", tool]
        if s == 'STOP':
            return ["STOP"]
        return None
    
    def _parse_conditional_block(self, lines: List[str], start_idx: int, expected_indent: int = 2):
        """Parse IF/ELSEIF/ELSE/ENDIF block"""
        then_block = []
        else_block = []
        current_block = then_block
        i = start_idx
        
        while i < len(lines):
            line = lines[i]  # Don't strip here - we need indentation
            i += 1
            
            stripped = line.strip()
            if stripped == 'ENDIF':
                break
            elif stripped.upper().startswith('ELSEIF') or stripped == 'ELSE':
                current_block = else_block
                if stripped.upper().startswith('ELSEIF'):
                    # Convert ELSEIF to nested IF in else block
                    condition = stripped[6:].strip()
                    if condition.startswith('(') and condition.endswith(')'):
                        condition = condition[1:-1].strip()
                    nested_then, nested_else, i = self._parse_conditional_block(lines, i, expected_indent)
                    else_block.append(["IF", condition, nested_then, nested_else])
                    break
                continue
            elif line.startswith(' ' * expected_indent):
                # Indented line - part of current block
                line_content = line[expected_indent:]  # Remove expected indentation
                
                if line_content.upper().startswith('IF'):
                    # Nested conditional - use deeper indentation
                    condition = line_content[2:].strip()
                    if condition.startswith('(') and condition.endswith(')'):
                        condition = condition[1:-1].strip()
                    nested_then, nested_else, new_i = self._parse_conditional_block(lines, i, expected_indent + 2)
                    current_block.append(["IF", condition, nested_then, nested_else])
                    i = new_i  # Update index to skip processed lines
                else:
                    cmd = self._parse_simple_command(line_content)
                    if cmd:
                        current_block.append(cmd)
        
        return then_block, else_block, i
    
    def _parse_while_block(self, lines: List[str], start_idx: int):
        """Parse WHILE/ENDWHILE block"""
        body_block = []
        i = start_idx
        expected_indent = 4  # WHILE body uses 4-space indentation
        
        while i < len(lines):
            line = lines[i]  # Don't strip - need indentation
            i += 1
            
            if line.strip() == 'ENDWHILE':
                break
            elif line.startswith(' ' * expected_indent):
                # Indented line - part of while body
                line_content = line[expected_indent:]  # Remove expected indentation
                if line_content.startswith('IF '):
                    # Nested conditional in while loop
                    condition = line_content[3:].strip()
                    if condition.startswith('(') and condition.endswith(')'):
                        condition = condition[1:-1].strip()
                    then_block, else_block, new_i = self._parse_conditional_block(lines, i, expected_indent + 4)
                    body_block.append(["IF", condition, then_block, else_block])
                    i = new_i  # Update index to skip processed lines
                else:
                    cmd = self._parse_simple_command(line_content)
                    if cmd:
                        body_block.append(cmd)
        
        return body_block, i

    async def _execute_dsl_flow(self, flow: List, task_memory, task_id: str, original_message: str, parent_fetch_result: str = "", skip_validation: bool = False) -> Dict[str, Any]:
        """Execute the new DSL flow structure"""
        execution_start = time.time()
        # If a delegate already redirected this task, stop executing here
        if task_id in getattr(self, '_redirects', {}):
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
                    result = await self.tools.execute_tool(tool, context, task_id=task_id, task_memory=task_memory, light_llm=self.light_llm, heavy_llm=self.heavy_llm, agent_id=self.agent_id, validation_mode=self.validation_mode)
                    last_fetch_result = result
                    
                    # Check if websearch tool has token info to add to task
                    if tool == "websearch" and hasattr(self.tools.tools["websearch"], '_last_token_info'):
                        token_info = getattr(self.tools.tools["websearch"], '_last_token_info')
                        if token_info:
                            # Add websearch tokens to task tracking
                            for _ in range(token_info.get("llm_calls", 0)):
                                self.logger.add_tokens(task_id, {
                                    "input_tokens": token_info.get("input_tokens", 0) // token_info.get("llm_calls", 1),
                                    "output_tokens": token_info.get("output_tokens", 0) // token_info.get("llm_calls", 1),
                                    "total_tokens": token_info.get("tokens_used", 0) // token_info.get("llm_calls", 1)
                                }, self.light_llm)
                
                elif command == "A":  # Action
                    tool = step[1]
                    console.info(f"Executing action", f"Tool: {tool}", task_id=task_id, agent_id=self.agent_id)
                    context = original_message
                    result = await self.tools.execute_tool(tool, context, task_id=task_id, task_memory=task_memory, light_llm=self.light_llm, heavy_llm=self.heavy_llm, agent_id=self.agent_id, validation_mode=self.validation_mode)
                    
                    # Check if websearch tool has token info to add to task
                    if tool == "websearch" and hasattr(self.tools.tools["websearch"], '_last_token_info'):
                        token_info = getattr(self.tools.tools["websearch"], '_last_token_info')
                        if token_info:
                            # Add websearch tokens to task tracking
                            for _ in range(token_info.get("llm_calls", 0)):
                                self.logger.add_tokens(task_id, {
                                    "input_tokens": token_info.get("input_tokens", 0) // token_info.get("llm_calls", 1),
                                    "output_tokens": token_info.get("output_tokens", 0) // token_info.get("llm_calls", 1),
                                    "total_tokens": token_info.get("tokens_used", 0) // token_info.get("llm_calls", 1)
                                }, self.light_llm)
                
                elif command == "IF":  # Conditional
                    condition = step[1]
                    then_steps = step[2] if len(step) > 2 else []
                    else_steps = step[3] if len(step) > 3 else []
                    
                    # Evaluate condition against last fetch result
                    condition_result, condition_time = await self._evaluate_dsl_condition(condition, last_fetch_result, task_memory, task_id, original_message)
                    
                    if condition_result:
                        console.info(f"✅ Condition met: {condition}", f"Executing THEN steps (Time: {condition_time:.2f}s)", task_id=task_id, agent_id=self.agent_id)
                        sub_result = await self._execute_dsl_flow(then_steps, task_memory, task_id, original_message, last_fetch_result, skip_validation)
                        # Only return early if STOP was used, or if not in skip-validation mode
                        if sub_result.get("stopped_via_command", False):
                            return sub_result
                        if not skip_validation and sub_result.get("completed", False):
                            return sub_result
                    else:
                        if else_steps:
                            console.info(f"❌ Condition not met: {condition}", f"Executing ELSE steps (Time: {condition_time:.2f}s)", task_id=task_id, agent_id=self.agent_id)
                            sub_result = await self._execute_dsl_flow(else_steps, task_memory, task_id, original_message, last_fetch_result, skip_validation)
                            # Only return early if STOP was used, or if not in skip-validation mode
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
                        sub_result = await self._execute_dsl_flow(body_steps, task_memory, task_id, original_message, last_fetch_result, skip_validation=True)
                        
                        # Check if STOP command was executed (always exit on STOP, even for infinite loops)
                        if sub_result.get("completed", False) and sub_result.get("stopped_via_command", False):
                            return sub_result
                        
                        # For finite loops, exit if sub-result is "completed"
                        if condition.lower() not in ["true", "1", "always"] and sub_result.get("completed", False):
                            return sub_result
                        
                        iteration += 1
                
                elif command == "WAIT":  # Wait/Sleep
                    minutes = step[1] if len(step) > 1 else 1
                    seconds = minutes / 60  # Fast-simulate minutes as seconds for examples/tests
                    console.info(f"Waiting", f"{minutes} minutes ({seconds}s)", task_id=task_id, agent_id=self.agent_id)
                    
                    # Track wait time for computational timing
                    wait_start = time.time()
                    await asyncio.sleep(seconds)
                    wait_duration = time.time() - wait_start
                    
                    # Add to total wait time for this task
                    if task_id not in self.wait_times:
                        self.wait_times[task_id] = 0.0
                    self.wait_times[task_id] += wait_duration
                
                elif command == "STOP":  # Stop execution
                    console.info(f"STOP command reached", "Completing task", task_id=task_id, agent_id=self.agent_id)
                    
                    # Run validation to get proper final message
                    validation_start = time.time()
                    validation = await self.llm_validate_completion(original_message, task_memory, task_id)
                    validation_time = time.time() - validation_start
                    console.debug("STOP validation completed", f"Time: {validation_time:.2f}s", task_id=task_id, agent_id=self.agent_id)
                    
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
                            console.info("Task incomplete", "Re-analyzing with additional context", task_id=task_id, agent_id=self.agent_id)
                            
                            # Create extended task description with continue guidance
                            extended_task = f"{original_message}\n\nPrevious progress context: {continue_message}"
                            
                            # Re-analyze the task with the additional context
                            analysis_start = time.time()
                            new_analysis = await self.llm_analyze_task(extended_task, task_memory, task_id)
                            analysis_time = time.time() - analysis_start
                            console.debug("Re-analysis completed", f"Time: {analysis_time:.2f}s", task_id=task_id, agent_id=self.agent_id)
                            
                            # Check if re-analysis resulted in a direct answer
                            if "direct_answer" in new_analysis:
                                console.debug("Direct answer provided on re-analysis", "Stopping execution", task_id=task_id, agent_id=self.agent_id)
                                execution_time = time.time() - execution_start
                                return {
                                    "completed": True,
                                    "final_message": new_analysis["direct_answer"],
                                    "execution_time": execution_time
                                }
                            
                            if new_analysis and new_analysis.get("flow"):
                                console.debug("Re-analysis generated new plan", "Continuing with new flow", task_id=task_id, agent_id=self.agent_id)
                                # Continue with new analysis - restart the loop
                                return await self._execute_dsl_flow(new_analysis["flow"], task_memory, task_id, original_message, "", skip_validation)
                            else:
                                console.warning("Re-analysis failed", "Completing with partial results", task_id=task_id, agent_id=self.agent_id)
                        
                        # If no continue_message or re-analysis failed, complete with current state
                        execution_time = time.time() - execution_start
                        return {
                            "completed": False,
                            "stopped_via_command": True,
                            "final_message": validation.get("final_message", "Task partially completed via STOP command"),
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
            console.debug("DSL flow validation completed", f"Time: {validation_time:.2f}s", task_id=task_id, agent_id=self.agent_id)
            
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

    async def _evaluate_dsl_condition(self, condition: str, last_result: str, task_memory, task_id: str, original_message: str = "") -> tuple[bool, float]:
        """Evaluate DSL condition against task memory and last result"""
        condition_start = time.time()
        
        # Optimize obvious conditions without LLM call
        condition_lower = condition.lower().strip()
        if condition_lower in ["true", "1", "always"]:
            condition_time = time.time() - condition_start
            return True, condition_time
        elif condition_lower in ["false", "0", "never"]:
            condition_time = time.time() - condition_start
            return False, condition_time
        
        # Get task memory for context (truncate to reduce tokens)
        memory_content = task_memory.get() if task_memory else ""
        memory_lines = memory_content.splitlines() if memory_content else []
        memory_content_lite = "\n".join(memory_lines[-15:])  # last 15 lines only

        # Single compact prompt for any condition; let LLM decide using provided context
        # Make the prompt more explicit about what to check
        if "AND" in condition or "found" in condition.lower() or "both" in condition.lower():
            # This is a completion check - scan all memory
            prompt = (
                f"Task: {original_message}\n"
                f"Question: {condition}\n"
                f"Memory:\n{memory_content_lite}\n\n"
                f"Answer with JSON: {{\"met\": true/false}}"
            )
        else:
            # This is an immediate check - use most recent memory entry
            recent_entry = memory_lines[-1] if memory_lines else ""
            # Extract just the core content without the timestamp/numbering
            if recent_entry and "] " in recent_entry:
                recent_content = recent_entry.split("] ", 1)[1] if "] " in recent_entry else recent_entry
            else:
                recent_content = recent_entry
                
            prompt = (
                f"Task: {original_message}\n"
                f"Question: {condition}\n"
                f"Recent: {recent_content}\n\n"
                f"Answer with JSON: {{\"met\": true/false}}"
            )
        
        try:
            llm_result = await llm_completion_async(
                model=self.light_llm,
                prompt=prompt,
                temperature=0.0,
                max_tokens=20,
                response_format="json"
            )

            response, norm_token_info = self._normalize_llm_result(llm_result)
            try:
                self.logger.add_tokens(task_id, norm_token_info, self.light_llm)
            except Exception:
                pass

            result = json.loads(response.strip())
            is_met = result.get("met", False)

        except Exception as e:
            is_met = False
            
        condition_time = time.time() - condition_start
        return is_met, condition_time

    async def llm_validate_completion(self, original_message: str, task_memory, task_id: str):
        # Get full memory content for validation
        memory_content = task_memory.get() if task_memory else ""
        mem_lines = memory_content.splitlines() if memory_content else []
        memory_content_lite = "\n".join(mem_lines[-15:])  
        validation_prompt = (
            f"Task: {original_message}\n"
            f"Results:\n{memory_content_lite}\n\n"
            f"Return JSON only. If done, set final_message (leave continue_message empty). "
            f"If not done, set continue_message (leave final_message empty).\n\n"
            f"{{\"final_message\": \"\", \"continue_message\": \"\"}}"
        )
        try:
            llm_result = await llm_completion_async(
                model=self.light_llm, prompt=validation_prompt, temperature=0.1, max_tokens=500, response_format="json"
            )

            response, norm_token_info = self._normalize_llm_result(llm_result)
            self.logger.add_tokens(task_id, norm_token_info, self.light_llm)

            # Clean up common JSON formatting issues
            response_clean = response.strip()
            
            # Fix common malformed patterns like extra commas and quotes
            # Remove patterns like: .", ",\n" ,
            response_clean = re.sub(r'\."\s*,\s*",\s*\\n"\s*,', '.",', response_clean)
            # Remove patterns like: .", ",
            response_clean = re.sub(r'\."\s*,\s*",', '.",', response_clean)
            # Remove extra commas before closing braces
            response_clean = re.sub(r',\s*}', '}', response_clean)
            
            result = json.loads(response_clean)
            final_message = result.get("final_message", "").strip()
            continue_message = result.get("continue_message", "").strip()
            
            # Determine completion based on which message is populated
            completed = bool(final_message and not continue_message)
            return {
                "completed": completed,
                "final_message": final_message if final_message else "Task execution completed",
                "continue_message": continue_message
            }
            
        except json.JSONDecodeError as e:
            # Try to extract just the final_message content if JSON is malformed
            try:
                # Look for final_message content between quotes
                final_msg_match = re.search(r'"final_message":\s*"([^"]*(?:\\"[^"]*)*)"', response)
                if final_msg_match:
                    final_message = final_msg_match.group(1).replace('\\"', '"')
                    return {
                        "completed": True,
                        "final_message": final_message,
                        "continue_message": ""
                    }
            except Exception:
                pass
        except Exception as e:
            console.error("LLM validation failed", str(e), task_id=task_id, agent_id=self.agent_id)
            return {"completed": True, "final_message": "Task execution completed successfully"}

    async def llm_delegate(self, message: str, task_id: str) -> str:
        """Fast delegate LLM: returns a single agent name (exact) or 'NONE' if no delegation."""
        # If no team attached, never delegate
        if not hasattr(self, 'team') or not getattr(self, 'team'):
            return 'NONE'
        # Build a compact numbered team description using available tools for each agent.
        # Team stores agents as a dict of name -> Agent
        member_entries = []
        index_to_name = []
        idx = 1
        for name, a in getattr(self.team, 'agents', {}).items():
            tools_obj = getattr(a, 'tools', None) or getattr(getattr(a, 'orchestrator', None), 'tools', None)
            tool_names = []
            try:
                if isinstance(tools_obj, (list, tuple, set)):
                    tool_names = [str(t) for t in tools_obj]
                elif hasattr(tools_obj, 'tools') and isinstance(getattr(tools_obj, 'tools'), dict):
                    tool_names = list(getattr(tools_obj, 'tools').keys())
                elif isinstance(tools_obj, dict):
                    tool_names = list(tools_obj.keys())
            except Exception:
                tool_names = []

            if tool_names:
                entry = f"{idx}.{name} ({', '.join(tool_names)})"
            else:
                entry = f"{idx}.{name}"
            member_entries.append(entry)
            index_to_name.append(name)
            idx += 1
        member_list = " and ".join(member_entries)


        prompt = f"""You are a fast router. Task: {message}
        Team agents: {member_list}
        Decide which agent is most suitable for this task based on the tools each agent has. Output only a single number: the index of the chosen agent (e.g. 1 or 2). If this agent (YOU) should handle it locally, output 0. Do not output names, IDs, punctuation or any extra text.

        Example:
        Task: "Post the latest admin email to Slack"
        Team agents: 1.backend-dev (jira, slack) and 2.qa-engineer (gmail, slack)
        Output: 2

        """
        try:
            llm_result = await llm_completion_async(model=self.light_llm, prompt=prompt, temperature=0.0, max_tokens=6)
            response, norm_token_info = self._normalize_llm_result(llm_result)
            try:
                self.logger.add_tokens(task_id, norm_token_info, self.light_llm)
            except Exception:
                pass
            raw = response.strip().splitlines()[0].strip() if response else ''
            try:
                console.debug("Delegate raw response", raw, task_id=task_id, agent_id=self.agent_id)
            except Exception:
                pass

            # Expect a single number. If 0 -> handle locally (NONE). If N -> map to Nth agent.
            num = None
            try:
                num = int(raw)
            except Exception:
                # Not a number -> treat as NONE (do not delegate)
                return 'NONE'

            if num == 0:
                return 'NONE'
            if 1 <= num <= len(index_to_name):
                return index_to_name[num-1]
            return 'NONE'
        except Exception:
            return 'NONE'

    async def _delegate_worker(self, message: str, task_id: str):
        """Background worker that asks llm_delegate and forwards task to selected agent by name."""
        try:
            choice = await self.llm_delegate(message, task_id)
            # Respect the model decision: if it returned NONE, do not fallback to heuristics
            if not choice or choice == 'NONE':
                return

            # Find agent by name in the team's agents dict
            team = getattr(self, 'team', None)
            if not team:
                return

            target_agent = None
            # team.agents is a dict: name -> Agent
            for name, a in getattr(team, 'agents', {}).items():
                if name == choice or name.lower() == choice.lower() or a.agent_id == choice:
                    target_agent = a
                    target_name = name
                    break

            if not target_agent:
                console.debug("Delegation: target not found", choice, task_id=task_id, agent_id=self.agent_id)
                return

            # If the chosen target is this same agent, ignore the delegation to avoid self-redirects
            try:
                if getattr(target_agent, 'agent_id', None) == getattr(self, 'agent_id', None):
                    console.debug("Delegation: chosen target is self; ignoring redirect", target_name, task_id=task_id, agent_id=self.agent_id)
                    return
            except Exception:
                pass

            # Mark redirect so local execution stops

            self._redirects[task_id] = {'target_name': target_name}

            console.task(f"Task {task_id} REDIRECTED [Agent: {self.agent_id}] -> {target_name}", task_id=task_id, agent_id=self.agent_id)

            # Ensure target agent is running (best-effort)
            try:
                if hasattr(target_agent, 'start'):
                    target_agent.start()
            except Exception:
                pass

            # Forward message to target agent's receive_message or run
            try:
                # Send a small envelope so the receiving agent knows this was delegated and from whom
                envelope = {'_forwarded': True, 'origin_agent': self.agent_id, 'origin_task_id': task_id, 'payload': message}
                if hasattr(target_agent, 'receive_message'):
                    target_agent.receive_message(envelope)
                elif hasattr(target_agent, 'run'):
                    # run may be a method to enqueue/start processing
                    target_agent.run(envelope)
                else:
                    console.debug("Delegation: target has no enqueue method", target_agent.agent_id, task_id=task_id, agent_id=self.agent_id)
            except Exception as e:
                console.error("Delegation forward failed", str(e), task_id=task_id, agent_id=self.agent_id)

        except Exception:
            pass
