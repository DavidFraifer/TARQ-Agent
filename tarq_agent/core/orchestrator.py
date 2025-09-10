from ..utils.logger import TARQLogger
from ..utils.console import console
from ..utils.llm_utils import normalize_llm_result
from ..tools.tool import ToolContainer
from ..tools.internal_tools import internal_tools
from ..memory.AgentMemory import AgentMemory
from ..internal.llm import llm_completion_async
from .dsl_parser import DSLParser
from .dsl_executor import DSLExecutor
from .delegation_manager import DelegationManager
import asyncio, threading, queue, time, uuid

class Orchestrator:
    # Constants
    MAX_WAIT_TIMES_ENTRIES = 100  # Limit concurrent task tracking
    
    def __init__(self, light_llm: str, heavy_llm: str, logger: TARQLogger, agent_id: str = "unknown", 
                 disable_delegation: bool = False, rag_engine=None, validation_mode: bool = False):
        # Core configuration
        self.logger = logger
        self.light_llm = light_llm
        self.heavy_llm = heavy_llm
        self.rag_engine = rag_engine
        self.agent_id = agent_id
        self.disable_delegation = disable_delegation
        self.validation_mode = validation_mode
        
        # Initialize components
        self.tools = ToolContainer()
        self.tool_descriptions = {}
        self.message_queue = queue.Queue()
        self.scheduler_thread = None
        self.running = False
        self.agent_memory = AgentMemory(f"Orchestrator-Agent-{agent_id}", max_tasks=50)
        self.wait_times = {}
        
        # Initialize component modules
        self.dsl_parser = DSLParser()
        self.dsl_executor = DSLExecutor(self.tools, self.logger, self.light_llm, self.heavy_llm, 
                                       self.agent_id, self.validation_mode)
        self.delegation_manager = DelegationManager(self.logger, self.light_llm, self.agent_id)

        # Add internal tools
        for tool_name, tool_func in internal_tools.items():
            self.tools.add_tool(tool_name, tool_func)

    def add_tool(self, name: str, func, description: str = None):
        self.tools.add_tool(name, func)
        if description: 
            self.tool_descriptions[name] = description

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
        return self.message_queue.put(message) if self.running else False
    
    def _cleanup_wait_times_if_needed(self):
        """Cleanup old wait_times entries if limit exceeded to prevent memory leak"""
        if len(self.wait_times) > self.MAX_WAIT_TIMES_ENTRIES:
            # Keep only the most recent MAX_WAIT_TIMES_ENTRIES // 2 entries
            keep_count = self.MAX_WAIT_TIMES_ENTRIES // 2
            # Sort by task creation time (extract timestamp from task-XXXXXXXX format)
            sorted_tasks = sorted(self.wait_times.keys())
            tasks_to_remove = sorted_tasks[:-keep_count] if len(sorted_tasks) > keep_count else []
            for task_id in tasks_to_remove:
                self.wait_times.pop(task_id, None)
    
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
        payload = message
        
        if isinstance(message, dict) and message.get('_forwarded'):
            is_forwarded = True
            payload = message.get('payload', '')

        # Initialize wait time tracking and logging
        self.wait_times[task_id] = 0.0
        display_message = payload[:60] + "..." if isinstance(payload, str) and len(payload) > 60 else payload
        console.task(f"Task {task_id} ADDED [Agent: {self.agent_id}] - {display_message}", task_id=task_id, agent_id=self.agent_id)
        self.logger.start_task(task_id, message, self.agent_id)

        # Start delegation worker in background if part of a team and not forwarded
        delegate_task = None
        try:
            if (hasattr(self, 'team') and getattr(self, 'team') and 
                not is_forwarded and not self.disable_delegation):
                delegate_task = asyncio.create_task(
                    self.delegation_manager.execute_delegation(payload, task_id, self.team, self.agent_id)
                )
        except Exception: 
            delegate_task = None

        try:
            # Step 1: Initial Analysis with timing
            analysis_start = time.time()
            analysis = await self.llm_analyze_task(payload, task_memory, task_id)
            analysis_time = time.time() - analysis_start

            # Wait for delegate to finish if still running
            if delegate_task and not delegate_task.done():
                try: 
                    await delegate_task
                except Exception: 
                    pass  # delegate failure is non-fatal

            # Process analysis results
            if self.delegation_manager.is_task_redirected(task_id):
                info = self.delegation_manager.get_redirect_info(task_id)
                target = info.get('target_name', 'unknown')
                console.debug("Task analysis suppressed due to redirect", f"Forwarded to {target}", 
                            task_id=task_id, agent_id=self.agent_id)
                result = {"completed": False, "status": "warning", "final_message": f"Task redirected to {target}"}
            else:
                if "direct_answer" in analysis:
                    console.debug("Direct answer provided", "Skipping DSL execution to save tokens", 
                                task_id=task_id, agent_id=self.agent_id)
                    result = {"completed": True, "status": "success", "final_message": analysis["direct_answer"]}
                elif "error" in analysis:
                    console.debug("DSL syntax validation failed", "Completing task with error", 
                                task_id=task_id, agent_id=self.agent_id)
                    result = {"completed": True, "status": "error", "final_message": f"DSL Syntax Error: {analysis['error']}"}
                else:
                    console.debug("Task analysis completed", f"Time: {analysis_time:.2f}s", 
                                task_id=task_id, agent_id=self.agent_id)
                    flow = analysis.get("flow", [])
                    if flow:
                        # Share wait_times reference safely (DSL executor only reads from it)
                        self.dsl_executor.wait_times = self.wait_times
                        result = await self.dsl_executor.execute_dsl_flow(flow, task_memory, task_id, payload)
                        
                        # Handle re-analysis if task needs continuation
                        if result.get("needs_reanalysis", False) and result.get("continue_message"):
                            console.info("Task needs continuation", "Re-analyzing with additional context", 
                                       task_id=task_id, agent_id=self.agent_id)
                            continue_message = result.get("continue_message", "")
                            reanalysis = await self.llm_analyze_task(continue_message, task_memory, task_id)
                            
                            if "flow" in reanalysis and reanalysis["flow"]:
                                console.debug("Continuation flow generated", "Executing additional steps", 
                                            task_id=task_id, agent_id=self.agent_id)
                                continuation_result = await self.dsl_executor.execute_dsl_flow(
                                    reanalysis["flow"], task_memory, task_id, continue_message)
                                # Merge results, keeping the original execution time
                                result.update({
                                    "completed": continuation_result.get("completed", result.get("completed")),
                                    "status": continuation_result.get("status", result.get("status")),
                                    "final_message": continuation_result.get("final_message", result.get("final_message"))
                                })
                    else:
                        result = {"completed": True, "status": "success", "final_message": "Task completed successfully"}
            
            # Calculate timing and get token info
            task_duration = time.time() - start_time
            total_wait_time = self.wait_times.get(task_id, 0.0)
            computational_time = task_duration - total_wait_time
            
            task_data = self.logger.active_tasks.get(task_id, {})
            token_info = {
                'tokens_used': task_data.get('tokens_used', 0), 
                'input_tokens': task_data.get('input_tokens', 0),
                'output_tokens': task_data.get('output_tokens', 0), 
                'llm_calls': task_data.get('llm_calls', 0),
                'total_cost': task_data.get('total_cost', 0.0)
            }
            
            # Generate task summary
            status = "completed" if result.get("completed", True) else "incomplete"
            task_status = result.get("status", "success")
            final_message = result.get('final_message', 
                                     "Task execution finished" if status == "completed" else "Task incomplete")
            
            console.task_summary(task_id, task_duration, token_info, status, final_message, 
                               computational_time, agent_id=self.agent_id, task_status=task_status)
            self.logger.complete_task(task_id, status, computational_time)
            
            # Cleanup completed task from wait_times to prevent memory leak
            self.wait_times.pop(task_id, None)
            self._cleanup_wait_times_if_needed()
            
        except Exception as e:
            console.error("Task execution failed", str(e), task_id=task_id, agent_id=self.agent_id)
            task_memory.set(f"ERROR: {str(e)}")
            elapsed = time.time() - start_time
            comp_time = elapsed - self.wait_times.get(task_id, 0.0)
            self.logger.complete_task(task_id, "error", comp_time)
            
            # Cleanup failed task from wait_times to prevent memory leak
            self.wait_times.pop(task_id, None)
            self._cleanup_wait_times_if_needed()
    
    async def llm_analyze_task(self, message: str, task_memory, task_id: str):
        # Prepare task memory content
        memory_content = task_memory.get() if task_memory else ""

        # Get RAG context if available
        rag_context = ""
        if self.rag_engine and self.rag_engine.is_enabled():
            rag_context = self.rag_engine.get_context(message, max_length=500)
            if rag_context: 
                rag_context = f"\nKnowledge Base:\n{rag_context}\n"

        # Build tools display
        available_tools = list(self.tools.tools.keys())
        tools_info = []
        for tool_name in available_tools:
            if tool_name in self.tool_descriptions:
                tools_info.append(f"{tool_name}: {self.tool_descriptions[tool_name]}")
            else:
                tools_info.append(tool_name)
        tools_display = "[" + ", ".join(tools_info) + "]"

        # Build analysis prompt
        analysis_prompt = f"""
Task: "{message}"
Context: {memory_content}{rag_context}
Tools: {tools_display}

DSL:
W N=wait N min | F TOOL=fetch (before IF/WHILE) | A TOOL=action | IF/ELSEIF/ELSE/ENDIF=conditions | WHILE/ENDWHILE=loops | STOP=complete

IMPORTANT: For A commands, do not use parameters. 
Correct: A sheets
Wrong: A sheets(spreadsheet_name='2025')

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
- Memory = action history used to verify

If user needs only a normal answer (no tool use):  
Answer: [response]
"""
        
        # Execute LLM analysis
        try:
            llm_result = await llm_completion_async(
                model=self.heavy_llm, 
                prompt=analysis_prompt, 
                temperature=0.0, 
                max_tokens=100, 
                response_format=None
            )
            response, norm_token_info = normalize_llm_result(llm_result)
            
            # Log tokens safely
            try: 
                self.logger.add_tokens(task_id, norm_token_info, self.heavy_llm)
            except Exception: 
                pass
            
            # Handle direct answers
            if response.strip().startswith("Answer:"): 
                return {"direct_answer": response.strip()[7:].strip()}

            # Parse DSL response
            try: 
                flow = self.dsl_parser.parse_text_dsl(response)
            except ValueError as e:
                console.error(f"DSL Syntax Error", str(e), task_id=task_id, agent_id=self.agent_id)
                return {"error": str(e)}
            
            # Display parsed flow structure
            console.debug("Parsed DSL Flow Structure", "", task_id=task_id, agent_id=self.agent_id)
            self.dsl_parser.print_flow_structure(flow)
            return {"flow": flow}
            
        except Exception: 
            return {"flow": []}
