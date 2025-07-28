# filepath: c:\Users\david\OneDrive\Documentos\Github\Hierarchical-LLM-Router\hlr_agent\c_layers\C1.py
import threading
import queue
import asyncio
from .C0.C0 import C0
from .C0.graph import Graph
from ..memory.AgentMemory import AgentMemory

class C1:
    def __init__(self, base_graph: Graph):
        self.message_queue = queue.Queue()
        self.scheduler_thread = None
        self.running = False
        self.base_graph = base_graph
        self.logger = None  # Will be set by Agent if logging is enabled
        
        # Initialize agent memory
        self.agent_memory = AgentMemory(f"C1-Agent-{id(self)}", max_tasks=50)
    
    def set_logger(self, logger):
        """Set the task logger."""
        self.logger = logger
    
    def start(self):
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
        self.scheduler_thread.start()
    
    def stop(self):
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
    
    def receive_message(self, message):
        if self.running:
            self.message_queue.put(message)
            return True
        return False
    
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
                    
                    done_tasks = {t for t in running_tasks if t.done()}
                    running_tasks -= done_tasks
                    
                    self.message_queue.task_done()
                    
                except queue.Empty:
                    await asyncio.sleep(0.1)  # Reduced from 10ms to 100ms polling
                    continue
                except Exception as e:
                    print(f"❌ Scheduler queue error: {e}")
                    await asyncio.sleep(0.1)
        
        try:
            loop.run_until_complete(scheduler_loop())
            
            if running_tasks:
                loop.run_until_complete(asyncio.gather(*running_tasks, return_exceptions=True))
                
        except Exception as e:
            print(f"❌ Scheduler loop error: {e}")
        finally:
            loop.close()
    
    async def _process_message_async(self, message):
        import time
        from datetime import datetime
        
        # Use high-resolution timestamp for efficient task ID
        start_time = time.time()
        task_id = f"{int(start_time * 1000) % 100000000:08d}"  # 8-digit ID from timestamp
        
        # Create task memory through agent memory
        task_memory = self.agent_memory.create_task_memory(f"Task-{task_id}")
        
        # Log task creation in agent memory summary
        print(f"[C1] Created task memory for {task_id}. Current Tasks: {self.agent_memory.get_task_count()}, Message: {message[:40]}...")
        
        try:
            filtered_graph, frequency, analysis_tokens, has_completion_condition = await self.llm_schedule_and_filter(message, self.base_graph)
        except Exception as e:
            print(f"❌ [C1] Task analysis error: {e}")
            print(f"🔄 [C1] Fallback: Using original graph with no frequency")
            filtered_graph, frequency, analysis_tokens, has_completion_condition = self.base_graph, 0, 0, False
        
        # Start logging after we have task analysis info
        if self.logger:
            is_periodic = frequency > 0
            self.logger.start_task(task_id, message, is_periodic, frequency, has_completion_condition)
            
        # Track analysis tokens
        if self.logger and analysis_tokens:
            self.logger.add_tokens(task_id, analysis_tokens)
        
        try:
            # Use shallow copy for better performance - graph state is isolated per task anyway
            task_graph = filtered_graph
            task_graph.reset()  # Reset state instead of deep copying
            
            c0_instance = C0(task_graph)
            
            # Pass logger to C0 if available
            if self.logger:
                c0_instance.set_logger(self.logger, task_id)
                # Also set logger on the graph
                task_graph.set_logger(self.logger, task_id)
            
            await c0_instance.run_iteration(
                message=message,
                memory=task_memory,
                frequency=frequency if frequency else 0,
                max_iterations=1 if frequency == 0 else -1
            )
            
            # Calculate task duration using cached start time
            task_duration = time.time() - start_time
            
            # Get tokens used if logger is available
            tokens_info = ""
            if self.logger and task_id in self.logger.active_tasks:
                tokens_used = self.logger.active_tasks[task_id].get('tokens_used', 0)
                tokens_info = f" | Tokens: {tokens_used}"
                
            print(f"[C1] Route: {task_graph.context.get('route', 'N/A')}")
            print(f"[C1] Task {task_id} completed successfully. Duration: {task_duration:.2f}s{tokens_info}")
            
            # Complete logging if enabled
            if self.logger:
                self.logger.complete_task(task_id, "completed")
            
        except Exception as e:
            error_time = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S.%f')[:-3]
            print(f"❌ [Task-{task_id}] Error at {error_time}: {e}")
            
            # Still keep the task memory for debugging
            task_memory.set(f"ERROR: {str(e)}")
            
            # Log error if enabled
            if self.logger:
                self.logger.complete_task(task_id, "error")
    
    async def llm_schedule_and_filter(self, user_query: str, base_graph: Graph):
        """Comprehensive task analysis using LLM to determine tools, routing, and scheduling."""
        from ..internal.llm import llm_completion_async
        import json
        
        all_available_tools = [node_id for node_id in base_graph.nodes.keys() 
                              if node_id not in ['Input', 'Output']]
        
        analysis_prompt = f"""Task: "{user_query}"
Tools: {', '.join(all_available_tools)}

Return valid JSON:
{{
    "frequency_minutes": int,
    "has_completion_condition": bool,
    "optimal_routing": {{"Input": ["tool"], "tool": ["Output"]}}
}}

frequency_minutes: 0=once, 60=hourly, 1440=daily...
has_completion_condition: true if task has stop criteria ("until X happens"), false otherwise
Use exact tool names from list above."""
        
        try:
            analysis_result, tokens = await llm_completion_async(
                model=base_graph.heavy_llm,
                prompt=analysis_prompt,
                temperature=0.1,
                max_tokens=150,
                response_format="json"
            )
            
            task_analysis = json.loads(analysis_result)
            
            # Extract analysis results
            frequency_minutes = task_analysis.get('frequency_minutes', 0)
            has_completion_condition = task_analysis.get('has_completion_condition', False)
            optimal_routing = task_analysis.get('optimal_routing', {})
            
            # Derive required tools from routing efficiently
            required_tools = set()
            for node, children in optimal_routing.items():
                if node not in ('Input', 'Output'):
                    required_tools.add(node)
                if isinstance(children, list):
                    required_tools.update(child for child in children 
                                        if child not in ('Input', 'Output'))
            required_tools = list(required_tools)
            
            # Determine if periodic based on frequency
            is_periodic = frequency_minutes > 0
            
            print(f"[C1] Task analysis: {'Periodic' if is_periodic else 'One-time'}, Tools: {required_tools}, Frequency: {frequency_minutes} minutes, Completion check: {has_completion_condition}")
            
            # Create optimized graph based on LLM analysis
            optimized_graph = self._create_optimized_graph(
                required_tools, optimal_routing, base_graph
            )
            
            # Set completion condition based on LLM analysis
            optimized_graph._has_completion_condition = has_completion_condition
            frequency = frequency_minutes * 60 if is_periodic else 0  # Convert minutes to seconds
            return optimized_graph, frequency, tokens, has_completion_condition
            
        except Exception as e:
            print(f"⚠️ [C1] Analysis failed: {e}, using fallback")
            base_graph._has_completion_condition = False
            return base_graph, 0, 0, False
    
    def _create_optimized_graph(self, required_tools: list, optimal_routing: dict, base_graph: Graph):
        """Create an optimized graph based on LLM analysis."""
        from .C0.node import Node
        from .C0.graph import Graph as OptimizedGraph
        
        # Get available tools from base graph
        available_tools = [node_id for node_id in base_graph.nodes.keys() 
                          if node_id not in ['Input', 'Output']]
        
        # Validate that required tools exist in base graph
        valid_tools = [tool for tool in required_tools if tool in base_graph.nodes]
        
        if not valid_tools:
            print(f"⚠️ [C1] No valid tools found, using all available tools")
            valid_tools = available_tools
        
        # Create optimized nodes
        optimized_nodes = []
        
        # Create Input node with LLM-optimized routing
        input_children = optimal_routing.get('Input', valid_tools + ['Output'])
        # Only add Output if not already specified in routing
        if 'Output' not in input_children and len(input_children) == 0:
            input_children.append('Output')
        optimized_nodes.append(Node("Input", children=input_children))
        
        # Create tool nodes with LLM-optimized connections
        for tool_name in valid_tools:
            if tool_name in base_graph.nodes:
                original_node = base_graph.nodes[tool_name]
                
                # Use LLM-suggested routing or fallback to default
                tool_children = optimal_routing.get(tool_name, ['Output'])
                
                # Only add Output if not already specified and no other children exist
                if 'Output' not in tool_children and len(tool_children) == 0:
                    tool_children.append('Output')
                
                optimized_nodes.append(Node(
                    node_id=tool_name,
                    children=tool_children,
                    func=original_node.func,
                    description=original_node.description
                ))
        
        # Create Output node
        optimized_nodes.append(Node(
            node_id="Output",
            children=None,
            func=base_graph.nodes["Output"].func,
            description=base_graph.nodes["Output"].description
        ))
        
        # Create optimized graph
        optimized_graph = OptimizedGraph(
            nodes=optimized_nodes,
            start_node_id="Input",
            end_node_id="Output",
            light_llm=base_graph.light_llm,
            heavy_llm=base_graph.heavy_llm
        )
        
        return optimized_graph