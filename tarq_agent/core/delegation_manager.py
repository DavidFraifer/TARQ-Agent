from ..utils.console import console
from ..internal.llm import llm_completion_async
import asyncio


class DelegationManager:
    """Handles task delegation to team members"""
    
    def __init__(self, logger, light_llm, agent_id):
        self.logger = logger
        self.light_llm = light_llm
        self.agent_id = agent_id
        self._redirects = {}

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

    async def delegate_task(self, message: str, task_id: str, team) -> str:
        """Fast delegate LLM: returns a single agent name (exact) or 'NONE' if no delegation."""
        if not team:
            return 'NONE'
            
        # Build a compact numbered team description using available tools for each agent
        member_entries, index_to_name = [], []
        idx = 1
        
        for name, agent in getattr(team, 'agents', {}).items():
            tools_obj = getattr(agent, 'tools', None) or getattr(getattr(agent, 'orchestrator', None), 'tools', None)
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

            entry = f"{idx}.{name} ({', '.join(tool_names)})" if tool_names else f"{idx}.{name}"
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

            try:
                num = int(raw)
            except Exception:
                return 'NONE'

            if num == 0:
                return 'NONE'
            if 1 <= num <= len(index_to_name):
                return index_to_name[num-1]
            return 'NONE'
        except Exception:
            return 'NONE'

    async def execute_delegation(self, message: str, task_id: str, team, orchestrator_agent_id):
        """Background worker that asks delegate and forwards task to selected agent by name."""
        try:
            choice = await self.delegate_task(message, task_id, team)
            if not choice or choice == 'NONE':
                return

            target_agent = None
            for name, agent in getattr(team, 'agents', {}).items():
                if name == choice or name.lower() == choice.lower() or getattr(agent, 'agent_id', None) == choice:
                    target_agent = agent
                    target_name = name
                    break

            if not target_agent:
                console.debug("Delegation: target not found", choice, task_id=task_id, agent_id=self.agent_id)
                return

            # If the chosen target is this same agent, ignore the delegation to avoid self-redirects
            try:
                if getattr(target_agent, 'agent_id', None) == orchestrator_agent_id:
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
                envelope = {
                    '_forwarded': True, 
                    'origin_agent': self.agent_id, 
                    'origin_task_id': task_id, 
                    'payload': message
                }
                if hasattr(target_agent, 'receive_message'):
                    target_agent.receive_message(envelope)
                elif hasattr(target_agent, 'run'):
                    target_agent.run(envelope)
                else:
                    console.debug("Delegation: target has no enqueue method", getattr(target_agent, 'agent_id', 'unknown'), task_id=task_id, agent_id=self.agent_id)
            except Exception as e:
                console.error("Delegation forward failed", str(e), task_id=task_id, agent_id=self.agent_id)

        except Exception:
            pass

    def is_task_redirected(self, task_id: str) -> bool:
        """Check if a task has been redirected"""
        return task_id in self._redirects

    def get_redirect_info(self, task_id: str) -> dict:
        """Get redirect information for a task"""
        return self._redirects.get(task_id, {})
