from typing import List, Dict, Optional, Union
from .agent import Agent
from .tools import Tool
from .utils.console import console
import uuid
import time

class AgentTeams:
    """
    AgentTeams class for managing multiple agents as a cohesive team.
    
    Features:
    - Team name and unique ID
    - Add/remove existing Agent objects to/from team
    - Team-level start/stop operations
    - Team monitoring and statistics
    
    Note: Agents must be created using Agent() before adding to the team.
    The team manages existing agent objects, it does not create them.
    To run tasks, get the agent from the team and call agent.run() directly.
    """
    
    def __init__(self, name: str, team_id: Optional[str] = None):
        self.name = name
        # Generate team ID if not provided
        self.team_id = team_id or f"team-{str(uuid.uuid4())[:6]}"
        self.agents: Dict[str, Agent] = {}
        self.running = False
        self.creation_time = time.time()
        
        console.system(f"Team created", f"Name: '{self.name}' | ID: {self.team_id}")
    
    def add_agent(self, agent_name: str, agent: Agent) -> None:
        """Add an existing agent to the team"""
        if not isinstance(agent, Agent):
            raise TypeError(f"Expected Agent object, got {type(agent)}")
        
        if agent_name in self.agents:
            console.warning(f"Agent '{agent_name}' already exists in team '{self.name}'", "Replacing existing agent")
        
        self.agents[agent_name] = agent
        # Attach back-reference so agent's orchestrator can see the team (used for delegation)
        try:
            if hasattr(agent, 'orchestrator'):
                agent.orchestrator.team = self
        except Exception:
            pass
        console.info(f"Agent added to team", f"Agent: '{agent_name}' (ID: {agent.agent_id}) | Team: '{self.name}'")
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent from the team"""
        if agent_name not in self.agents:
            console.warning(f"Agent '{agent_name}' not found in team '{self.name}'")
            return False
        
        # Stop the agent if it's running
        agent = self.agents[agent_name]
        if agent.running:
            agent.stop()
        
        del self.agents[agent_name]
        console.info(f"Agent removed from team", f"Agent: '{agent_name}' | Team: '{self.name}'")
        return True
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get a specific agent from the team"""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all agent names in the team"""
        return list(self.agents.keys())
    
    def start_all(self) -> None:
        """Start all agents in the team"""
        if self.running:
            console.warning(f"Team '{self.name}' is already running")
            return
        
        console.system(f"Starting team", f"Team: '{self.name}' | Agents: {len(self.agents)}")
        # Ensure each agent has a logger so token usage is recorded when running in a team
        for agent_name, agent in self.agents.items():
            self._ensure_agent_has_logger(agent)

            # Ensure orchestrator knows its team
            try:
                if hasattr(agent, 'orchestrator'):
                    agent.orchestrator.team = self
            except Exception:
                pass

            if not agent.running:
                agent.start()
                console.debug(f"Agent started", f"Agent: '{agent_name}' | Team: '{self.name}'")
        
        self.running = True
        console.system(f"Team started successfully", f"Team: '{self.name}' | Active agents: {len(self.agents)}")
    
    def stop_all(self) -> None:
        """Stop all agents in the team"""
        if not self.running:
            console.warning(f"Team '{self.name}' is not running")
            return
        
        console.system(f"Stopping team", f"Team: '{self.name}'")
        
        for agent_name, agent in self.agents.items():
            if agent.running:
                agent.stop()
                console.debug(f"Agent stopped", f"Agent: '{agent_name}' | Team: '{self.name}'")
        
        self.running = False
        console.system(f"Team stopped", f"Team: '{self.name}'")
    
    def get_team_info(self) -> Dict:
        """Get comprehensive information about the team"""
        agent_info = {}
        for agent_name, agent in self.agents.items():
            agent_info[agent_name] = {
                "agent_id": agent.agent_id,
                "running": agent.running,
                "tools": agent.get_available_tools(),
                "light_llm": agent.light_llm,
                "heavy_llm": agent.heavy_llm,
                "logging_enabled": agent.logger is not None
            }
        
        return {
            "team_name": self.name,
            "team_id": self.team_id,
            "running": self.running,
            "creation_time": self.creation_time,
            "agent_count": len(self.agents),
            "agents": agent_info
        }
    
    def get_team_stats(self) -> Dict:
        """Get statistics for the entire team"""
        stats = {
            "team_name": self.name,
            "team_id": self.team_id,
            "total_agents": len(self.agents),
            "running_agents": sum(1 for agent in self.agents.values() if agent.running),
            "uptime": time.time() - self.creation_time if self.running else 0,
            "agent_logs": {}
        }
        
        for agent_name, agent in self.agents.items():
            if agent.logger:
                stats["agent_logs"][agent_name] = agent.get_log_stats()
            else:
                stats["agent_logs"][agent_name] = {"logging": "disabled"}
        
        return stats
    
    def __str__(self) -> str:
            return f"AgentTeams(name='{self.name}', id='{self.team_id}', agents={len(self.agents)}, running={self.running})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def _ensure_agent_has_logger(self, agent: Agent) -> None:
        """Attach a default HLRLogger to the agent and its orchestrator if missing."""
        from .utils.logger import HLRLogger

        if getattr(agent, 'logger', None) is None:
            agent.logger = HLRLogger()
            try:
                agent.orchestrator.set_logger(agent.logger)
            except Exception:
                pass
