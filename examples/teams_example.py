

from hlr_agent import Agent, AgentTeams
import time
# Create teams (one custom ID, one auto-generated)
auto_team = AgentTeams(name="Auto Team")

# Create agents (one custom ID, one auto-generated)
marketing_agent = Agent(tools=['jira', 'sheets'], agent_id="marketing-agent-001")
sales_agent = Agent(tools=['gmail', 'sheets'], agent_id="sales-agent-001")

# Add agents to teams
auto_team.add_agent("marketing",marketing_agent)
auto_team.add_agent("sales", sales_agent)


# Start all agents in both teams
auto_team.start_all()

marketing_agent.run("Go to gmail and make a report in sheets about the last email")

time.sleep(60) 
# Stop all agents in both teams
auto_team.stop_all()

