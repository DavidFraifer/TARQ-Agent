

from tarq_agent import Agent, AgentTeams
import time
auto_team = AgentTeams(name="Auto Team")


marketing_agent = Agent(tools=['jira', 'sheets'], light_llm="gemini-2.5-flash-lite", heavy_llm="gemini-2.5-flash-lite")
sales_agent = Agent(tools=['gmail', 'sheets'], light_llm="gemini-2.5-flash-lite", heavy_llm="gemini-2.5-flash-lite")

auto_team.add_agent("marketing",marketing_agent)
auto_team.add_agent("sales", sales_agent)

# Start all agents in both teams
auto_team.start_all()

marketing_agent.run("Go to gmail and make a report in sheets about the last email")

time.sleep(60) 
# Stop all agents in both teams
auto_team.stop_all()

