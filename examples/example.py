
"""
Basic HLR Agent example showing email processing workflow.
"""

from hlr_agent import Agent
import time

# Create and configure agent
agent = Agent(
    tools=["gmail", "jira", "sheets"], 
    light_llm="gemini-2.5-flash-lite",    
    heavy_llm="gemini-2.5-flash-lite",
    enable_logging=True
    
)

# Start agent, submit task, wait, and stop
agent.start()
agent.run("Check my gmail inbox and look for an email from john.doe@example.com then create a Jira ticket with the email subject.")

time.sleep(15)

agent.stop()
print("✅ Example completed!")