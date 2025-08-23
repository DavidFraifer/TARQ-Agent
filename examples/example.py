"""
Simple interactive chat example with HLR Agent.
Type messages and see the agent process them in real-time.
"""

import time
from hlr_agent import Agent


agent = Agent(
    tools=['jira', 'gmail', 'sheets', 'drive', 'calendar', 'slack'],
    light_llm="gemini-2.5-flash-lite",
    heavy_llm="gemini-2.5-flash-lite",
    enable_logging=True
)

agent.start()

# ================== TEST EXAMPLES - INCREASING DIFFICULTY ==================

# LEVEL 1 (Easy): Simple single tool usage - Duration: 2.72s Tokens: 427 (input: 389, output: 38) | LLM Calls: 2

# LEVEL 2 (Basic): Simple conditional logic with two tools -  Duration: 4.20s Tokens: 584 (input: 507, output: 77) | LLM Calls: 3
# user_input = "Check the mail, if you get an email from 'support@google.com' upload that report to drive, otherwise send a message to #general in slack"

# LEVEL 3 (Intermediate): Multi-tool workflow with validation - Duration: 4.20s Tokens: 425 (input: 368, output: 57) | LLM Calls: 3
# user_input = "Check gmail for monthly reports, create a summary spreadsheet with the email details, save it to Drive, and notify the team via Slack"

# LEVEL 4 (Advanced): Periodic monitoring with conditional execution - Duration: 3.98s Tokens: 438 (input: 378, output: 60) | LLM Calls: 3
# user_input = "Watch the gmail each hour until you receive a report in an email. When that happens you have to upload the details into the spreadsheet called '2025' and finish the task"

# LEVEL 5 (Expert): Complex multi-conditional with feedback loops and multiple integrations - COMPLETED: DURATION AND TOKENS CANNOT BE TESTED
#user_input = "Check gmail periodically every 15 minutes for monthly reports. If sender is from 'admin@google.com', immediately create a Jira ticket and schedule emergency meeting. If from 'support@google.com', update spreadsheet and send success notification to Slack."

user_input = "Check gmail periodically every 15 minutes for monthly reports. If sender is from 'admin@google.com', immediately create a Jira ticket and schedule emergency meeting. If from 'support@google.com', update spreadsheet and send success notification to Slack."
agent.run(user_input)
        
time.sleep(60) 
agent.stop()
