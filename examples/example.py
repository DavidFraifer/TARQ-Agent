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

# LEVEL 1 (Easy): Simple single tool usage - Duration: 2.91s Tokens: 268 (input: 198, output: 70) | LLM Calls: 2
# user_input = "Check my gmail inbox for new messages"

# LEVEL 2 (Basic): Simple conditional logic with two tools - Duration: 4.46s Tokens: 423 (input: 331, output: 92) | LLM Calls: 3
# user_input = "Check the mail, if you get an email from 'support@google.com' upload that report to drive, otherwise send a message to #general in slack"

# LEVEL 3 (Intermediate): Multi-tool workflow with validation - Duration: 2.64s Tokens: 364 (input: 287, output: 77) | LLM Calls: 2
# user_input = "Check gmail for project reports, create a summary spreadsheet with the email details, save it to Drive, and notify the team via Slack"

# LEVEL 4 (Advanced): Periodic monitoring with conditional execution - 3.98s Tokens: 443 (input: 353, output: 90) | LLM Calls: 3
# user_input = "Watch the gmail each hour until you receive a report in an email. When that happens you have to upload the details into the spreadsheet called '2025' and finish the task"

# LEVEL 5 (Expert): Complex multi-conditional with feedback loops and multiple integrations - X DOES NOT FINISHED
# user_input = "Monitor gmail every 30 minutes for emails from either 'admin@google.com' or 'support@google.com'. If from admin, create a Jira ticket and schedule a calendar meeting. If from support, update the monthly spreadsheet in Drive and send detailed analysis to Slack. Continue until both types of emails are received and processed"

# Current active test:
user_input = "Monitor gmail every 30 minutes for emails from either 'admin@google.com' or 'support@google.com'. If from admin, create a Jira ticket and schedule a calendar meeting. If from support, update the monthly spreadsheet in Drive and send detailed analysis to Slack. Continue until both types of emails are received and processed"
agent.run(user_input)
        
time.sleep(60) 
agent.stop()
