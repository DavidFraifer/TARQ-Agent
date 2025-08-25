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

# LEVEL 1 (Easy): Simple single tool usage - Duration: 2.52s (compute: 2.52s) Tokens: 283 (input: 248, output: 35) | LLM Calls: 2
# user_input = "What is the last mail"

# LEVEL 2 (Basic): Simple conditional logic with two tools -  Duration: 3.77s (compute: 3.77s) Tokens: 426 (input: 366, output: 60) | LLM Calls: 3
# user_input = "Check the mail, if you get an email from 'support@google.com' upload that report to drive, otherwise send a message to #general in slack"

# LEVEL 3 (Intermediate): Multi-tool workflow with validation - Duration: 2.46s (compute: 2.46s) Tokens: 361 (input: 322, output: 39) | LLM Calls: 2
# user_input = "Check gmail for monthly reports, create a summary spreadsheet with the email details, save it to Drive, and notify the team via Slack"

# LEVEL 4 (Advanced): Periodic monitoring with conditional execution -  Duration: 3.77s (compute: 3.77s) Tokens: 478 (input: 388, output: 90) | LLM Calls: 3
# user_input = "Watch the gmail each hour until you receive a report in an email subject. When that happens you have to upload the details into the spreadsheet called '2025' and finish the task"

# LEVEL 5 (Expert): Complex multi-conditional with feedback loops and multiple integrations - COMPLETED: DURATION AND TOKENS CANNOT BE TESTED
#user_input = "Check gmail periodically every 15 minutes for monthly reports. If sender is from 'admin@google.com', immediately create a Jira ticket and schedule emergency meeting. If from 'support@google.com', update spreadsheet and send success notification to Slack."

user_input = "Check gmail periodically every 15 minutes for monthly reports. If sender is from 'admin@google.com', immediately create a Jira ticket and schedule emergency meeting. If from 'support@google.com', update spreadsheet and send success notification to Slack."
agent.run(user_input)
        
time.sleep(60) 
agent.stop()
