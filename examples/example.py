

import time
from tarq_agent import Agent


agent = Agent(
    tools=['jira', 'gmail', 'sheets', 'drive', 'calendar', 'slack'],
    light_llm="gemini-2.5-flash-lite",
    heavy_llm="gemini-2.5-flash-lite",
    enable_logging=True
)

agent.start()

# ================== TEST EXAMPLES - INCREASING DIFFICULTY ==================

# LEVEL 1 (Easy): Simple single tool usage - Duration: 2.73s (compute: 2.73s) Tokens: 307 (input: 272, output: 35) | LLM Calls: 2
# user_input = "What is the last mail"

# LEVEL 2 (Basic): Simple conditional logic with two tools -  Duration: 4.23s (compute: 4.23s) Tokens: 454 (input: 388, output: 66) | LLM Calls: 3
# user_input = "Check the mail, if you get an email from 'support@google.com' upload that report to drive, otherwise send a message to #general in slack"

# LEVEL 3 (Intermediate): Multi-tool workflow with validation - Duration: 2.73s (compute: 2.73s) Tokens: 385 (input: 346, output: 39) | LLM Calls: 2
# user_input = "Check gmail for monthly reports, create a summary spreadsheet with the email details, save it to Drive, and notify the team via Slack"

# LEVEL 4 (Advanced): Periodic monitoring with conditional execution -   Duration: 4.42s (compute: 4.42s) Tokens: 484 (input: 412, output: 72) | LLM Calls: 3
# user_input = "Watch the gmail each hour until you receive a report in an email subject. When that happens you have to upload the details into the spreadsheet called '2025' and finish the task"

# LEVEL 5 (Expert): Complex multi-conditional with feedback loops and multiple integrations - COMPLETED: DURATION AND TOKENS CANNOT BE TESTED
#user_input = "Check gmail periodically every 15 minutes for monthly reports. If sender is from 'admin@google.com', immediately create a Jira ticket and schedule emergency meeting. If from 'support@google.com', update spreadsheet and send success notification to Slack."

user_input = "Watch the gmail each hour until you receive a report in an email subject. When that happens you have to upload the details into the spreadsheet called '2025' and finish the task"
agent.run(user_input)
        
time.sleep(60) 
agent.stop()
