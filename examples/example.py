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

user_input = "Enter to gmail, grab the report from the last email and upload that data to a new sheets"
#Enter to gmail, grab the report from the last email and upload that data to a new sheets
#"Watch the gmail each hour until you receive a report in an email. When that happens you have to upload the details into the spreadsheet called '2025' and finish the task."
agent.run(user_input)
        
time.sleep(5) 
agent.stop()
