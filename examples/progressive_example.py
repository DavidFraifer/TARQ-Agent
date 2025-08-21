"""
Progressive Tool Usage Example for HLR Agent
This example demonstrates 4 different tasks with increasing complexity:
1. Single tool (Gmail)
2. Two tools (Gmail + Sheets)
3. Three tools (Gmail + Sheets + Drive)
4. Four tools (Gmail + Sheets + Drive + Slack)
"""

import time
from hlr_agent import Agent

def main():
    # Initialize agent with all available tools
    agent = Agent(
        tools=['jira', 'gmail', 'sheets', 'drive', 'calendar', 'slack'],
        light_llm="gemini-2.5-flash",
        heavy_llm="gemini-2.5-flash",
        enable_logging=True
    )
    
    agent.start()
    
    # Task 1: Single tool (Gmail only)
    print("\n" + "="*60)
    print("TASK 1: Single Tool - Gmail Check")
    print("="*60)
    task1 = "Check my gmail inbox for any new messages"
    print(f"Input: {task1}")
    agent.run(task1)
    time.sleep(5)  # Wait for task to complete before next one
    
    # Task 2: Two tools (Gmail + Sheets)
    print("\n" + "="*60)
    print("TASK 2: Two Tools - Gmail + Sheets")
    print("="*60)
    task2 = "Check my gmail for any reports and create a summary spreadsheet with the email details"
    print(f"Input: {task2}")
    agent.run(task2)
    time.sleep(5)  # Wait for task to complete before next one
    
    # Task 3: Three tools (Gmail + Sheets + Drive)
    print("\n" + "="*60)
    print("TASK 3: Three Tools - Gmail + Sheets + Drive")
    print("="*60)
    task3 = "Check gmail for project updates, compile them into a spreadsheet, and save the file to Google Drive"
    print(f"Input: {task3}")
    agent.run(task3)
    time.sleep(5)  # Wait for task to complete before next one
    
    # Task 4: Four tools (Gmail + Sheets + Drive + Slack)
    print("\n" + "="*60)
    print("TASK 4: Four Tools - Gmail + Sheets + Drive + Slack")
    print("="*60)
    task4 = "Check gmail for daily reports, create a data analysis spreadsheet, save it to Drive, and send a summary message to the team via Slack"
    print(f"Input: {task4}")
    agent.run(task4)
    time.sleep(5)  # Wait for final task to complete
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED")
    print("="*60)
    
    # Show final statistics
    if agent.logger:
        stats = agent.get_log_stats()
        print(f"\nFinal Statistics:")
        print(f"Total Tasks: {stats.get('total_tasks', 0)}")
        print(f"Total Duration: {stats.get('total_duration', 0):.2f}s")
        print(f"Total Tokens: {stats.get('total_tokens', 0)}")
        print(f"Total LLM Calls: {stats.get('total_llm_calls', 0)}")
        print(f"Average Duration per Task: {stats.get('avg_duration', 0):.2f}s")
    
    agent.stop()

if __name__ == "__main__":
    main()
