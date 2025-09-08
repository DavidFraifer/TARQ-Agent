

from tarq_agent.agent import Agent, Tool
from tarq_agent.utils.console import console


def my_tool(user_input="", task_id=None, task_memory=None, agent_id=None):
    """A simple custom tool"""
    console.tool(f"[MY_TOOL] Processing: {user_input}", task_id=task_id, agent_id=agent_id)
    
    result = f"Processed: {user_input.upper()}"
    
    if task_memory:
        task_memory.set(result)
    
    return result


def main():
    # Create custom tool
    custom_tool = Tool(name="my_tool", func=my_tool, description="Processes text")
    
    # Create agent with custom tool
    agent = Agent(
        agent_id="test-agent",
        tools=["websearch", custom_tool],
        light_llm="gemini-2.5-flash-lite",
        heavy_llm="gemini-2.5-flash-lite"
    )
    
    # Use the agent
    agent.start()
    agent.run("Use my tool to process: hello world")
    
    # Let it process for a moment
    import time
    time.sleep(3)
    
    agent.stop()


if __name__ == "__main__":
    main()
