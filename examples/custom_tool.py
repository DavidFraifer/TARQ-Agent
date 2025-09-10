

from tarq_agent.agent import Agent, Tool
from tarq_agent.utils.console import console

import time


def memory_and_token_tool(user_input="", task_id=None, task_memory=None, agent_id=None):
    """Demonstrates both memory access and token tracking"""
    console.tool(f"[DEMO] Processing: {user_input}", task_id=task_id, agent_id=agent_id)
    
    # READ from task memory
    current_memory = task_memory.get() if task_memory else "Empty"
    memory_count = len(current_memory.splitlines()) if current_memory != "Empty" else 0
    
    # Process input
    result = f"Found {memory_count} memory lines. Processing: {user_input.upper()}"
    
    # WRITE to task memory
    if task_memory:
        task_memory.set(f"Processed: {user_input}")
    
    # TRACK tokens (gets added to agent counter)
    setattr(memory_and_token_tool, '_last_token_info', {
        "tokens_used": 1000,
        "input_tokens": 700, 
        "output_tokens": 300,
        "llm_calls": 1
    })
    
    return result


def main():
    # Create custom tool
    demo_tool = Tool(name="demo_tool", func=memory_and_token_tool, 
                    description="Shows memory access and token tracking")
    
    # Create agent
    agent = Agent(
        tools=[demo_tool],
        light_llm="gemini-2.5-flash-lite",
        heavy_llm="gemini-2.5-flash-lite"
    )
    
    # Test the functionality
    agent.start()
    agent.run("Use demo_tool to test memory: hello world")
    
    time.sleep(10)

    agent.stop()


if __name__ == "__main__":
    main()
