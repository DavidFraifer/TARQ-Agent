
from tarq_agent import Agent

def main():
    # Create agent with basic tools
    agent = Agent(
        tools=['gmail', 'sheets', 'slack', 'drive', 'websearch','jira'],
        light_llm="gemini-2.5-flash-lite",
        heavy_llm="gemini-2.5-flash-lite"
    )

    agent.start()
    print("Agent started! You can now send multiple tasks concurrently.")
    print("The agent processes tasks asynchronously in the background.")
    print("Type 'quit' to exit\n")
    

    try:
        while True:
            # Get user input
            user_input = input(f"\n[Enter task (or 'quit'): ")
            
            if user_input.lower() == 'quit':
                print("Shutting down...")
                agent.stop()
                break
            
            if user_input.strip():
                agent.run(user_input)

    except KeyboardInterrupt:
        agent.stop()
        print("\n🛑 Interrupted by user")
    

if __name__ == "__main__":
    main()
