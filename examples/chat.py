"""
Simple interactive chat example with HLR Agent.
Type messages and see the agent process them in real-time.
"""

import time
from hlr_agent import Agent

def main():
    print("🤖 Starting HLR Agent Chat")
    print("=" * 40)
    
    # Create agent with chat-friendly tools
    agent = Agent(
        tools=["gmail", "jira", "sheets"], 
        light_llm="gemini-2.5-flash-lite",
        heavy_llm="gemini-2.5-flash-lite",
        enable_logging=True
    )
    
    agent.start()
    print(f"✅ Agent started with tools: {', '.join(agent.get_available_tools())}")
    print("\n💬 Chat started! Type your messages ('quit' to exit):")
    print("=" * 50)
    
    try:
        while True:
            # Get user input
            user_input = input("\n📝 Your message: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting chat...")
                break
            
            # Skip empty messages
            if not user_input:
                continue
            
            # Send message to agent
            print(f"✉️  Processing: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
            print("-" * 50)
            
            agent.run(user_input)
            
            # Small delay to let processing complete
            time.sleep(1)
            print("-" * 50)
                
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Chat interrupted...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if agent.running:
            agent.stop()
        print("✅ Chat session ended")

if __name__ == "__main__":
    main()
