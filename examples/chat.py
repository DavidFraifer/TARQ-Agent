"""
Interactive chat example with HLR Agent.
Type messages and see the agent process them in real-time.
"""

import asyncio
import threading
from hlr_agent import Agent

def input_listener(agent):
    """Handle user input in a separate thread."""
    print("\n💬 Chat started! Type your messages ('quit' to exit):")
    print("=" * 50)
    
    while agent.running:
        try:
            user_input = input("📝 Your message: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Exiting chat...")
                break
                
            if user_input:
                agent.run(user_input)
                print(f"✉️  Message sent: {user_input[:50]}...")
                
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Chat interrupted...")
            break
    
    if agent.running:
        agent.stop()

async def main():
    print("🤖 Starting HLR Agent Chat")
    print("=" * 40)
    
    # Create agent with chat-friendly tools
    agent = Agent(
        tools=["gmail", "jira", "sheets"], 
        light_llm="gemini-2.0-flash",
        heavy_llm="gemini-2.0-flash"
    )
    
    agent.start()
    print(f"✅ Agent started with tools: {', '.join(agent.get_available_tools())}")
    
    # Start input listener thread
    input_thread = threading.Thread(target=input_listener, args=(agent,), daemon=True)
    input_thread.start()
    
    try:
        # Keep main thread alive while chat is active
        while agent.running and input_thread.is_alive():
            await asyncio.sleep(0.25)
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted...")
    finally:
        if agent.running:
            agent.stop()
        if input_thread.is_alive():
            input_thread.join(timeout=2)
        print("✅ Program terminated correctly")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


