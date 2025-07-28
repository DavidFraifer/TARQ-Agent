import asyncio
import random
from .tool import Tool

def _create_tool_func(name: str, emoji: str, action: str, delay: float = 1.0):
    """Factory function to create consistent tool functions."""
    def tool_func(graph):
        task_id = graph._memory_ref.name if graph._memory_ref else "unknown"
        
        # Special logic for Gmail to simulate different emails
        if name == "gmail":
            emails = ["support@company.com", "user@example.com", "admin@wikipedia.org", "john.doe@example.com"]
            current_email = random.choice(emails)
            message = f"Processed email from: {current_email}"
            print(message)
        else:
            message = f"{action} completed"
        
        # Use asyncio.sleep for non-blocking delay
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, use async sleep
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    import time
                    executor.submit(time.sleep, delay).result()
            else:
                import time
                time.sleep(delay)
        except:
            # Fallback to sync sleep if async is not available
            import time
            time.sleep(delay)
        
        if graph._memory_ref:
            graph._memory_ref.set(message)
    
    return tool_func

# Internal tools registry
INTERNAL_TOOLS = {
    "jira": Tool(
        name="jira",
        func=_create_tool_func("jira", "📋", "Creating Jira ticket", 1.5),
        description="Create and manage Jira tickets for task tracking"
    ),
    "gmail": Tool(
        name="gmail", 
        func=_create_tool_func("gmail", "📧", "Processing email", 1.0),
        description="Read, process, and send emails via Gmail"
    ),
    "sheets": Tool(
        name="sheets",
        func=_create_tool_func("sheets", "📊", "Updating Google Sheets", 2.0),
        description="Read and update Google Sheets spreadsheets"
    ),
    "drive": Tool(
        name="drive",
        func=_create_tool_func("drive", "💾", "Uploading to Google Drive", 1.2),
        description="Upload, download, and manage files in Google Drive"
    ),
    "calendar": Tool(
        name="calendar",
        func=_create_tool_func("calendar", "📅", "Creating calendar event", 0.8),
        description="Create and manage events in Google Calendar"
    ),
    "slack": Tool(
        name="slack",
        func=_create_tool_func("slack", "💬", "Sending Slack message", 0.5),
        description="Send messages and notifications via Slack"
    )
}

def get_internal_tool(tool_name: str) -> Tool:
    """Get an internal tool by name."""
    if tool_name not in INTERNAL_TOOLS:
        available = list(INTERNAL_TOOLS.keys())
        raise ValueError(f"Internal tool '{tool_name}' not found. Available tools: {available}")
    return INTERNAL_TOOLS[tool_name]
