import random
import time

def _create_tool_func(name: str, action: str):
    """Create a tool function for the simplified architecture"""
    def tool_func(context: str = "", task_id: str = None):
        """Tool function that takes context string and optional task_id"""
        # Import console here to avoid circular imports
        try:
            from ..utils.console import console
        except ImportError:
            # Fallback if console is not available
            console = None
        
        if name == "gmail":
            emails = ["support@google.com", "admin@google.com"]
            current_email = random.choice(emails)
            message = f"Email checked - Found message from: {current_email} with subject: 'Monthly Report Available'"
            if console:
                console.tool(f"[GMAIL] {message}", task_id=task_id)
            else:
                print(f"[Gmail] {message}")
        elif name == "sheets":
            message = f"Spreadsheet updated - Added new data row with timestamp {time.strftime('%Y-%m-%d %H:%M:%S')}"
            if console:
                console.tool(f"[SHEETS] {message}", task_id=task_id)
            else:
                print(f"[Sheets] {message}")
        elif name == "drive":
            message = f"File uploaded to Google Drive - Document saved to /Reports/ folder"
            if console:
                console.tool(f"[DRIVE] {message}", task_id=task_id)
            else:
                print(f"[Drive] {message}")
        elif name == "jira":
            ticket_id = f"HLR-{random.randint(1000, 9999)}"
            message = f"Jira ticket created - {ticket_id}: Task tracking ticket generated"
            if console:
                console.tool(f"[JIRA] {message}", task_id=task_id)
            else:
                print(f"[Jira] {message}")
        elif name == "calendar":
            event_time = time.strftime('%Y-%m-%d %H:%M:%S')
            message = f"Calendar event created - Meeting scheduled for {event_time}"
            if console:
                console.tool(f"[CALENDAR] {message}", task_id=task_id)
            else:
                print(f"[Calendar] {message}")
        elif name == "slack":
            message = f"Slack message sent - Notification delivered to #general channel"
            if console:
                console.tool(f"[SLACK] {message}", task_id=task_id)
            else:
                print(f"[Slack] {message}")
        else:
            message = f"{action} completed successfully"
            if console:
                console.tool(f"[{name.upper()}] {message}", task_id=task_id)
            else:
                print(f"[{name}] {message}")
        
        return message
    
    return tool_func

# Internal tools registry
internal_tools = {
    "jira": _create_tool_func("jira", "Creating Jira ticket"),
    "gmail": _create_tool_func("gmail", "Processing email"),
    "sheets": _create_tool_func("sheets", "Updating Google Sheets"),
    "drive": _create_tool_func("drive", "Uploading to Google Drive"),
    "calendar": _create_tool_func("calendar", "Creating calendar event"),
    "slack": _create_tool_func("slack", "Sending Slack message")
}
