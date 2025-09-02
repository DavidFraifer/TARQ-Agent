import random
import time
import asyncio

from ..utils.console import console
from .websearch import web_search


def gmail_tool(user_input: str = "", task_id: str = None, task_memory=None):
    emails = ["support@google.com", "admin@google.com"]
    current_email = random.choice(emails)
    message = f"Email checked - Found message from: {current_email} with subject: 'Monthly Report Available'"
    if task_memory:
        try:
            task_memory.set(message)
        except Exception:
            pass
    try:
        console.tool(f"[GMAIL] {message}", task_id=task_id)
    except Exception:
        print(f"[Gmail] {message}")


def sheets_tool(user_input: str = "", task_id: str = None, task_memory=None):
    message = f"Spreadsheet updated - Added new data row with timestamp {time.strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        console.tool(f"[SHEETS] {message}", task_id=task_id)
    except Exception:
        print(f"[Sheets] {message}")
    if task_memory:
        try:
            task_memory.set(f"Tool sheets: {message}")
        except Exception:
            pass


def drive_tool(user_input: str = "", task_id: str = None, task_memory=None):
    message = f"File uploaded to Google Drive - Document saved to /Reports/ folder"
    try:
        console.tool(f"[DRIVE] {message}", task_id=task_id)
    except Exception:
        print(f"[Drive] {message}")
    if task_memory:
        try:
            task_memory.set(f"Tool drive: {message}")
        except Exception:
            pass


def jira_tool(user_input: str = "", task_id: str = None, task_memory=None):
    ticket_id = f"TARQ-{random.randint(1000, 9999)}"
    message = f"Jira ticket created - {ticket_id}: Task tracking ticket generated"
    try:
        console.tool(f"[JIRA] {message}", task_id=task_id)
    except Exception:
        print(f"[Jira] {message}")
    if task_memory:
        try:
            task_memory.set(f"Tool jira: {message}")
        except Exception:
            pass


def calendar_tool(user_input: str = "", task_id: str = None, task_memory=None):
    event_time = time.strftime('%Y-%m-%d %H:%M:%S')
    message = f"Calendar event created - Meeting scheduled for {event_time}"
    try:
        console.tool(f"[CALENDAR] {message}", task_id=task_id)
    except Exception:
        print(f"[Calendar] {message}")
    if task_memory:
        try:
            task_memory.set(f"Tool calendar: {message}")
        except Exception:
            pass


def slack_tool(user_input: str = "", task_id: str = None, task_memory=None):
    message = f"Slack message sent - Notification delivered to #general channel"
    try:
        console.tool(f"[SLACK] {message}", task_id=task_id)
    except Exception:
        print(f"[Slack] {message}")
    if task_memory:
        try:
            task_memory.set(f"Tool slack: {message}")
        except Exception:
            pass


async def websearch_tool(user_input: str = "", task_id: str = None, task_memory=None, light_llm=None):
    """
    Web search tool that performs intelligent web search with LLM-powered query extraction and summarization.
    
    Args:
        user_input: The user's search query or request
        task_id: Task identifier for logging
        task_memory: Task memory object for storing results
        light_llm: The light LLM model to use for processing
    
    Returns:
        Search results summary
    """
    try:
        result, token_info = await web_search(
            task_memory=task_memory or [],
            text=user_input,
            task_id=task_id or "websearch",
            fast_search=True,  
            light_llm=light_llm
        )
        
        # Store result in task memory if available
        if task_memory:
            try:
                task_memory.set(f"Web search result: {result}")
            except Exception:
                pass

        # Store token info in a way the orchestrator can access it
        # We'll attach it to the result string in a way that can be parsed
        setattr(websearch_tool, '_last_token_info', token_info)
        
        return result
        
    except Exception as e:
        error_msg = f"Web search failed: {str(e)}"
        try:
            console.error("WEBSEARCH", error_msg, task_id=task_id)
        except Exception:
            print(f"[WebSearch Error] {error_msg}")
        
        if task_memory:
            try:
                task_memory.set(f"Web search error: {error_msg}")
            except Exception:
                pass
        
        return f"Sorry, web search encountered an error: {str(e)}"



# Internal tools registry
internal_tools = {
    "jira": jira_tool,
    "gmail": gmail_tool,
    "sheets": sheets_tool,
    "drive": drive_tool,
    "calendar": calendar_tool,
    "slack": slack_tool,
    "websearch": websearch_tool,
}
