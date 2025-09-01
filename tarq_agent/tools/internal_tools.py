import random
import time

from ..utils.console import console
from .websearch import *


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


def websearch_tool(user_input: str = "", task_id: str = None, task_memory=None):
    try:
        search_web(task_memory, user_input, task_id)
    except Exception:
        pass



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
