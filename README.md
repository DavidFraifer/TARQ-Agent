# HLR (Hierarchical LLM Routing)

[![PyPI version](https://badge.fury.io/py/hlr-agent.svg)](https://badge.fury.io/py/hlr-agent)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**HLR** is a flexible Python library for creating and managing hierarchical workflows driven by nodes and powered by Large Language Models (LLMs).

Each node represents a distinct step or action in your process. It can execute custom Python functions, update a shared context, and intelligently decide the next node to transition to, either programmatically or by leveraging an LLM's understanding. This makes HLR ideal for applications needing sequential task processing with dynamic, context-aware routing.

## Features

-   **Node-Based Workflows:** Structure complex processes into manageable, reusable nodes.
-   **Hierarchical Flow:** Design workflows where nodes can branch and converge based on logic or LLM decisions.
-   **Non-Linear Dependencies:** Support for circular dependencies between nodes for complex workflows.
-   **Dynamic Routing:**
    -   Nodes can explicitly return the ID of the next node.
    -   Alternatively, let an integrated LLM choose the next node based on descriptions and the current context.
-   **Shared Context:** Maintain state across nodes using a simple dictionary (`agent.context`). Nodes can read and write data (like logs, intermediate results, or the execution path).
    -   `agent.context["info"]`: Commonly used for accumulating logs or data.
    -   `agent.context["route"]`: Automatically tracks the sequence of nodes visited.
-   **LLM Integration:** Seamlessly uses Google's Gemini models (`gemini-2.0-flash`, `gemini-2.5-flash-lite`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`) and OpenAI (`gpt-4o`) for routing decisions.
-   **Flexible Message Handling:** Pass user messages at runtime through the `run()` method for dynamic execution.
-   **Robust Validation:** Ensures required parameters are provided during initialization and runtime, preventing common errors.

## Installation

Install HLR directly from PyPI:

```bash
pip install hlr-agent
```

*(Note: PyPI normalizes names, so `pip install hlr_agent` also works).*

## Configuration

HLR automatically loads API keys from environment variables. Create a `.env` file in your project root or execution directory:

```bash
# .env file
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
BRAVE_API_KEY=your_brave_api_key_here
```

### Supported API Key Formats

The system automatically detects API keys in multiple formats:
- **Google/Gemini**: `GOOGLE_API_KEY`, `GOOGLE_AI_API_KEY`, `GEMINI_API_KEY`
- **OpenAI**: `OPENAI_API_KEY`
- **Brave Search**: `BRAVE_API_KEY`

Copy the provided `.env.example` file and add your actual API keys. No need to pass API keys manually in your code!

## Usage Example

Here's how to define nodes, functions, and run the agent:

### 1. Define Node Functions

Each function associated with a node receives the `agent` instance, allowing it to access and modify `agent.context`.

```python
# example.py
from hlr_agent import Node, Agent
import time

# Example context keys used: "info", "route", "emails_to_send"



def func_database(agent):
    print("Querying database...")
    found_emails = [# Simulate database query result
        "user1@example.com",
        "test.user@sample.net",
        "another@domain.org"
    ]
    print(f"Database: Found {len(found_emails)} emails.")
    agent.context["info"] += "\nQuery result:\n"+ str(found_emails)


def func_mailing(agent): 
    print("Emails has been formatted.")
    redacted_emails = [ # Simulate redaction of emails
        ["Subject1", "Message1", "user1@example.com"],
        ["Subject2", "Message2", "test.user@sample.net"],
        ["Subject3", "Message3", "another@domain.org"],
    ]
    agent.context["info"] += "\nRedacted emails:\n" + str(redacted_emails)
    print("Emails has been sent.")


def func_output(agent):
    if "info" in agent.context: 
        print("LOGS:\n" + agent.context["info"])
    if "route" in agent.context:
        print(agent.context["route"]) 

```

### 2. Configure Nodes

Define the structure of your workflow using `Node` objects. Nodes can have circular dependencies for complex workflows:

```python
# example.py (continued)

nodes = [
    Node("Input", children=["Database", "Mailing", "None"]),
    Node("Database", children=["Mailing","Output"], func=func_database, description="Select this if the user wants to use a database"),
    Node("Mailing", children=["Database","Output"], func=func_mailing, description="Select this if the user wants to use a mailing related functionality."),
    Node("None", children=["Output"], description="Select this node if the rest o the nodes are not valid for the request"),
    Node("Output", children=None, func=func_output),
]
```

### 3. Initialize and Run the Agent

Create an `Agent` instance and execute the workflow using `agent.run()`. The user message is now passed directly to the `run()` method:

```python
# example.py (continued)

# --- Agent Initialization ---
graph = Graph(
    nodes=nodes,
    start_node_id="Input",
    end_node_id="Output",
    model="gemini-1.5-flash-8b"  # API key automatically loaded from .env
)

# --- Run the Agent ---
user_request = "Please get the emails from the database and send them a welcome message."

print("--- Starting Graph Run ---")
try:
    graph.run(user_message=user_request)  # API key automatically loaded
except ValueError as e:
    print(f"Graph Error: {e}")
print("\n--- Graph Run Finished ---")

```

## How It Works

1.  **Initialization (`Agent(...)`):**
    *   Validates all required parameters (`nodes`, `start_node_id`, `end_node_id`, `model`, `api_key`).
    *   Checks for duplicate `node_id`s.
    *   Builds an internal dictionary of nodes for quick access.
    *   Initializes `agent.context` (e.g., setting the initial `route`).

2.  **Execution (`agent.run(user_message=...)`):**
    *   Takes the `user_message` for the current run.
    *   Starts at the `start_node_id`.
    *   Enters a loop that continues until a terminal node is reached or the step limit is exceeded.
    *   **In each step:**
        *   Executes the `func` of the current node (if defined). This function can modify `agent.context`.
        *   Determines the `next_node_id`:
            *   **Explicit:** If `func` returns a valid node ID, use that.
            *   **Implicit/LLM:** If `func` returns `None`:
                *   If the current node has no children, the flow ends.
                *   If it has one valid child (with description or is `end_node_id`), transition to it.
                *   If multiple valid children exist, call the LLM (`get_next_node`).
                    *   The LLM receives: the list of valid child IDs and their descriptions, the `user_message` for the run, and `extra_context` (a string combining `agent.context["route"]` and `agent.context["info"]`).
                    *   The LLM returns the chosen child ID.
        *   Updates `agent.context["route"]` by appending the `next_node_id`.
        *   Sets the `current_id` to the `next_node_id` for the next iteration.

3.  **Termination:** The loop ends when `current_id` becomes `None` (either explicitly set by a node, reaching a node with no children, or encountering an error like an invalid explicit return). The `func_output` (or the function of the last node) typically handles final reporting.

## What's New in v0.3.0

-   **Runtime Message Passing:** User messages are now passed to `agent.run(user_message="...")` instead of during agent initialization, allowing for more flexible and dynamic execution.
-   **Additional Gemini Models:** Added support for `gemini-1.5-flash` and `gemini-1.5-flash-8b` alongside the existing `gemini-2.0-flash` and `gpt-4o`.
-   **Enhanced Non-Linear Workflows:** Improved support for circular dependencies between nodes, enabling more complex workflow patterns where nodes can reference each other in cycles.
-   **Better Circular Dependency Handling:** The routing system now better handles cases where nodes have bidirectional relationships for sophisticated task management.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/DavidFraifer/HLR/issues) or submit a pull request.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

## Contact

David Serrano Díaz – davidsd.2704@gmail.com

Project Link: [https://github.com/DavidFraifer/HLR](https://github.com/DavidFraifer/HLR)