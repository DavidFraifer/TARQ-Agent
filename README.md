# HLR (Hierarchical LLM Routing)

**HLR** is a flexible and easy-to-use library for managing hierarchical workflows based on nodes.  
Each node represents a step in your workflow, executes a function, and decides which node to execute next.  
This makes it ideal for applications requiring sequential task processing with dynamic routing and context sharing.

## Features

- **Hierarchical Flow:** Organize your workflow into nodes that can branch out or converge based on custom logic.
- **Dynamic Routing:** Each node can either explicitly determine the next node by returning an ID, or let an LLM decide the next step based on node descriptions.
- **Shared Context:** Data (e.g., logs or variable values) is maintained in a common context object that nodes can read or update.
- **LLM Integration:** Easily integrate with language models like Google’s Gemini (gemini-2.0-flash) or OpenAI’s gpt-4o  for decision making.
- **Input Validation:** The library ensures that all required fields are provided and unique node IDs are enforced.

## Installation

Install HLR from PyPI:

```bash
pip install hlr_agent
```

> **Note:** Package names are normalized by PyPI. Although the package is defined as `hlr_agent`, you can install it using either `hlr_agent` or `hlr-agent`.

## Usage

Below is an example of how to set up and run your hierarchical workflow.

### Define Node Functions

Each node function takes the agent as a parameter (so it can update the context or decide the next node explicitly). For example:

```python
from hlr_agent import Node, Agent

def func_input(agent):
    print("AGENT IN INPUT NODE")
    agent.context["context"] = "- Input node executed.\n"

def func_database(agent):
    print("AGENT IN DATABASE NODE")
    agent.context["context"] += "- Database node executed.\n"

def func_files(agent):
    print("AGENT IN FILES NODE")
    agent.context["context"] += "- Files node executed.\n"

def func_mailing(agent):
    print("AGENT IN MAILING NODE")
    agent.context["context"] += "- Mailing node executed.\n"

def func_output(agent):
    print("AGENT IN OUTPUT NODE")
    print("LOGS:\n" + agent.context["context"])
```

### Configure Nodes

Nodes are created by providing a unique node ID, a list of candidate child node IDs, an optional function to execute, and an optional description. For example:

```python
nodes = [
    Node("Input", children=["Database", "Files", "Mailing", "None"], func=func_input),
    Node("Database", children=["Output"], func=func_database, description="Select this if the user wants to use a database"),
    Node("Files", children=["Output"], func=func_files, description="Select this if the user wants to use a file"),
    Node("Mailing", children=["Output"], func=func_mailing, description="Select this if the user wants to use a mailing related functionality."),
    Node("None", children=["Output"], func=None, description="Select this node if the rest of the nodes are not valid for the request"),
    Node("Output", children=None, func=func_output),
]
```

### Initialize and Run the Agent

The `Agent` class handles the execution flow. It validates all required parameters and ensures that node IDs are unique.  
It also integrates with an LLM for dynamic routing when a node does not explicitly return the next node's ID.

```python
# Example user message that the LLM can consider when deciding the next node.
user_message = "I want to send a message to my boss"

agent = Agent(
    nodes=nodes,
    start_node_id="Input",
    end_node_id="Output",
    model="gemini-2.0-flash",
    api_key="YOUR_API_KEY_HERE",  # Replace with your API key
    user_message=user_message
)

agent.run()
```


## How It Works

1. **Initialization:**  
   Upon creating an `Agent`, mandatory fields are validated and an internal map of nodes is built.  
   Duplicate node IDs or missing parameters raise exceptions immediately.

2. **Execution Flow:**  
   The agent starts at the given `start_node_id` and executes the corresponding node’s function.  
   The node function may return the next node’s ID explicitly, or the agent will consult the LLM using the node descriptions and an optional shared context.

3. **Dynamic Routing:**  
   If more than one child node is available and their descriptions are valid, the LLM (via `get_next_node`) will determine the most relevant child node, based on the user message and any extra context provided.

4. **Termination:**  
   The workflow finishes when a node with no children is reached, or the agent is directed to the `end_node_id`, which may itself execute a termination function.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under Creative Commons Attribution-NonCommercial 4.0 International License
## Contact

For questions or issues, please contact davidsd.2704@gmail.com
