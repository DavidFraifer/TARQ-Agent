# HLR (Hierarchical LLM Routing)

**HLR** is a flexible and easy-to-use library for managing hierarchical workflows based on nodes, where each node executes a function and decides which node to execute next. It is ideal for applications that require sequential task processing with dynamic routing and redirection of the workflow.

## Features

- **Hierarchical Flow**: Nodes are organized hierarchically, allowing for structured workflows.
- **Flexibility**: Nodes can dynamically redirect the flow based on results or interactions.
- **Shared Context**: Nodes can share data with each other through a common context, allowing for the passing of information from one node to another.
- **Easy Integration**: The library is simple to integrate and can be used in a wide variety of projects.

## Installation

### From PyPI

You can install the **HLR** library from PyPI using the following command:

```bash
pip install hlr_agent
