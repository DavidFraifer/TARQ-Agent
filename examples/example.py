from hlr_agent import Node, Agent

def func_input(agent):
    print("AGENT IN INPUT NODE")
    agent.context["context"] = "- Input node executed.\n"

def func_database(agent):
    print("AGENT IN DATABASE NODE")
    agent.context["log"] += "- Database node executed.\n"

def func_files(agent):
    print("AGENT IN FILES NODE")
    agent.context["log"] += "- Files node executed.\n"

def func_mailing(agent):
    print("AGENT IN MAILING NODE")
    agent.context["log"] += "- Mailing node executed.\n"

def func_output(agent):
    print("AGENT IN OUTPUT NODE")
    print("LOGS:\n" + agent.context["log"])

nodes = [
    Node("Input", children=["Database","Files", "Mailing", "None"], func=func_input),
    Node("Database", children=["Output"], func=func_database, description="Select this if the user wants to use a database"),
    Node("Files", children=["Output"], func=func_files, description="Select this if the user wants to use a file"),
    Node("Mailing", children=["Output"], func=func_mailing, description="Select this if the user wants to use a mailing related functionality."),
    Node("None", children=["Output"], func=None, description="Select this node if the rest o the nodes are not valid for the request"),
    Node("Output", children=None, func=func_output),
]

# Example user message that you want the LLM to consider when deciding the next node.
user_message = "Quiero que mires mi base de datos"

agent = Agent(
    nodes=nodes,    
    start_node_id="Input",
    end_node_id="Output",
    model="gemini-2.0-flash",
    api_key="",
    user_message=user_message
)

agent.run()