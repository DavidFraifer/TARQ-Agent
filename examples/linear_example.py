import time
from hlr_agent import Node, Agent


def func_database(agent):
    print("Querying database...")
    found_emails = [  # Simulate database query result
        "user1@example.com",
        "test.user@sample.net",
        "another@domain.org"
    ]
    print(f"Database: Found {len(found_emails)} emails.")
    agent.context["info"] += "\nQuery result:\n" + str(found_emails)


def func_mailing(agent):
    print("Emails has been formatted.")
    redacted_emails = [  # Simulate redaction of emails
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


nodes = [
    Node("Input", children=["Database", "Mailing", "None"]),
    Node("Database", children=["Output"], func=func_database, description="Select this if the user wants to use a database"),
    Node("Mailing", children=["Output"], func=func_mailing, description="Select this if the user wants to use a mailing related functionality."),
    Node("None", children=["Output"], description="Select this node if the rest o the nodes are not valid for the request"),
    Node("Output", children=None, func=func_output),
]

agent = Agent(
    nodes=nodes,
    start_node_id="Input",
    end_node_id="Output",
    model="gemini-1.5-flash-8b",  # Updated to use one of the new models
    api_key=""
)

start_time = time.time()
agent.run(
    """
    Query in the database the users registered.
    """
)
end_time = time.time()
print(f"Time spent in agent.run: {end_time - start_time:.2f} seconds")