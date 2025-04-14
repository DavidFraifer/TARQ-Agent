import openai
from json import loads, JSONDecodeError
from google import genai
from google.genai import types
def get_next_node(
    children_ids: list[str],
    children_descriptions: list[str],
    model: str,
    api_key: str,
    user_message: str,
    extra_context: str = ""
) -> str:
    """
    Returns the ID of the node that best matches the given user_message and extra_context.
    """

    if model == "gpt-4o":
        openai.api_key = api_key
        # Incorporate extra_context into system_message
        system_message = (
            "SYSTEM_MESSAGE:\n"
            "You are an AI assistant that helps decide the next node in a conversation.\n"
            f"Additional context:\n{extra_context}\n\n"
            "Given the following list of nodes with their descriptions, return ONLY the ID of the node that is most relevant.\n"
            "Do not include any additional text. "
            f"User's main request:\n{user_message}"
        )
        user_nodes = "\n".join(
            f"{node_id}: {desc}" for node_id, desc in zip(children_ids, children_descriptions)
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_nodes},
            ],
            max_tokens=10,
            temperature=0,
        )
        return response.choices[0].message.content.strip().split()[0]

    elif model == "gemini-2.0-flash":


        client = genai.Client(api_key=api_key)
        # Incorporate extra_context into system_message
        system_message = (
            "SYSTEM_MESSAGE:\n"
            "You are an AI assistant that helps decide the next node in a conversation.\n"
            f"Additional context:\n{extra_context}\n\n"
            "Given the following list of nodes with their descriptions, return ONLY the ID of the node that is most relevant.\n"
            "Do not include any additional text. "
            f"User's main request:\n{user_message}"
        )
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_message)]
            )
        ]
        nodes_text = "\n".join(
            f"{node_id}: {desc}" for node_id, desc in zip(children_ids, children_descriptions)
        )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=nodes_text)]
            )
        )
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["nodeId"],
                properties={
                    "nodeId": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The ID of the selected node."
                    )
                }
            )
        )
        result = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
        try:
            response_json = loads(result)
            return response_json.get("nodeId", "").strip().split()[0]
        except JSONDecodeError as e:
            raise ValueError(f"Error converting result to dict: {e}. Raw response: {result}")

    else:
        raise ValueError(f"Model '{model}' not supported.")