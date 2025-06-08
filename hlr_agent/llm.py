from openai import OpenAI
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
        client = OpenAI(api_key=api_key)

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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_nodes},
            ],
        )
        return response.choices[0].message.content.strip().split()[0]


    elif model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]: # Added new Gemini models


        client = genai.Client(api_key=api_key)
        
        full_prompt = (
            "SYSTEM_MESSAGE:\\n"
            "You are an AI assistant that helps decide the next node in a conversation.\\n"
            f"Additional context:\\n{extra_context}\\n\\n"
            "Given the following list of nodes with their descriptions, return ONLY the ID of the node that is most relevant.\\n"
            "Do not include any additional text. \\n"
            f"User's main request:\\n{user_message}\\n\\n"
            "Available nodes:\\n" +
            "\\n".join(
                f"{node_id}: {desc}" for node_id, desc in zip(children_ids, children_descriptions)
            )
        )

        contents = [
            types.Content(
                role="user", 
                parts=[types.Part.from_text(text=full_prompt)]
            )
        ]
        
        # Optimized generation config for faster responses
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,          
            max_output_tokens=32, 
            top_p=0.1,              
            top_k=1,                 
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
        
        response = client.models.generate_content(
            model=model, #8b: 1.35s, # gemini 1-5-flash: 1.4s 
            contents=contents,
            config=generate_content_config, 
        )
        
        try:
      
            response_json = loads(response.text)
            return response_json.get("nodeId", "").strip().split()[0]
        except JSONDecodeError as e:
            raise ValueError(f"Error converting result to dict: {e}. Raw response: {response.text}")
        except AttributeError: 
            raise ValueError(f"Unexpected response format from LLM. Full response: {response}")

    else:
        raise ValueError(f"Model '{model}' not supported.")