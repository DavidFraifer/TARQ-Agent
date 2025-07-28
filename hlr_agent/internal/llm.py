from openai import AsyncOpenAI
from json import loads, JSONDecodeError
from google import genai
from google.genai import types
import asyncio

ROUTING_SYSTEM_MESSAGE = (
    "Choose best next node. Prefer 'Output' if done. Return node ID only."
)


def _get_api_key(model: str) -> str:
    from ..config import get_cached_api_key
    if model == "gpt-4o":
        return get_cached_api_key('openai')
    elif model.startswith("gemini"):
        return get_cached_api_key('gemini')
    else:
        raise ValueError(f"Unsupported model: {model}")


async def llm_completion_async(
    model: str,
    prompt: str,
    system_message: str = "",
    temperature: float = 0.1,
    max_tokens: int = 150,
    response_format: str = "text"
) -> tuple[str, int]:
    
    api_key = _get_api_key(model)
    
    if model == "gpt-4o":
        client = AsyncOpenAI(api_key=api_key)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        extra_params = {}
        if response_format == "json":
            extra_params["response_format"] = {"type": "json_object"}
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params
        )
        
        content = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        return content, tokens
    
    elif model.startswith("gemini"):
        # Run Gemini in thread pool since it's not async
        return await asyncio.to_thread(_gemini_sync, model, prompt, system_message, temperature, max_tokens, response_format, api_key)
    
    else:
        raise ValueError(f"Unsupported model: {model}")


def _gemini_sync(model: str, prompt: str, system_message: str, temperature: float, max_tokens: int, response_format: str, api_key: str) -> tuple[str, int]:
    client = genai.Client(api_key=api_key)
    full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
    
    content_for_tokens = [types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])]
    
    config_params = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "top_p": 0.9,
        "top_k": 40
    }
    if response_format == "json":
        config_params["response_mime_type"] = "application/json"
    
    if model == "gemini-2.5-flash-lite":
        config_params["thinking_config"] = types.ThinkingConfig(
            thinking_budget=0,
        )
    
    response = client.models.generate_content(
        model=model,
        contents=content_for_tokens,
        config=types.GenerateContentConfig(**config_params)
    )
    
    content = response.text.strip()
    
    try:
        input_token_response = client.models.count_tokens(
            model=model, 
            contents=content_for_tokens
        )
        input_tokens = input_token_response.total_tokens
        
        # Count output tokens
        output_content = [types.Content(role="assistant", parts=[types.Part.from_text(text=content)])]
        output_token_response = client.models.count_tokens(
            model=model,
            contents=output_content
        )
        output_tokens = output_token_response.total_tokens
        
        total_tokens = input_tokens + output_tokens
    except Exception as e:
        print(f"Warning: Could not count tokens for Gemini: {e}")
        total_tokens = len(full_prompt + content) // 4
    
    return content, total_tokens


async def get_next_node_async(
    children_ids: list[str],
    children_descriptions: list[str],
    model: str,
    user_message: str,
    extra_context: str = ""
) -> tuple[str, int]:
    
    from ..utils.cleaner import clean_tool_name
    
    system_message = f"{ROUTING_SYSTEM_MESSAGE} Context: {extra_context}" if extra_context else ROUTING_SYSTEM_MESSAGE
    user_nodes = "\n".join(f"{node_id}: {desc}" for node_id, desc in zip(children_ids, children_descriptions))
    prompt = f"{user_message}\n\nNodes:\n{user_nodes}"

    try:
        if model == "gpt-4o":
            response, tokens = await llm_completion_async(model, prompt, system_message, 0.1, 32)
            raw_choice = response.strip().split()[0]
        else:
            response, tokens = await llm_completion_async(model, prompt, system_message + "\nJSON: {\"nodeId\": \"<id>\"}", 0.1, 32, "json")
            raw_choice = loads(response).get("nodeId", "").strip().split()[0]
        
        if raw_choice in children_ids:
            return raw_choice, tokens
        else:
            cleaned_choice = clean_tool_name(raw_choice, children_ids)
            if cleaned_choice:
                return cleaned_choice, tokens
            else:
                return raw_choice, tokens  
                
    except Exception as e:
        print(f"❌ LLM routing error: {e}")
        return (children_ids[0] if children_ids else "Output"), 0


async def is_task_complete(user_message: str, graph_result: str, model: str) -> tuple[bool, int]:
    
    system_message = """Check if request is complete.

Rules:
1. Periodic tasks (daily, hourly): need multiple iterations
2. Single tasks: check if goal achieved
3. "Multiple/all emails": one email insufficient

JSON: {"complete": true/false}"""
    
    prompt = f"""Request: {user_message}
Results: {graph_result}

Complete?"""
    
    try:
        response, tokens = await llm_completion_async(model, prompt, system_message, 0.1, 50, "json")
        result = loads(response)
        complete = result.get("complete", False)
        
        return complete, tokens
    except Exception as e:
        print(f"❌ Task completion check error: {e}")
        if "periodic" in user_message.lower() or "times" in user_message.lower() or "daily" in user_message.lower():
            return False, 0
        return False, 0