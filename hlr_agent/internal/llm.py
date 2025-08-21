from openai import AsyncOpenAI
from google import genai
from google.genai import types
import asyncio

def _get_api_key(model: str) -> str:
    from ..config import get_cached_api_key
    if model.startswith("gpt-"):  # Handle all GPT models (gpt-4o, gpt-4o-mini, etc.)
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
) -> tuple[str, dict]:
    
    api_key = _get_api_key(model)
    
    if model.startswith("gpt-"):  # Handle all GPT models
        client = AsyncOpenAI(api_key=api_key)
        
        # GPT-5 models use the new responses API
        if model.startswith("gpt-5"):
            # Combine system message and prompt for GPT-5
            full_input = f"{system_message}\n\n{prompt}" if system_message else prompt
            
            extra_params = {
                "reasoning": {"effort": "minimal"},
                "text": {"verbosity": "low"}
            }
            
            if response_format == "json":
                # For GPT-5, modify the input to request JSON format
                full_input += "\n\nRespond with valid JSON only."
            
            response = await client.responses.create(
                model=model,
                input=full_input,
                **extra_params
            )
            
            # Access the text content correctly from GPT-5 response
            content = response.output_text.strip()
            token_info = {
                "input_tokens": getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0,
                "output_tokens": getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0,
                "total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
            }
            
        else:
            # GPT-4 and older models use chat completions API
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            extra_params = {}
            if response_format == "json":
                extra_params["response_format"] = {"type": "json_object"}
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_params
            )
            
            content = response.choices[0].message.content.strip()
            token_info = {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        return content, token_info
    
    elif model.startswith("gemini"):
        # Run Gemini in thread pool since it's not async
        return await asyncio.to_thread(_gemini_sync, model, prompt, system_message, temperature, max_tokens, response_format, api_key)
    
    else:
        raise ValueError(f"Unsupported model: {model}")


def _gemini_sync(model: str, prompt: str, system_message: str, temperature: float, max_tokens: int, response_format: str, api_key: str) -> tuple[str, dict]:
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
    
    if model in ["gemini-2.5-flash-lite", "gemini-2.5-flash"]:
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
        
        token_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    except Exception as e:
        print(f"Warning: Could not count tokens for Gemini: {e}")
        estimated_total = len(full_prompt + content) // 4
        token_info = {
            "input_tokens": len(full_prompt) // 4,
            "output_tokens": len(content) // 4,
            "total_tokens": estimated_total
        }
    
    return content, token_info