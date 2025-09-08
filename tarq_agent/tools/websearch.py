from ..utils.console import console
import requests
from bs4 import BeautifulSoup
from ..internal.llm import llm_completion_async
from ..config import get_cached_api_key
from ..utils.pricing import llm_pricing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def _get_page_content(url, task_id=None, agent_id=None):
    """Extract clean content from a webpage."""
    start_time = time.perf_counter()
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Eliminar elementos no deseados
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'meta', 'link']):
            element.decompose()
        
        # Extraer contenido del body
        body = soup.body
        if body:
            text = body.get_text(separator=' ', strip=True)
            duration = time.perf_counter() - start_time
            console.info("Web Scraping", f"Fetched and cleaned content from {url} in {duration:.2f}s", task_id=task_id, agent_id=agent_id)
            return ' '.join(text.split())
        return None
    except Exception as e:
        console.error("Web Scraping", f"Error fetching content from {url}: {str(e)}", task_id=task_id, agent_id=agent_id)
        return None



def _search_web(task_memory, user_input, task_id, fast_search):
    """Search the web using Brave API and scrape pages in parallel with threads, with detailed logging."""
    start_time = time.perf_counter()

    # Get API key from centralized config system
    try:
        brave_key = get_cached_api_key('brave')
    except ValueError as e:
        console.error("Web Search", f"BRAVE API key not found: {str(e)}", task_id=task_id)
        return []

    console.info("Web Scraping", f"Getting Results for: {user_input}", task_id=task_id)
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "x-subscription-token": brave_key
            },
            params={
                "q": user_input,
                "count": 1 if fast_search else 5
            }
        ).json()
    except Exception as e:
        console.error("Web Search", f"API request failed: {str(e)}", task_id=task_id)
        return []

    search_results = []
    if 'web' in response and 'results' in response['web']:
        results = response['web']['results']
        console.info("Web Scraping", f"Launching scraping in parallel for {len(results)} URLs", task_id=task_id)

        # Parallelize with threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_result = {}
            for r in results:
                console.info("Thread Manager", f"Scheduling URL: {r['url']}", task_id=task_id)
                future = executor.submit(_threaded_scrape_wrapper, r['url'], task_id)
                future_to_result[future] = r

            for future in as_completed(future_to_result):
                r = future_to_result[future]
                try:
                    content = future.result()
                except Exception as e:
                    content = None
                    console.error("Web Scraping", f"Error processing {r['url']}: {str(e)}", task_id=task_id)

                search_results.append({
                    'title': r['title'],
                    'url': r['url'],
                    'description': r['description'],
                    'content': content
                })

    duration = time.perf_counter() - start_time
    console.success("Web Search", f"Completed search for '{user_input}' in {duration:.2f}s", task_id=task_id)
    return search_results


def _threaded_scrape_wrapper(url, task_id=None, agent_id=None):
    """Wrapper around _get_page_content to add logging per thread."""
    thread_name = threading.current_thread().name
    start_time = time.perf_counter()

    content = _get_page_content(url, task_id, agent_id)

    duration = time.perf_counter() - start_time
    console.success("Thread Done", f"[{thread_name}] Finished {url} in {duration:.2f}s", task_id=task_id, agent_id=agent_id)
    return content



async def _search_and_summarize(task_memory: list, query: str, task_id: int = 1, fast_search: bool = True, light_llm: str = "gemini-2.5-flash-lite"):
    start_time = time.perf_counter()
    LIMIT_LLM_CONTENT = 5000
    
    console.tool("Web Search", "Brave API - Data extracting", task_id=task_id)
    results = _search_web(task_memory=task_memory, user_input=query, task_id=task_id, fast_search=fast_search)
    
    if not results:
        console.error("Web Search", "No results found or API error", task_id=task_id)
        return "No results found", {"tokens_used": 0, "input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "total_cost": 0.0}
    
    console.success("Web Search", "Data extracted correctly", task_id=task_id)
    
    console.tool("Web Search", "Preparing LLM summary", task_id=task_id)
    content_to_summarize = f"Search Query: {query}\n\n"
    for result in results:
        content_to_summarize += f"Title: {result['title']}\n"
        content_to_summarize += f"Content: {result['content'][:LIMIT_LLM_CONTENT] if result['content'] else 'No content available'}\n\n"
    
    prompt = f"""Please provide a concise summary of the following web search results:
    {content_to_summarize}
    Provide a clear and informative summary that captures the key information from these results."""
    
    llm_start = time.perf_counter()
    summary, token_info = await llm_completion_async(
        model=light_llm,
        prompt=prompt,
        system_message=f"You are a helpful assistant that summarizes web search results clearly and concisely. The starting request was {query}, compose your summary based on that",
        max_tokens=500
    )
    llm_duration = time.perf_counter() - llm_start
    total_duration = time.perf_counter() - start_time
    console.info("LLM Summarization", f"Completed in {llm_duration:.2f}s, total function time {total_duration:.2f}s", task_id=task_id)

    return summary, token_info


async def web_search(task_memory, text, task_id=1, fast_search=True, light_llm: str = "gemini-2.5-flash-lite", agent_id: str = None):
    start_time = time.perf_counter()
    console.tool("Web Search", "LLM Query inference - Extracting search query", task_id=task_id, agent_id=agent_id)

    # Step 1: Extract user_input from text
    prompt = f"""
    Extract the browser search query from the following text. 
    Provide only the query string without extra explanation.

    Text: \"\"\"{text}\"\"\"
    """

    llm_start = time.perf_counter()
    user_input, token_info_extract = await llm_completion_async(
        model=light_llm,
        prompt=prompt,
        system_message="You are an assistant that extracts the user's search query from any given text.",
        max_tokens=100
    )

    

    llm_duration = time.perf_counter() - llm_start
    user_input = user_input.strip()
    console.info("Web Search", f"Inferred search query: '{user_input}' in {llm_duration:.2f}s", task_id=task_id)
  
    console.tool("Web Search", "Proceeding to web search", task_id=task_id)
    results, token_info_summary = await _search_and_summarize(task_memory=task_memory, query=user_input, task_id=task_id, fast_search=fast_search, light_llm=light_llm)
    
    total_duration = time.perf_counter() - start_time
    console.info("Web Search", f"Total web_search execution time: {total_duration:.2f}s", task_id=task_id)

    # Merge token info de ambas llamadas al LLM y calcular costo
    input_tokens_total = token_info_extract.get("input_tokens", 0) + token_info_summary.get("input_tokens", 0)
    output_tokens_total = token_info_extract.get("output_tokens", 0) + token_info_summary.get("output_tokens", 0)
    
    # Calculate total cost using the pricing system
    total_cost, _ = llm_pricing.calculate_cost(light_llm, input_tokens_total, output_tokens_total)
    
    merged_tokens = {
        "tokens_used": token_info_extract.get("total_tokens", 0) + token_info_summary.get("total_tokens", 0),
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
        "llm_calls": 2,  # websearch always makes 2 LLM calls (query extraction + summarization)
        "total_cost": total_cost
    }

    # Log websearch completion with token info (but don't complete the task)
    console.info("Web Search", f"Search completed. Tokens: {merged_tokens['tokens_used']} | Calls: {merged_tokens['llm_calls']} | Cost: ${merged_tokens['total_cost']:.5f}", task_id=task_id)

    return results, merged_tokens

