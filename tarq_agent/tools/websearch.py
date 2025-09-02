from ..utils.console import console
import requests
from pathlib import Path
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from ..internal.llm import llm_completion_async
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def _normalize_tokens(token_info: dict) -> dict:
    return {
        "tokens_used": token_info.get("tokens_used", token_info.get("total_tokens", 0)),
        "input_tokens": token_info.get("input_tokens", 0),
        "output_tokens": token_info.get("output_tokens", 0),
        "llm_calls": token_info.get("llm_calls", 0) or 1,  # cada llamada = 1
        "total_cost": token_info.get("total_cost", 0.0)
    }


def _get_page_content(url):
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
            console.info("Web Scraping", f"Fetched and cleaned content from {url} in {duration:.2f}s")
            return ' '.join(text.split())
        return None
    except Exception as e:
        console.error("Web Scraping", f"Error fetching content from {url}: {str(e)}")
        return None



# Si se van a buscar más de 5 resultados refactorizar para aiohttp y asyncio
def _search_web(task_memory, user_input, task_id, fast_search):
    """Search the web using Brave API and scrape pages in parallel with threads, with detailed logging."""
    start_time = time.perf_counter()

    # Load API key
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    brave_key = os.getenv('BRAVE_KEY')

    console.info("Web Scraping", f"Getting Results for: {user_input}")
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
        console.error("Web Search", f"API request failed: {str(e)}")
        return []

    search_results = []
    if 'web' in response and 'results' in response['web']:
        results = response['web']['results']
        console.info("Web Scraping", f"Launching scraping in parallel for {len(results)} URLs")

        # 👇 paralelizamos con threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_result = {}
            for r in results:
                console.info("Thread Manager", f"Scheduling URL: {r['url']}")
                future = executor.submit(_threaded_scrape_wrapper, r['url'])
                future_to_result[future] = r

            for future in as_completed(future_to_result):
                r = future_to_result[future]
                try:
                    content = future.result()
                except Exception as e:
                    content = None
                    console.error("Web Scraping", f"Error processing {r['url']}: {str(e)}")

                search_results.append({
                    'title': r['title'],
                    'url': r['url'],
                    'description': r['description'],
                    'content': content
                })

    duration = time.perf_counter() - start_time
    console.success("Web Search", f"Completed search for '{user_input}' in {duration:.2f}s")
    return search_results


def _threaded_scrape_wrapper(url):
    """Wrapper around _get_page_content to add logging per thread."""
    thread_name = threading.current_thread().name
    #console.info("Thread Start", f"[{thread_name}] Starting scrape for {url}")
    start_time = time.perf_counter()

    content = _get_page_content(url)

    duration = time.perf_counter() - start_time
    console.success("Thread Done", f"[{thread_name}] Finished {url} in {duration:.2f}s")
    return content



async def _search_and_summarize(task_memory: list, query: str, task_id: int = 1, fast_search: bool = True):
    start_time = time.perf_counter()
    LIMIT_LLM_CONTENT = 5000
    
    console.tool("Web Search", "Brave API - Data extracting")
    results = _search_web(task_memory=task_memory, user_input=query, task_id=task_id, fast_search=fast_search)
    
    if not results:
        console.error("Web Search", "No results found or API error")
        return "No results found", {"tokens_used": 0, "input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "total_cost": 0.0}
    
    console.success("Web Search", "Data extracted correctly")
    
    console.tool("Web Search", "Preparing LLM summary")
    content_to_summarize = f"Search Query: {query}\n\n"
    for result in results:
        content_to_summarize += f"Title: {result['title']}\n"
        content_to_summarize += f"Content: {result['content'][:LIMIT_LLM_CONTENT] if result['content'] else 'No content available'}\n\n"
    
    prompt = f"""Please provide a concise summary of the following web search results:
    {content_to_summarize}
    Provide a clear and informative summary that captures the key information from these results."""
    
    llm_start = time.perf_counter()
    summary, token_info = await llm_completion_async(
        model="gemini-2.5-flash-lite",
        prompt=prompt,
        system_message=f"You are a helpful assistant that summarizes web search results clearly and concisely. The starting request was {query}, compose your summary based on that",
        max_tokens=500
    )
    llm_duration = time.perf_counter() - llm_start
    total_duration = time.perf_counter() - start_time
    console.info("LLM Summarization", f"Completed in {llm_duration:.2f}s, total function time {total_duration:.2f}s")

    return summary, token_info


async def web_search(task_memory, text, task_id=1, fast_search=True):
    start_time = time.perf_counter()
    console.tool("Web Search", "LLM Query inference - Extracting search query")

    # Step 1: Extract user_input from text
    prompt = f"""
    Extract the browser search query from the following text. 
    Provide only the query string without extra explanation.

    Text: \"\"\"{text}\"\"\"
    """

    llm_start = time.perf_counter()
    user_input, token_info_extract = await llm_completion_async(
        model="gemini-2.5-flash-lite",
        prompt=prompt,
        system_message="You are an assistant that extracts the user's search query from any given text.",
        max_tokens=100
    )

    

    llm_duration = time.perf_counter() - llm_start
    user_input = user_input.strip()
    console.info("Web Search", f"Inferred search query: '{user_input}' in {llm_duration:.2f}s")
  
    console.tool("Web Search", "Proceeding to web search")
    results, token_info_summary = await _search_and_summarize(task_memory=task_memory, query=user_input, task_id=task_id, fast_search=fast_search)
    
    total_duration = time.perf_counter() - start_time
    console.info("Web Search", f"Total web_search execution time: {total_duration:.2f}s")

    # Merge token info de ambas llamadas al LLM
    merged_tokens = {
        "tokens_used": token_info_extract.get("total_tokens", 0) + token_info_summary.get("total_tokens", 0),
        "input_tokens": token_info_extract.get("input_tokens", 0) + token_info_summary.get("input_tokens", 0),
        "output_tokens": token_info_extract.get("output_tokens", 0) + token_info_summary.get("output_tokens", 0),
        "llm_calls": token_info_extract.get("llm_calls", 0) + token_info_summary.get("llm_calls", 0),               # NOT RETURNED IN SUMMARY
        "total_cost": token_info_extract.get("total_cost", 0.0) + token_info_summary.get("total_cost", 0.0)         # NOT RETURNED IN SUMMARY
    }

    # Aquí ya puedes llamar a task_summary
    console.task_summary(
        task_id=task_id,
        duration=total_duration,
        tokens=merged_tokens,
        status="completed",
        final_message="Web search and summarization done"
    )

    return results

