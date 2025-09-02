import asyncio
from .websearch import web_search


from ..utils.console import console


async def main():
    # Example usage
    text = "How is the weather in azuqueca de henares?"
    console.info (f"User query: \" {text} \" ")
    summary = await web_search(
        task_memory=[],  # Empty list for task memory
        text=text,  # Your search query
        task_id=1,  # Task identifier
        fast_search=True  # True for single result, False for multiple results
    )
    
    print("\n=== Search Summary ===")
    print(summary)
    print("=====================")

if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())

    