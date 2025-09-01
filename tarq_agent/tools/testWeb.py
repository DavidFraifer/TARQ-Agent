import asyncio
from .websearch import web_search


from ..utils.console import console


async def main():
    # Example usage
    text = "Search in web about catenary, i dont know what it is, basic explanation"
    console.info (f"User query: \" {text} \" ")
    summary = await web_search(
        task_memory=[],  # Empty list for task memory
        text=text,  # Your search query
        task_id=1,  # Task identifier
        fast_search=False  # True for single result, False for multiple results
    )
    
    print("\n=== Search Summary ===")
    print(summary)
    print("=====================")

if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())

    