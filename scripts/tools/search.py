import os
import httpx
import logging
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("tavily-search")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

async def tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> str:
    """
    Search the web using Tavily API.
    """
    if not TAVILY_API_KEY:
        log.error("TAVILY_API_KEY not found in environment variables.")
        return "Tavily API key is missing."

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    return "No results found."

                formatted_results = []
                for res in results:
                    formatted_results.append(f"Source: {res.get('url')}\nContent: {res.get('content')}\n")
                
                return "\n---\n".join(formatted_results)
            else:
                log.error(f"Tavily API error: {response.status_code} - {response.text}")
                return f"Error from Tavily API: {response.status_code}"
    except Exception as e:
        log.error(f"Tavily search failed: {e}")
        return f"Search tool error: {str(e)}"

if __name__ == "__main__":
    import asyncio
    # Simple test
    async def main():
        res = await tavily_search("Weather in Hanoi today")
        print(res)
    asyncio.run(main())