from tests.workers.rollout.my_tools import WebSearchTool
import asyncio


if __name__ == "__main__":
    # Ensure this script is run as an async program.
    # asyncio.run() handles the event loop.
    cache_file = "/workspace/cache/serper_search_cache.jsonl"
    api_key=""
    web_search_tool = WebSearchTool(api_key=api_key, cache_file=cache_file)

    web_search_tool.save_cache()

