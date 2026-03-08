import ray 
import asyncio
from tests.workers.rollout.my_tools_sever import WebSearchToolClient

@ray.remote
def save_web_search_cache_remote(base_url: str, parameters: dict):
    async def _save():
        client = WebSearchToolClient(base_url=base_url, parameters=parameters)
        return await client.save_cache()
    return asyncio.run(_save())


@ray.remote
class WebSearchCacheSaver: # Using a class as a remote actor for better state management
    def __init__(self, base_url: str, parameters: dict):
        self.base_url = base_url
        self.parameters = parameters
        # Client initialized once per actor
        self.client = WebSearchToolClient(base_url=self.base_url, parameters=self.parameters)

    async def save_cache(self):
        import traceback
        # No asyncio.run here, as this method runs within Ray's event loop
        try:
            print(f"[DEBUG] Connecting to: {self.base_url}/save_cache")
            result = await self.client.save_cache()
            print(f"[DEBUG] save_cache result: {result}")
            await self.client.aclose() # Close client after use
            return result
        except Exception as e:
            print(f"❌ [ERROR] Exception in save_cache: {e}")
            traceback.print_exc()
            return f"[Client Error] {str(e)}"
