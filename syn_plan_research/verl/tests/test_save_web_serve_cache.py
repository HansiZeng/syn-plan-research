import http
import os
import ray
import json
import asyncio
import tempfile
from pathlib import Path

# Modification: Import WebSearchToolServer, WebSearchToolClient directly if they are in the same directory,
# or adjust path if they are in a different module.
# Assuming they are in the same relative path from where this script runs, or sys.path is correctly configured.
from tests.workers.rollout.my_tools_sever import WebSearchToolServer, WebSearchToolClient

API_KEY = "005fba2b8daa23f10be87a7d76a5bd37c99627c7"
cache_file = "/workspace/cache/serper_search_cache.json"

# Modification: Make this actor itself an async actor and remove asyncio.run
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

# Modification: Call the async method on the actor
@ray.remote(num_cpus=1)
def test_crawl_server(cache_file: str):
    from fastapi import FastAPI
    import httpx
    import time
    import uvicorn # Ensure uvicorn is imported here too, if _start_fastapi_server is in WebSearchToolServer

    print(f"🧪 [TEST] Using cache file: {cache_file}")

    # Step 1: 启动 WebSearchToolServer
    search_actor = WebSearchToolServer.options(name="web_search_server").remote(
        api_key=API_KEY,
        cache_file=cache_file,
    )

    # Step 2: 获取 server address 和参数 schema
    addr = ray.get(search_actor.get_server_address.remote())
    parameters = ray.get(search_actor.get_parameters.remote())

    print(f"🌐 [TEST] WebSearchToolServer is running at: http://{addr}")
    print(f"📄 [TEST] Tool parameters schema: {json.dumps(parameters, indent=2)}")

    # Step 3: 构造请求
    print("💾 Start to save web search cache...")
    
    # Modification: Instantiate the WebSearchCacheSaver actor
    cache_saver_actor = WebSearchCacheSaver.remote(
        base_url=f"http://{addr}",
        parameters=parameters,
    )
    # Modification: Call the async save_cache method on the actor and ray.get its result
    status = ray.get(cache_saver_actor.save_cache.remote()) 
    
    print(f"✅ Web search cache saved: {status}")


if __name__ == "__main__":
    # Ensure ray.init() is called only once at the top level
    ray.init() 
    
    ray.get(test_crawl_server.remote(cache_file))
    ray.shutdown()