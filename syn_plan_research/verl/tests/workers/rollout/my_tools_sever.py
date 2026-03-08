import requests
import time
import random
import logging
import asyncio
import socket
import json
from itertools import cycle

from typing import List, Dict, Optional, Union, Any
import httpx
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import aiohttp
from typing import Union, Optional, List, Dict
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import os
import json
import fastapi
import uvicorn
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import ray

from tests.workers.rollout.my_utils import json_loads
from tests.workers.rollout.my_tools import WebSearchTool, CrawlWebpageTool, CrawlWebpageToolV2


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

# -------------- server --------------- 
@ray.remote(num_cpus=1)
class WebSearchToolServer:
    def __init__(self, api_key: str, cache_file: Optional[str] = None):
        self.tool = WebSearchTool(api_key=api_key, cache_file=cache_file)
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            print("[WebSearchToolServer] FastAPI startup.")
            self.server_ready.set()
            yield
            print("[WebSearchToolServer] FastAPI shutdown.")
            os._exit(0)

        app = FastAPI(lifespan=lifespan)

        @app.post("/call")
        async def call_tool(request: Request):
            try:
                params = await request.json()
                result = await self.tool.call(params)
                return JSONResponse(content={"result": result, "error": None})
            except Exception as e:
                return JSONResponse(content={"result": None, "error": f"[Server Error] {str(e)}"})

        @app.post("/save_cache")
        async def save_cache():
            try:
                self.tool.save_cache()
                return JSONResponse(content={"status": "Success", "message": "Cache saved", "error": None})
            except Exception as e:
                return JSONResponse(content={"status": "Failed", "error": f"[Server Error] {str(e)}"})

        self.port = self._get_free_port()
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

        @app.get("/metadata")
        async def get_metadata():
            return JSONResponse(content=self.tool.metadata())

    def _get_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
        
    def get_metadata(self) -> Dict:
        return self.tool.metadata()

    async def get_server_address(self) -> str:
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"
    
    def get_parameters(self) -> Union[List[dict], dict]:
        return self.tool.parameters

# 08/06/2025: increase the CPUs to 4. 
# Verl 训练时候 我 inference 调用 这个 tool 疯狂遇到 timeout error
# 可能是因为这个 tool 的并发量太大了，导致 CPU 占用率过高。
# 现在改成 4 个 CPU，
@ray.remote(num_cpus=4)
class CrawlWebpageToolServer:
    def __init__(self, 
                 semaphore_limit: int = 32, 
                 cache_file: Optional[str] = None, 
                 snippet_cache_file: Optional[str] = None,
                 fetch_timeout: int = 300):
        self._semaphore = asyncio.Semaphore(semaphore_limit)
        self.tool = CrawlWebpageTool(
            semaphore=self._semaphore,
            cache_file=cache_file,
            snippet_cache_file=snippet_cache_file,
            fetch_timeout=fetch_timeout
        )
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_server())

    async def _start_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print("[FastAPI] startup")
            self.server_ready.set()
            yield
            print("[FastAPI] shutdown — killing process")
            os._exit(0)

        app = fastapi.FastAPI(lifespan=lifespan)

        @app.post("/call")
        async def call_tool(request: Request):
            try:
                data = await request.json()
                params = data.get("params", {})
                keep_links = data.get("keep_links", False)
                result = await self.tool.call(params=params, keep_links=keep_links)
                return JSONResponse(content={"result": result, "error": None})
            except Exception as e:
                return JSONResponse(content={"result": None, "error": f"[Server Error] {str(e)}"})

        @app.post("/save_snippet")
        async def save_snippet(request: Request):
            try:
                data = await request.json()
                self.tool.save_snippet(data["url"], data["snippet"])
                return JSONResponse(content={"status": "ok", "error": None})
            except Exception as e:
                return JSONResponse(content={"status": "failed", "error": f"[Server Error] {str(e)}"})

        @app.post("/save_content")
        async def save_content(request: Request):
            try:
                data = await request.json()
                self.tool.save_content(data["url"], data["content"])
                return JSONResponse(content={"status": "ok", "error": None})
            except Exception as e:
                return JSONResponse(content={"status": "failed", "error": f"[Server Error] {str(e)}"})

        @app.get("/get_snippet")
        async def get_snippet(request: Request):
            try:
                url = request.query_params.get("url")
                result = self.tool.get_snippet(url)
                return JSONResponse(content={"snippet": result, "error": None})
            except Exception as e:
                return JSONResponse(content={"snippet": None, "error": f"[Server Error] {str(e)}"})

        @app.get("/get_content")
        async def get_content(request: Request):
            try:
                url = request.query_params.get("url")
                result = self.tool.get_content(url)
                return JSONResponse(content={"content": result, "error": None})
            except Exception as e:
                return JSONResponse(content={"content": None, "error": f"[Server Error] {str(e)}"})

        self.port = self._get_free_port()
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

        @app.get("/metadata")
        async def get_metadata():
            return JSONResponse(content=self.tool.metadata())

    def _get_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    async def get_server_address(self) -> str:
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    def get_metadata(self) -> Dict:
        return self.tool.metadata()

    def get_parameters(self) -> Union[List[dict], dict]:
        return self.tool.parameters


# 08/13/2025: 我发现了rollout 时候的 bottlneck 在 CrawlWebpageToolServer 上。
@ray.remote(num_cpus=1)
class CrawlWebpageToolServerV2:
    def __init__(
        self,
        semaphore_limit: int = 1,
        cache_file: Optional[str] = None,
        snippet_cache_file: Optional[str] = None,
        fetch_timeout: int = 300,
    ):
        self._semaphore_limit = semaphore_limit
        self.tool = CrawlWebpageToolV2(
            semaphore=asyncio.Semaphore(semaphore_limit),
            cache_file=cache_file,
            snippet_cache_file=snippet_cache_file,
            fetch_timeout=fetch_timeout,
        )
        self.address = ray._private.services.get_node_ip_address()
        self.port: Optional[int] = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_server())

    def _get_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    async def _start_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            # startup
            self.server_ready.set()
            yield
            # shutdown（FastAPI 自己的优雅退出不会调用我们的 aclose，这里兜底）
            try:
                await self.tool.aclose()
            finally:
                pass

        app = fastapi.FastAPI(lifespan=lifespan)

        @app.post("/call")
        async def call_tool(request: Request):
            try:
                data = await request.json()
                params = data.get("params", {})
                keep_links = data.get("keep_links", False)
                result = await self.tool.call(params=params, keep_links=keep_links)
                return JSONResponse(content={"result": result, "error": None})
            except Exception as e:
                return JSONResponse(content={"result": None, "error": f"[Server Error] {str(e)}"})

        @app.post("/save_snippet")
        async def save_snippet(request: Request):
            try:
                data = await request.json()
                self.tool.save_snippet(data["url"], data["snippet"])
                return JSONResponse(content={"status": "ok", "error": None})
            except Exception as e:
                return JSONResponse(content={"status": "failed", "error": f"[Server Error] {str(e)}"})

        @app.post("/save_content")
        async def save_content(request: Request):
            try:
                data = await request.json()
                self.tool.save_content(data["url"], data["content"])
                return JSONResponse(content={"status": "ok", "error": None})
            except Exception as e:
                return JSONResponse(content={"status": "failed", "error": f"[Server Error] {str(e)}"})

        @app.get("/get_snippet")
        async def get_snippet(url: str):
            try:
                result = self.tool.get_snippet(url)
                return JSONResponse(content={"snippet": result, "error": None})
            except Exception as e:
                return JSONResponse(content={"snippet": None, "error": f"[Server Error] {str(e)}"})

        @app.get("/get_content")
        async def get_content(url: str):
            try:
                result = self.tool.get_content(url)
                return JSONResponse(content={"content": result, "error": None})
            except Exception as e:
                return JSONResponse(content={"content": None, "error": f"[Server Error] {str(e)}"})

        @app.get("/metadata")
        async def get_metadata():
            return JSONResponse(content=self.tool.metadata())

        @app.post("/shutdown")
        async def http_shutdown():
            # 允许通过 HTTP 触发优雅关机（可选）
            asyncio.create_task(self._graceful_exit())
            return {"status": "shutting_down"}

        self.port = self._get_free_port()
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    async def _graceful_exit(self):
        try:
            await self.tool.aclose()
        finally:
            os._exit(0)

    async def get_server_address(self) -> str:
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    # 供外部（Ray）调用的优雅停机
    async def shutdown(self):
        await self._graceful_exit()

    # 兼容原接口
    def get_metadata(self) -> Dict[str, Any]:
        return self.tool.metadata()

    def get_parameters(self) -> Union[List[dict], dict]:
        return self.tool.parameters
    

# ------------------ client ---------------

class WebSearchToolClient:
    def __init__(self, base_url: str, parameters: Union[List[dict], dict], timeout: float = 60.0, max_concurrency: int = 8):
        """
        Args:
            base_url: e.g., "http://127.0.0.1:45678"
        """
        self.parameters = parameters
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=timeout)
        self.save_cache_client = httpx.AsyncClient(timeout=timeout*3)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def call(self, params: Union[str, Dict]) -> Union[List[Dict], str]:
        """
        Args:
            query: Either a raw query string or dict with {'query': ...}
        Returns:
            List of results or error string.
        """
        async with self._semaphore:  # Apply concurrency control
            try:
                resp = await self.client.post(f"{self.base_url}/call", json=params)
                resp.raise_for_status()
                resp_json = resp.json()
                if resp_json.get("error"):
                    return f"[Server Error] {resp_json['error']}"
                return resp_json["result"]
            except httpx.RequestError as e:
                return f"[Client Error in call] Request failed: {str(e)}"
            except httpx.TimeoutException:
                return "[Client Error] (in call) Request timed out"
            except Exception as e:
                return f"[Client Error in call] Unexpected error: {str(e)}"

    async def save_cache(self) -> str:
        try:
            resp = await self.save_cache_client.post(f"{self.base_url}/save_cache")
            resp.raise_for_status()
            resp_json = resp.json()
            if resp_json.get("error"):
                return f"[Server Error] {resp_json['error']}"
            return resp_json["status"]
        except Exception as e:
            return f"[Client Error in save_cache] {str(e)}"

    async def aclose(self):
        await self.client.aclose()

    async def get_metadata(self) -> Dict[str, Any]:
        resp = await self.client.get(f"{self.base_url}/metadata")
        resp.raise_for_status()
        return resp.json()
        
    def _verify_json_format_args(self, params: Union[str, dict], strict_json: bool = False) -> dict:
        """Verify the parameters of the function call"""
        if isinstance(params, str):
            try:
                if strict_json:
                    params_json: dict = json.loads(params)
                else:
                    params_json: dict = json_loads(params)
            except json.decoder.JSONDecodeError:
                raise ValueError('Parameters must be formatted as a valid JSON!')
        else:
            params_json: dict = params
        if isinstance(self.parameters, list):
            for param in self.parameters:
                if 'required' in param and param['required']:
                    if param['name'] not in params_json:
                        raise ValueError('Parameters %s is required!' % param['name'])
        elif isinstance(self.parameters, dict):
            import jsonschema
            jsonschema.validate(instance=params_json, schema=self.parameters)
        else:
            raise ValueError
        return params_json

class CrawlWebpageToolClient:
    def __init__(self, base_url: str, parameters: Union[List[dict], dict], call_timeout: float = 310.0, default_timeout: float = 60.0, max_concurrency: int = 8):
        self.base_url = base_url
        self.parameters = parameters
        self.call_client = httpx.AsyncClient(timeout=httpx.Timeout(call_timeout))
        self.default_client = httpx.AsyncClient(timeout=httpx.Timeout(default_timeout))
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def call(self, params: Union[str, Dict], keep_links: bool = False) -> str:
        payload = {"params": params, "keep_links": keep_links}
        async with self._semaphore:  # ✅ 在内部控制并发
            try:
                resp = await self.call_client.post(f"{self.base_url}/call", json=payload)
                resp.raise_for_status()  # 会抛出 HTTPStatusError 如果是 4xx/5xx

                try:
                    resp_json = resp.json()
                except Exception as e:
                    return f"[Client Error] (in call) Failed to parse JSON response: {str(e)}"

                if resp_json.get("error"):
                    return f"[Server Error] {resp_json['error']}"
                return resp_json["result"]

            except httpx.HTTPStatusError as e:
                return f"[Client Error] (in call) Bad HTTP status {e.response.status_code}: {e.response.text}"
            except httpx.TimeoutException:
                return "[Client Error] (in call) Request timed out"
            except httpx.RequestError as e:
                return f"[Client Error] (in call) Request failed: {str(e)}"
            except Exception as e:
                return f"[Client Error] (in call) Unexpected: {str(e)}"


    async def save_content(self, url: str, content: str) -> str:
        payload = {"url": url, "content": content}
        try:
            resp = await self.default_client.post(f"{self.base_url}/save_content", json=payload)
            resp.raise_for_status()
            return resp.json().get("status", "[Server Error] Unknown")
        except Exception as e:
            return f"[Client Error] (in save_content) {str(e)}"

    async def save_snippet(self, url: str, snippet: str) -> str:
        payload = {"url": url, "snippet": snippet}
        try:
            resp = await self.default_client.post(f"{self.base_url}/save_snippet", json=payload)
            resp.raise_for_status()
            return resp.json().get("status", "[Server Error] Unknown")
        except Exception as e:
            return f"[Client Error] (in save_snippet) {str(e)}"

    async def get_content(self, url: str) -> Optional[str]:
        try:
            resp = await self.default_client.get(f"{self.base_url}/get_content", params={"url": url})
            resp.raise_for_status()
            return resp.json().get("content", "[Server Error] No content returned")
        except Exception as e:
            return f"[Client Error] (in get_content) {str(e)}"

    async def get_snippet(self, url: str) -> Optional[str]:
        try:
            resp = await self.default_client.get(f"{self.base_url}/get_snippet", params={"url": url})
            resp.raise_for_status()
            return resp.json().get("snippet", "[Server Error] No snippet returned")
        except Exception as e:
            return f"[Client Error] (in get_snippet) {str(e)}"

    async def aclose(self):
        await self.call_client.aclose()
        await self.default_client.aclose()

    async def get_metadata(self) -> Dict[str, Any]:
        resp = await self.client.get(f"{self.base_url}/metadata")
        resp.raise_for_status()
        return resp.json()
    
    def _verify_json_format_args(self, params: Union[str, dict], strict_json: bool = False) -> dict:
        """Verify the parameters of the function call"""
        if isinstance(params, str):
            try:
                if strict_json:
                    params_json: dict = json.loads(params)
                else:
                    params_json: dict = json_loads(params)
            except json.decoder.JSONDecodeError:
                raise ValueError('Parameters must be formatted as a valid JSON!')
        else:
            params_json: dict = params
        if isinstance(self.parameters, list):
            for param in self.parameters:
                if 'required' in param and param['required']:
                    if param['name'] not in params_json:
                        raise ValueError('Parameters %s is required!' % param['name'])
        elif isinstance(self.parameters, dict):
            import jsonschema
            jsonschema.validate(instance=params_json, schema=self.parameters)
        else:
            raise ValueError
        return params_json
    

class CrawlWebpageToolClientV2:
    def __init__(
        self,
        base_url: Union[str, List[str]],
        parameters: Union[List[dict], dict],
        call_timeout: float = 60.0 * 5 + 10,  # 310s
        default_timeout: float = 60.0,
        max_concurrency: int = 64,           # 全局并发
        per_endpoint_concurrency: int = 1,   # 每端点并发
        retries: int = 3,
    ):
        self.parameters = parameters
        self.retries = retries

        # 支持 str / list，去重保序
        urls = [base_url] if isinstance(base_url, str) else list(base_url)
        seen, dedup = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u); dedup.append(u)
        self.urls: List[str] = dedup
        assert self.urls, "base_url 不能为空"

        # 为每个端点建 Client + 端点级信号量
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._ep_sema: Dict[str, asyncio.Semaphore] = {}
        for u in self.urls:
            limits = httpx.Limits(
                max_connections=per_endpoint_concurrency,
                max_keepalive_connections=per_endpoint_concurrency
            )
            self._clients[u] = httpx.AsyncClient(timeout=httpx.Timeout(call_timeout), limits=limits)
            self._ep_sema[u] = asyncio.Semaphore(per_endpoint_concurrency)

        # 轮询
        random.shuffle(self.urls)
        self._rr = cycle(self.urls)

        # 全局并发
        self._global_sema = asyncio.Semaphore(max_concurrency)

        # 非 call 接口的默认端点
        self._primary = self.urls[0]
        self.default_client = httpx.AsyncClient(timeout=httpx.Timeout(default_timeout))
        # self._last_used_url = None

    async def call(self, params: Union[str, Dict], keep_links: bool = False, return_endpoint: bool = False) -> str:
        payload = {"params": params, "keep_links": keep_links}
        async with self._global_sema:
            tried = set()
            for attempt in range(self.retries + 1):
                url = next(self._rr)
                # 避免一轮内重复挑同一个端点
                for _ in range(len(self.urls)):
                    if url in tried:
                        url = next(self._rr)
                    else:
                        break
                tried.add(url)

                ep = url
                client = self._clients[url]
                ep_sema = self._ep_sema[url]
                try:
                    async with ep_sema:
                        resp = await client.post(f"{url}/call", json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        if data.get("error"):
                            if return_endpoint:
                                return {"result": f"[Server Error] {data['error']}", "endpoint": ep}
                            return f"[Server Error] {data['error']}"
                        if return_endpoint:
                            return {"result": data["result"], "endpoint": ep}
                        return data["result"]
                except httpx.HTTPStatusError as e:
                    # 429/5xx 可重试
                    if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < self.retries:
                        continue
                    return f"[Client Error] (in call) Bad HTTP status {e.response.status_code}: {e.response.text}"
                except (httpx.TimeoutException, httpx.RequestError) as e:
                    if attempt < self.retries:
                        continue
                    return f"[Client Error] (in call) {type(e).__name__}: {str(e)}"
                except Exception as e:
                    return f"[Client Error] (in call) Unexpected: {str(e)}"

    # 下面这些辅助接口，默认走主端点（也可以改成随机选一个端点）
    async def save_content(self, url: str, content: str) -> str:
        payload = {"url": url, "content": content}
        try:
            r = await self.default_client.post(f"{self._primary}/save_content", json=payload)
            r.raise_for_status()
            return r.json().get("status", "[Server Error] Unknown")
        except Exception as e:
            return f"[Client Error] (in save_content) {str(e)}"

    async def save_snippet(self, url: str, snippet: str) -> str:
        payload = {"url": url, "snippet": snippet}
        try:
            r = await self.default_client.post(f"{self._primary}/save_snippet", json=payload)
            r.raise_for_status()
            return r.json().get("status", "[Server Error] Unknown")
        except Exception as e:
            return f"[Client Error] (in save_snippet) {str(e)}"

    async def get_content(self, url: str) -> Optional[str]:
        try:
            r = await self.default_client.get(f"{self._primary}/get_content", params={"url": url})
            r.raise_for_status()
            return r.json().get("content", "[Server Error] No content returned")
        except Exception as e:
            return f"[Client Error] (in get_content) {str(e)}"

    async def get_snippet(self, url: str) -> Optional[str]:
        try:
            r = await self.default_client.get(f"{self._primary}/get_snippet", params={"url": url})
            r.raise_for_status()
            return r.json().get("snippet", "[Server Error] No snippet returned")
        except Exception as e:
            return f"[Client Error] (in get_snippet) {str(e)}"

    async def aclose(self):
        await asyncio.gather(*[c.aclose() for c in self._clients.values()], return_exceptions=True)
        await self.default_client.aclose()

    #（可选）如果你需要这个接口
    async def get_metadata(self) -> Dict[str, Any]:
        r = await self.default_client.get(f"{self._primary}/metadata")
        r.raise_for_status()
        return r.json()

    #（可选）形参校验器（按你已有逻辑保留）
    def _verify_json_format_args(self, params: Union[str, dict], strict_json: bool = False) -> dict:
        if isinstance(params, str):
            try:
                if strict_json:
                    params_json: dict = json.loads(params)
                else:
                    params_json: dict = json_loads(params)
            except json.decoder.JSONDecodeError:
                raise ValueError('Parameters must be formatted as a valid JSON!')
        else:
            params_json: dict = params
        if isinstance(self.parameters, list):
            for param in self.parameters:
                if param.get('required') and param['name'] not in params_json:
                    raise ValueError('Parameters %s is required!' % param['name'])
        elif isinstance(self.parameters, dict):
            import jsonschema
            jsonschema.validate(instance=params_json, schema=self.parameters)
        else:
            raise ValueError
        return params_json


    


if __name__ == "__main__":
    import ray
    import asyncio
    import httpx

    async def main():
        ray.init()
        # 启动 WebSearchToolServer
        web_search_actor = WebSearchToolServer.remote(
            api_key="005fba2b8daa23f10be87a7d76a5bd37c99627c7",  # ✅ 记得替换成你的 Serper API Key
            cache_file="/workspace/cache/serper_search_cache.json"
        )
        web_search_addr = await web_search_actor.get_server_address.remote()
        print(f"✅ WebSearchToolServer started at {web_search_addr}")

        # 启动 CrawlWebpageToolServer
        crawl_actor = CrawlWebpageToolServer.remote(
            semaphore_limit=32,
            cache_file="/workspace/cache/crawl4ai_url_cache.json",
        )
        crawl_addr = await crawl_actor.get_server_address.remote()
        print(f"✅ CrawlWebpageToolServer started at {crawl_addr}")

        search_params = {"query": "What is the capital of France?"}
        crawl_params = {"url": "https://en.wikipedia.org/wiki/Paris"}
        # test_url = {"params": {"url": "https://en.wikipedia.org/wiki/Paris"}, "keep_links": False}

        search_client = WebSearchToolClient(base_url=f"http://{web_search_addr}")
        crawl_client = CrawlWebpageToolClient(base_url=f"http://{crawl_addr}") 

        # call search
        print("\n🔍 Sending search query...")
        resp = await search_client.call(search_params)
        print("🔁 WebSearch Response:")
        print(resp)

        # call crawl
        print("\n🌐 Sending crawl request...")
        resp = await crawl_client.call(crawl_params, keep_links=False)
        print("🔁 CrawlWebpage Response (prefix):")
        print(resp[:300] + "...")

        # 保存片段
        snippet_data = {
            "url": "https://en.wikipedia.org/wiki/Paris",
            "snippet": "Paris is the capital city of France, known for its art, fashion, and culture."
        }
        print("\n💾 Saving snippet...")
        save_resp = await crawl_client.save_snippet(snippet_data["url"], snippet_data["snippet"])
        print("🔁 Save Snippet Response:")
        print(save_resp)

        # 获取片段
        print("\n📄 Retrieving snippet...")
        get_snippet_resp = await crawl_client.get_snippet(snippet_data["url"])
        print("🔁 Get Snippet Response:")
        print(get_snippet_resp)

        # 获取内容
        print("\n📄 Retrieving content...")
        get_content_resp = await crawl_client.get_content(snippet_data["url"])
        print("🔁 Get Content Response:")
        print(get_content_resp[:300] + "...")

        # 保存和获取缓存
        print("\n💾 Saving search cache...")
        save_cache_resp = await search_client.save_cache()
        print("🔁 Save Cache Response:")
        print(save_cache_resp)

        # 关闭客户端
        await search_client.aclose()
        await crawl_client.aclose()


    asyncio.run(main())