import os
import logging
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import asyncio
from dotenv import load_dotenv

# 缓存相关
from cachetools import LRUCache, cached # LRU缓存
import functools # 用于 @functools.wraps

# 从 tools.py 导入工具类
from tools import GoogleSearchTool, CrawlWebpageTool

# 加载环境变量
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_app")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(
    title="Modular Search & Crawl API",
    description="Separate API endpoints for Google Search and Webpage Content Crawling with caching.",
    version="1.0.0"
)

google_search_tool = GoogleSearchTool()
crawl_webpage_tool = CrawlWebpageTool()

google_search_cache = LRUCache(maxsize=100)

webpage_content_cache = LRUCache(maxsize=500)


class GoogleSearchRequest(BaseModel):
    query: str
    topk: int = 3

class CrawlWebpageRequest(BaseModel):
    url: str
    world_limit: Optional[int] = 1000

@app.post("/google_search", response_model=List[Dict[str, Any]])
@cached(cache=google_search_cache, key=lambda query_req: (query_req.query, query_req.topk))
async def google_search_endppoint(request: GoogleSearchRequest):
    """
    Performs a Google web search and returns top-k results (title, link, snippet).
    Results are cached based on query and topk.
    """
    logger.debug(f"API: Received Google Search request for '{request.query}', topk={request.topk}.")

    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        logger.error("Google API Key or CSE ID not configured.")
        raise HTTPException(status_code=500, detail="Server configuration error: Google API Key or CSE ID missing.")

    results = google_search_tool.execute(
        query=request.query,
        api_key=api_key,
        cse_id=cse_id,
        topk=request.topk
    )
    return results

@app.post("/crawl_webpage", response_model=str)
async def crawl_webpage_endpoint(request: CrawlWebpageRequest):
    """
    Fetches the full content of a webpage using Crawl4AI.
    Content is cached based on the URL.
    """
    logger.debug(f"API: Received Crawl Webpage request for '{request.url}', timeout={request.timeout}.")

    # 检查缓存
    if request.url in webpage_content_cache:
        logger.debug(f"API: Cache hit for {request.url}")
        return webpage_content_cache[request.url]

    # 调用 Crawl Webpage Tool (异步操作)
    content = await crawl_webpage_tool.execute(
        url=request.url,
        timeout=request.timeout,
        enable_javascript=request.enable_javascript
    )

    # 如果抓取成功，则缓存结果
    if not str(content).startswith("Crawl4AI Error:") and not str(content).startswith("Crawl4AI Unexpected Error:"):
        webpage_content_cache[request.url] = content
        logger.debug(f"API: Cached content for {request.url}.")
    
    return content

# --- 应用启动和关闭事件 ---
@app.on_event("startup")
async def startup_event():
    logger.debug("FastAPI application started. Tools initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.debug("FastAPI application shutting down.")