import requests
import time
import random
import logging
import asyncio
import socket
import json
import uuid

from typing import List, Dict, Optional, Any
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
from fastapi import Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import ray

from tests.workers.rollout.my_utils import json_loads

# --- 日志配置 ---
def get_logger(name: str = __name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger("individual_tools")
logging.getLogger("crawl4ai").setLevel(logging.WARNING) # 减少 Crawl4AI 内部日志


def extract_relevant_info_serper(search_results):
    """
    Extract relevant information from Google Serper search results.

    Args:
        search_results (dict): JSON response from the Google Serper API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    if 'organic' in search_results:
        for i, result in enumerate(search_results['organic']):
            # Try to extract domain for site_name, or leave empty
            site_name = ''
            try:
                site_name = urlparse(result.get('link', '')).netloc
            except Exception:
                pass

            info = {
                'id': i + 1,
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'site_name': site_name, # Serper doesn't directly provide siteName, try to parse from URL
                'date': result.get('date', ''), # Serper might not always provide date
                'snippet': result.get('snippet', ''),
            }
            useful_info.append(info)
    return useful_info

def format_search_results(relevant_info: List[Dict]) -> str:
    """Format search results into a markdown"""
    formatted_result = "\n\n".join([
    f"[{r['id']}] [{r['title']}]({r['url']}) {r['date']}\n{r['snippet']}\n\n{r['page_info'] if 'page_info' in r else ''}"
    for r in relevant_info
    ])
    return formatted_result

def is_jsonl(path: str):
    return path.endswith('.jsonl')

def is_json(path: str):
    return path.endswith('.json')

def generate_tmp_file(cache_file: str) -> str:
    base = os.path.basename(cache_file)
    suffix = f"{os.getpid()}.{uuid.uuid4().hex}.tmp.jsonl" if is_jsonl(cache_file) else f"{os.getpid()}.{uuid.uuid4().hex}.tmp.json"
    return os.path.join(os.path.dirname(cache_file), f"{base}.{suffix}")


class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: Union[List[dict], dict] = []


    def __init__(self, *args, **kwargs):
        pass 

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

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


# --- Google Search Tool Definition ---
class WebSearchTool(BaseTool):
    name = 'web_search'
    description = 'Search for information from the internet.'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
            }
        },
        'required': ['query'],
    }
    is_async = True
    has_cache = True
    def __init__(self, api_key: str, cache_file: Optional[str] = None):
        self.api_key = api_key
        self._init_cache(cache_file)
        self.cache_file = cache_file

    def _init_cache(self, cache_file: Optional[str] = None):
        if not cache_file or not os.path.exists(cache_file):
            self.cache = {}
            print("Cache initialized with 0 entries.")
            return

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                if is_jsonl(cache_file):
                    self.cache = {}
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            key, val = obj.get("key"), obj.get("value")
                            if key is not None:
                                self.cache[key] = val
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid line in JSONL cache: {line}")
                elif is_json(cache_file):
                    self.cache = json.load(f)
                else:
                    logger.warning(f"Unknown cache format for {cache_file}. Starting fresh.")
                    self.cache = {}
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.warning(f"Failed to load cache from {cache_file}: {e}. Starting fresh.")
            self.cache = {}

        print("Cache initialized with", len(self.cache), "entries.")

    def save_cache(self, cache_file: Optional[str] = None):
        cache_file = self.cache_file if cache_file is None else cache_file
        if not cache_file:
            return

        tmp_file = generate_tmp_file(cache_file)

        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                if is_jsonl(cache_file):
                    for key, value in self.cache.items():
                        json.dump({"key": key, "value": value}, f, ensure_ascii=False)
                        f.write("\n")
                elif is_json(cache_file):
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported cache format: {cache_file}")
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_file, cache_file)  # atomic: one process "wins"
            logger.info(f"✅ Cache safely saved to {cache_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save cache to {cache_file}: {e}")
        finally:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass  # it's ok if cleanup fails


    async def call(self, params: Union[str, dict], timeout: int = 20, **kwargs) -> List[Dict]:
        params = self._verify_json_format_args(params)
        query = params["query"]

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers_serper = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        max_retries = 5
        retry_count = 0
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        if query in self.cache:
            cache_result = self.cache[query]
            if isinstance(cache_result, dict) and 'organic' in cache_result:
                print(f"✅ [Cache Hit] Query: {query}")
                return extract_relevant_info_serper(cache_result)
        

        async with aiohttp.ClientSession() as session:
            while retry_count < max_retries:
                try:
                    async with session.post(url, headers=headers_serper, data=payload, timeout=client_timeout) as response:
                        response.raise_for_status()
                        result = await response.json()

                        assert "organic" in result
                        self.cache[query] = result
                        print(f"❌ [No Cache Hit] Fetched and cached result for query: {query}")
                        return extract_relevant_info_serper(result)
                except asyncio.TimeoutError:
                    retry_count += 1
                    print(f"[Timeout] Retry {retry_count}/{max_retries} for query: {query}")
                except aiohttp.ClientError as e:
                    retry_count += 1
                    print(f"[HTTPError] Retry {retry_count}/{max_retries} for query: {query}. Error: {e}")

                await asyncio.sleep(1)  # Optional delay before retry

        # Still failed after retries — let the exception propagate
        raise RuntimeError(f"Google Serper API failed after {max_retries} retries for query: {query}")
    
    def metadata(self):
        return {
            "name": self.name,
            "name_for_human": getattr(self, "name_for_human", self.name),
            "name_for_model": getattr(self, "name_for_model", self.name),
            "description": self.description,
            "parameters": self.parameters,
            "args_format": getattr(self, "args_format", ""),
        }
    

# --- Crawl Webpage Tool Definition ---
class CrawlWebpageTool(BaseTool):
    is_async = True
    has_cache = True
    name = "crawl_webpage"
    description = "A tool for fetching content of a webpage based on its URL"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The url of a webpage to fetch"
            }
        },
        "required": ["url"]
        }

    def __init__(self, 
                 semaphore: asyncio.Semaphore,
                 cache_file: Optional[str] = None,
                 snippet_cache_file: Optional[str] = None,
                 fetch_timeout: int = 300):
       self._semaphore = semaphore
       self._fetch_timeout = fetch_timeout  # seconds
       self._init_cache(cache_file, snippet_cache_file)
       self.cache_file = cache_file
       self.snippet_cache_file = snippet_cache_file
       self.snippet_cache
    
    def _init_cache(self, cache_file: Optional[str] = None, snippet_cache_file: Optional[str] = None):
        if cache_file:
            with open(cache_file, 'r', encoding='utf-8') as f:
                try:
                    self.cache = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Cache file {cache_file} is empty or invalid. Starting fresh.")
                    self.cache = {}
        else:
            self.cache = {}

        if snippet_cache_file:
            with open(snippet_cache_file, 'r', encoding='utf-8') as f:
                try:
                    self.snippet_cache = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Snippet cache file {snippet_cache_file} is empty or invalid. Starting fresh.")
                    self.snippet_cache = {}
        else:
            self.snippet_cache = {}

        # Log cache initialization
        print("Cache initialized with", len(self.cache), "entries and snippet cache with", len(self.snippet_cache), "entries.")

    def save_cache(self, cache_file: Optional[str] = None, snippet_cache_file: Optional[str] = None):
        cache_file = self.cache_file if cache_file is None else cache_file
        snippet_cache_file = self.snippet_cache_file if snippet_cache_file is None else snippet_cache_file
        if cache_file:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Cache saved to {cache_file}")
        if snippet_cache_file:
            with open(snippet_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.snippet_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Snippet cache saved to {snippet_cache_file}")

    def save_content(self, url: str, content):
        self.cache[url] = content

    def save_snippet(self, url: str, snippet: str):
        self.snippet_cache[url] = snippet

    def get_content(self, url: str) -> Optional[str]:
        return self.cache.get(url, None)

    def get_snippet(self, url: str) -> Optional[str]:
        return self.snippet_cache.get(url, None)

    async def call(self, 
                   params: Union[str, dict],
                   keep_links: bool = False) -> str:
        params = self._verify_json_format_args(params)
        url = params["url"]

        # 1. 检查缓存（可选）
        cached = self.get_content(url)
        if cached:
            logger.info(f"[Cache Hit] URL: {url}")
            return cached
    
        async with self._semaphore:
            crawler_run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,          
            excluded_tags=["nav", "footer", "aside", "script", "style", "form",
                            "noscript", "iframe", "svg", "video", "audio",
                            "canvas", "object", "embed"], # 去掉常见无用区域
            remove_overlay_elements=True,  
            exclude_social_media_links=True,
            exclude_external_images=True,           # 去弹窗/遮罩
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                min_word_threshold=10,
                ),
                options={
                    "ignore_links": not keep_links,
                }
            )) 

            # 2. Retry最多尝试2次
            for attempt in range(2):
                try:
                    logger.info(f"[Fetch Try {attempt+1}] URL: {url}")
                    async with AsyncWebCrawler() as crawler:
                        result = await asyncio.wait_for(
                            crawler.arun(url, config=crawler_run_config),
                            timeout=min(self._fetch_timeout, 60)  # 最大不超过 60 秒
                        )

                        if result and result.markdown:
                            markdown_content = result.markdown.fit_markdown if result.markdown.fit_markdown else ""
                            markdown_content = markdown_content.replace('---', '-').replace('===', '=').replace('   ', ' ').replace('   ', ' ')
                            self.save_content(url, markdown_content)
                            
                            return markdown_content

                        elif result and result.error:
                            return f"Crawl4AI Error: {result.error}"
                        else:
                            return "Crawl4AI Error: No content available."

                except asyncio.TimeoutError:
                    logger.warning(f"[Timeout] Try {attempt+1} - Timeout after {min(self._fetch_timeout, 60)}s for URL: {url}")
                except Exception as e:
                    logger.warning(f"[Error] Try {attempt+1} - Unexpected error for URL {url}: {e}")

            # 如果 retry 全失败
            return f"Crawl4AI Error: failed after 2 attempts for {url}"
            
    def metadata(self):
        return {
            "name": self.name,
            "name_for_human": getattr(self, "name_for_human", self.name),
            "name_for_model": getattr(self, "name_for_model", self.name),
            "description": self.description,
            "parameters": self.parameters,
            "args_format": getattr(self, "args_format", ""),
        }
            

class CrawlWebpageToolV2(BaseTool):
    is_async = True
    has_cache = True
    name = "crawl_webpage"
    description = "A tool for fetching content of a webpage based on its URL"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The url of a webpage to fetch"
            }
        },
        "required": ["url"]
    }

    def __init__(
        self,
        semaphore: asyncio.Semaphore,
        cache_file: Optional[str] = None,
        snippet_cache_file: Optional[str] = None,
        fetch_timeout: int = 300
    ):
        self._semaphore = semaphore
        self._fetch_timeout = fetch_timeout  # seconds

        # 缓存初始化（与 v1 相同）
        self._init_cache(cache_file, snippet_cache_file)
        self.cache_file = cache_file
        self.snippet_cache_file = snippet_cache_file
        self.snippet_cache

        # ==== v2: 浏览器单实例（懒加载）====
        self._crawler = None
        self._crawler_lock = asyncio.Lock()
        self._browser_args = [
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--no-zygote",
            "--disable-software-rasterizer",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-renderer-backgrounding",
            "--disable-features=Translate,BackForwardCache",
            "--renderer-process-limit=1",
            # "--single-process",  # 可选：如不稳定请注释掉
        ]

        # ==== v2: 预先构建 run config，避免每次 new ====
        self._cfg_with_links = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            excluded_tags=[
                "nav","footer","aside","script","style","form","noscript",
                "iframe","svg","video","audio","canvas","object","embed"
            ],
            remove_overlay_elements=True,
            exclude_social_media_links=True,
            exclude_external_images=True,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(min_word_threshold=10),
                options={"ignore_links": False},
            ),
        )
        self._cfg_no_links = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            excluded_tags=[
                "nav","footer","aside","script","style","form","noscript",
                "iframe","svg","video","audio","canvas","object","embed"
            ],
            remove_overlay_elements=True,
            exclude_social_media_links=True,
            exclude_external_images=True,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(min_word_threshold=10),
                options={"ignore_links": True},
            ),
        )

    # -------- v1 保留：缓存相关 --------
    def _init_cache(self, cache_file: Optional[str] = None, snippet_cache_file: Optional[str] = None):
        if cache_file:
            with open(cache_file, 'r', encoding='utf-8') as f:
                try:
                    self.cache = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Cache file {cache_file} is empty or invalid. Starting fresh.")
                    self.cache = {}
        else:
            self.cache = {}

        if snippet_cache_file:
            with open(snippet_cache_file, 'r', encoding='utf-8') as f:
                try:
                    self.snippet_cache = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Snippet cache file {snippet_cache_file} is empty or invalid. Starting fresh.")
                    self.snippet_cache = {}
        else:
            self.snippet_cache = {}

        print("Cache initialized with", len(self.cache), "entries and snippet cache with", len(self.snippet_cache), "entries.")

    def save_cache(self, cache_file: Optional[str] = None, snippet_cache_file: Optional[str] = None):
        cache_file = self.cache_file if cache_file is None else cache_file
        snippet_cache_file = self.snippet_cache_file if snippet_cache_file is None else snippet_cache_file
        if cache_file:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Cache saved to {cache_file}")
        if snippet_cache_file:
            with open(snippet_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.snippet_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Snippet cache saved to {snippet_cache_file}")

    def save_content(self, url: str, content):
        self.cache[url] = content

    def save_snippet(self, url: str, snippet: str):
        self.snippet_cache[url] = snippet

    def get_content(self, url: str) -> Optional[str]:
        return self.cache.get(url, None)

    def get_snippet(self, url: str) -> Optional[str]:
        return self.snippet_cache.get(url, None)

    # -------- v2 新增：懒加载/复用 crawler + 关闭 --------
    async def _get_crawler(self):
        if self._crawler is None:
            async with self._crawler_lock:
                if self._crawler is None:
                    # 只创建一次，整个 server 生命周期复用
                    self._crawler = await AsyncWebCrawler(
                        browser_type="chromium",
                        headless=True,
                        browser_args=self._browser_args,
                    ).__aenter__()
        return self._crawler

    async def aclose(self):
        """优雅关闭浏览器实例，供服务停机时调用"""
        try:
            if self._crawler is not None:
                await self._crawler.__aexit__(None, None, None)
        finally:
            self._crawler = None

    # -------- v2：call 使用复用 crawler --------
    async def call(self, params: Union[str, dict], keep_links: bool = False) -> str:
        params = self._verify_json_format_args(params)
        url = params["url"]

        # 1) 缓存
        cached = self.get_content(url)
        if cached:
            logger.info(f"[Cache Hit] URL: {url}")
            return cached

        # 2) 并发控制
        async with self._semaphore:
            cfg = self._cfg_with_links if keep_links else self._cfg_no_links

            # 3) Retry 最多 2 次
            for attempt in range(2):
                try:
                    logger.info(f"[Fetch Try {attempt+1}] URL: {url}")
                    crawler = await self._get_crawler()

                    # 注意：单次抓取的“操作级超时”建议 ≤20s，尾延迟更稳
                    result = await asyncio.wait_for(
                        crawler.arun(url, config=cfg),
                        timeout=min(self._fetch_timeout, 20),
                    )

                    if result and result.markdown:
                        markdown_content = result.markdown.fit_markdown or ""
                        markdown_content = (
                            markdown_content
                            .replace('---', '-')
                            .replace('===', '=')
                            .replace('   ', ' ')
                        )
                        self.save_content(url, markdown_content)
                        return markdown_content

                    elif result and result.error:
                        return f"Crawl4AI Error: {result.error}"
                    else:
                        return "Crawl4AI Error: No content available."

                except asyncio.TimeoutError:
                    logger.warning(f"[Timeout] Try {attempt+1} - Timeout after {min(self._fetch_timeout, 20)}s for URL: {url}")
                except Exception as e:
                    logger.warning(f"[Error] Try {attempt+1} - Unexpected error for URL {url}: {e}")

            return f"Crawl4AI Error: failed after 2 attempts for {url}"

    def metadata(self):
        return {
            "name": self.name,
            "name_for_human": getattr(self, "name_for_human", self.name),
            "name_for_model": getattr(self, "name_for_model", self.name),
            "description": self.description,
            "parameters": self.parameters,
            "args_format": getattr(self, "args_format", ""),
        }
   

# --- Main Execution Block for Debugging ---
async def main():
    """
    Main function to run the CrawlWebpageTool for testing.
    """
    logger.debug("Starting CrawlWebpageTool debug session.")

    # Instantiate your tool
    crawler_tool = CrawlWebpageTool()
    cache_file = "./tmp/crawl4ai_cache.json"
    url_to_content = {}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Define URLs to test
    test_urls = ['https://pubchem.ncbi.nlm.nih.gov/compound/Dimethyl-Fumarate',
                'https://en.wikipedia.org/wiki/Dimethyl_fumarate',
                'https://pubmed.ncbi.nlm.nih.gov/32672401/',
                'https://www.selleckchem.com/products/dimethyl-Fumarate.html',
                'https://go.drugbank.com/drugs/DB08908',
                'https://www.scbt.com/p/dimethyl-fumarate-624-49-7',
                'https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.15485',
                'https://www.sigmaaldrich.com/US/en/product/aldrich/242926',
                'https://www.chemspider.com/Chemical-Structure.553171.html',
                'https://journals.iucr.org/paper?oj3025',
                'https://www.reddit.com/r/BillyJoel/comments/1101jfe/whats_the_true_meaning_of_shes_always_a_woman/',
                'https://americansongwriter.com/the-empowering-meaning-behind-billy-joels-shes-always-a-woman/',
                'https://www.reddit.com/r/BillyJoel/comments/9nb1ae/weekly_song_thread_shes_always_a_woman/',
                'https://medium.com/@gigiries/is-the-message-behind-billy-joels-she-s-always-a-woman-one-of-empowerment-or-limitation-7ab2e549c413',
                'https://en.wikipedia.org/wiki/She%27s_Always_a_Woman',
                'https://www.onefinalserenade.com/shes-always-a-woman.html',
                'https://genius.com/Billy-joel-shes-always-a-woman-lyrics',
                'https://www.thegearpage.net/board/index.php?threads/lyrics-discussion-shes-always-a-woman-to-me.2068813/',
                'https://songmeanings.com/songs/view/1491/',
                'https://open.spotify.com/track/5RgFlk1fcClZd0Y4SGYhqH',
                "https://groups.cs.umass.edu/zamani/"]

    for url in test_urls:
        print(f"\n--- Testing URL: {url} ---")
        try:
            # Call the async call method.
            # asyncio.run() is used here because main() is typically the entry point
            # in a script and needs to run an async function.
            # In a FastAPI app, you wouldn't use asyncio.run() directly inside a route.
            content = await crawler_tool.call(url, timeout=60, enable_javascript=True)
            
            print(f"Result for {url}:")
            if content.startswith("Crawl4AI Error:") or content.startswith("Crawl4AI Unexpected Error:"):
                print(f"  Error: {content}")
            else:
                # Print a preview of the content
                print(f"  Content (first 500 chars):\n{content[:500]}...")
        except Exception as e:
            logger.critical(f"Unhandled exception during test for {url}: {e}", exc_info=True)
            content = f"Crawl4AI Error: {e}"
        
        url_to_content[url] = content

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(url_to_content, f, indent=4, ensure_ascii=False)

    logger.debug("CrawlWebpageTool debug session completed.")

CandidateTools = {
    'web_search': WebSearchTool,
    'crawl_webpage': CrawlWebpageTool,
}


if __name__ == "__main__":
    # Ensure this script is run as an async program.
    # asyncio.run() handles the event loop.
    asyncio.run(main())