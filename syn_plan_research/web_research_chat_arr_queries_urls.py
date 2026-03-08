# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from openai import AsyncOpenAI
import asyncio
from argparse import Namespace
from transformers import AutoTokenizer
import re

from data_types import GenRecord, Message
from tools import extract_relevant_info_serper, format_search_results, WebSearchTool, CrawlWebpageTool
from utils import extract_snippet_with_context
from tools import BaseTool
from prompts import PROMPT_REACT
from web_research_chat import AsyncWebResearchChatAgent


TOOL_DESC = (
    "<tool>\n"
    "Tool Name: {name_for_model}\n"
    "Description: {description_for_model}\n"
    "Usage: This tool lets the agent interact with the {name_for_human} API.\n"
    "Parameters (JSON Schema): {parameters}\n"
    "{args_format}\n"
    "</tool>"
)



ASSISTANT = "assistant"
TOOL = "tool"
USER = "user"
SYSTEM = "system"


def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    try:
        pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
        # Run pattern matching with timeout
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            return matches[0][::-1].strip()
        return None
    except Exception as e:
        print(f"---Error:---\n{str(e)}")
        print(f"-------------------")
        return None

async def generate_response(
    client: AsyncOpenAI,
    tokenizer:  AutoTokenizer,
    messages: List[Message],
    semaphore: asyncio.Semaphore,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.05,
    top_k: int = 20,
    model_name: str = "QwQ-32B",
    retry_limit: int = 3,
    enable_thinking: bool = True,
) -> str:
    """Generate a single response with retry logic"""
    # print("[DEBUG]: enable_thinking:", enable_thinking, "top_k:", top_k, "repetition_penalty:", repetition_penalty, "top_p:", top_p, "temperature:", temperature, "max_tokens:", max_tokens)
    formatted_prompt = tokenizer.apply_chat_template(
        [msg.to_dict() for msg in messages], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                response = await client.completions.create(
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=[tokenizer.eos_token],
                    extra_body={
                        'top_k': top_k,
                        'repetition_penalty': repetition_penalty,
                    },
                    timeout=3600,
                )
                return formatted_prompt, response.choices[0].text
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            # print(prompt)
            if "maximum context length" in str(e).lower():
                # If length exceeds limit, reduce max_tokens by half
                max_tokens = max_tokens // 2
                print(f"Reducing max_tokens to {max_tokens}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return "", ""
            await asyncio.sleep(1 * (attempt + 1))
    return "", ""



class AsyncWebResearchChatAgentArrQueriesUrls(AsyncWebResearchChatAgent):
    """This agent use ReAct format to call tools"""

    def __init__(self,
                 client: AsyncOpenAI,
                 tokenizer: AutoTokenizer,
                 tool_maps: Dict[str, BaseTool],
                 model_name: str = "QwQ-32B",
                 name: Optional[str] = "web-research-agent",
                 description: Optional[str] = None,
                 concurrent_limit: int = 32,
                 max_llm_call: int = 10,
                 gen_config: Optional[Dict] = None,
                 web_result_config: Optional[Dict] = None,
                 max_tokens: int = 32768,
                 enable_thinking: bool = True,
                 prompt_react: Optional[str] = None,
                 **kwargs):

        super().__init__(client, tokenizer, tool_maps, model_name, name, description,
                         concurrent_limit, max_llm_call, gen_config, web_result_config,
                         max_tokens, enable_thinking, prompt_react, **kwargs)


    async def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
       
        tool = self.function_map[tool_name]
        assert tool.is_async, f'Tool {tool_name} is not async, but called in async mode.' 
        # async with self.tool_semaphore:
        try:
            tool_result = await tool.call(tool_args, **kwargs)

            if tool_name == "web_search":
                if self.web_result_config["snippet_only"]:
                    observation = format_search_results(tool_result)
                    for result in tool_result:
                        self.function_map["crawl_webpage"].save_snippet(result['url'], result["snippet"])
                    return observation
                else:
                    crawler_tool = self.function_map["crawl_webpage"]
                    urls = [{"url": result['url']} for result in tool_result]
                    tasks = [crawler_tool.call(url) for url in urls]
                    web_contents = await asyncio.gather(*tasks)

                    for content, result in zip(web_contents, tool_result):
                        has_error = "Crawl4AI Error" in content
                        if not has_error:
                            result["page_info"] = self._postprocess_web_content(
                                result, content, self.web_result_config["content_extract_method"], context_char=self.web_result_config["web_search_context_chars"])
                            self.function_map["crawl_webpage"].save_content(result['url'], content)
                            self.function_map["crawl_webpage"].save_snippet(result['url'], result["snippet"])
                        else:
                            result["page_info"] = ""

                    observation = format_search_results(tool_result)
                    return observation
            elif tool_name == "crawl_webpage":
                urls = tool_args["url"]
                try:
                    json_tool_args = tool._verify_json_format_args(tool_args)
                    # print("[DEBUG] crawl_webpage tool_args:", json_tool_args)
                    snippet = self.function_map["crawl_webpage"].get_snippet(json_tool_args["url"])
                    # print("[DEBUG] crawl_webpage snippet:", snippet[:100] + "...")
                except Exception as e:
                    snippet = None 
                if snippet:
                    if self.web_result_config["snippet_only"]:
                        return self._postprocess_web_content({"snippet": snippet}, tool_result, self.web_result_config["content_extract_method"], context_char=self.web_result_config["context_chars"])
                    else:
                        print("[DEBUG] content_extract_method: ", self.web_result_config["content_extract_method"], self.web_result_config["context_chars"])
                        return self._postprocess_web_content({"snippet": snippet}, tool_result, self.web_result_config["content_extract_method"], context_char=self.web_result_config["context_chars"])
                else:
                    return tool_result[:4000]

        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n'
            print(error_message)
            return error_message
            