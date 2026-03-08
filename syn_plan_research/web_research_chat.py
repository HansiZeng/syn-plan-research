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

DEFAULT_CLAUDE_SYSTEM_PROMPT = (
    f"You are a helpful AI assistant."
    " Do not mention your name or creators. Be concise for simple questions, and thorough for complex ones."
)


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

def postprocess_claude_output(output: str) -> str:
    if output.endswith("</tool_call>") or output.endswith("</answer>"):
        return output

    if "</think>\n\n<tool_call>" in output:
        return output.rstrip() + "\n</tool_call>"

    if "</think>\n\n<answer>" in output:
        return output.strip() + "\n</answer>"

    # fallback: invalid format
    return output.rstrip() + "\n</wrong_format>"

async def generate_response_claude(
    messages: List[Message],
    semaphore: asyncio.Semaphore,
    model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region: str = "us-west-2",
    system_prompt: str = DEFAULT_CLAUDE_SYSTEM_PROMPT,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    retry_limit: int = 10,
    tool_thinking_prompt = None,
    stop_words: List[str] = ["</tool_call>", "</answer>"]
) -> Tuple[str, str]:
    """
    Asynchronously call Claude (Bedrock) for a single response.

    Returns:
        Tuple[prompt_str, response_str]
    """
    from botocore.config import Config
    import botocore.session
    from aiobotocore.session import get_session
    
    session = get_session()
    legacy_session = botocore.session.get_session()
    credentials = legacy_session.get_credentials().get_frozen_credentials()

    messages = [msg.to_dict() for msg in messages]
    if tool_thinking_prompt is not None:
        messages.append({
            "role": ASSISTANT,
            "content": f"<think>\n{tool_thinking_prompt}"
        })

    body = json.dumps({
        "messages": messages,
        "system": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "anthropic_version": "bedrock-2023-05-31",
        "stop_sequences": stop_words,
    })

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                async with session.create_client(
                    "bedrock-runtime",
                    region_name=region,
                    aws_access_key_id=credentials.access_key,
                    aws_secret_access_key=credentials.secret_key,
                    aws_session_token=credentials.token,
                    config=Config(region_name=region),
                ) as client:
                    response = await client.invoke_model(
                        modelId=model_id,
                        body=body,
                        contentType="application/json",
                        accept="application/json",
                    )
                    response_body = await response["body"].read()
                    parsed = json.loads(response_body)
                    if "content" in parsed:
                        response_text = parsed["content"][0]["text"]
                        response_text = postprocess_claude_output(response_text)

                        if tool_thinking_prompt is None:
                            return messages[0]["content"], response_text
                        else:
                            return messages[0]["content"], f"<think>\n{tool_thinking_prompt}" + response_text
                    else:
                        return messages[0]["content"], parsed.get("completion", "")
        except Exception as e:
            print(f"[Claude Error] attempt {attempt+1}/{retry_limit}: {e}")
            if "maximum context length" in str(e).lower() and max_tokens > 256:
                max_tokens = max_tokens // 2
                print(f"→ Reducing max_tokens to {max_tokens}")
            if attempt == retry_limit - 1:
                return "", ""
            await asyncio.sleep(2 * (attempt + 1))

    return "", ""

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
    tool_thinking_prompt = None,
) -> str:
    """Generate a single response with retry logic"""
    # print("[DEBUG]: enable_thinking:", enable_thinking, "top_k:", top_k, "repetition_penalty:", repetition_penalty, "top_p:", top_p, "temperature:", temperature, "max_tokens:", max_tokens)
    if tool_thinking_prompt is None:
        formatted_prompt = tokenizer.apply_chat_template(
            [msg.to_dict() for msg in messages], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    else:
        formatted_prompt = tokenizer.apply_chat_template(
            [msg.to_dict() for msg in messages], tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking)
        formatted_prompt += '<|im_start|>assistant\n<think>\n' + tool_thinking_prompt

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

                if tool_thinking_prompt is None:
                    return formatted_prompt, response.choices[0].text
                else:
                    return formatted_prompt, "<think>\n" + tool_thinking_prompt + response.choices[0].text
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



class AsyncWebResearchChatAgent:
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
                 build_prompt: Optional[callable] = None,
                 tool_to_thinking_prompt: Optional[Dict[str, str]] = None,
                 build_prompt_with_tool_plan: Optional[callable] = None,
                 bedrock_call: bool = False,
                 tool_to_exclude_in_prompt: Optional[List[str]] = None,
                 **kwargs):

        self.client = client
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(concurrent_limit)
        self.max_llm_call = max_llm_call
        self.gen_config = gen_config 
        self.web_result_config = web_result_config 
        self.url_to_pageinfo = {}
        self.function_map = tool_maps
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.prompt_react = prompt_react or PROMPT_REACT
        self.build_prompt = build_prompt
        self.tool_to_thinking_prompt = tool_to_thinking_prompt
        self.build_prompt_with_tool_plan = build_prompt_with_tool_plan
        self.bedrock_call = bedrock_call
        self.tool_to_exclude_in_prompt = tool_to_exclude_in_prompt or []

        self.name = name
        self.description = description or "Web Research Agent that can search the web and crawl webpages to answer questions."

    async def _call_llm(self, messages, **kwargs) -> str:
        if not self.bedrock_call:
            return await generate_response(
                client=self.client,
                tokenizer=self.tokenizer,
                messages=messages,
                semaphore=self.semaphore,
                temperature=self.gen_config.get('temperature', 0.7),
                top_p=self.gen_config.get('top_p', 0.8),
                max_tokens=self.gen_config.get('max_tokens', 32768),
                repetition_penalty=self.gen_config.get('repetition_penalty', 1.05),
                top_k=self.gen_config.get('top_k_sampling', 20),
                model_name=self.model_name,
                retry_limit=kwargs.get('retry_limit', 3),
                enable_thinking=self.enable_thinking,
                tool_thinking_prompt=kwargs.get('tool_thinking_prompt', None)
            )
        else:
            return await generate_response_claude(
                messages=messages,
                semaphore=self.semaphore,
                model_id=self.model_name,
                region=kwargs.get('region', "us-west-2"),
                system_prompt=kwargs.get('system_prompt', DEFAULT_CLAUDE_SYSTEM_PROMPT),
                max_tokens=kwargs.get('max_tokens', 2048),
                temperature=kwargs.get('temperature', 0.7),
                retry_limit=kwargs.get('retry_limit', 3),
                tool_thinking_prompt=kwargs.get('tool_thinking_prompt', None),
                stop_words=kwargs.get('stop_words', ["</tool_call>", "</answer>"])
            )

    async def run(self, question: str, **kwargs) -> Dict:
        messages = [Message(role=USER, content=self._prepend_react_prompt(question))]

        seq = dict(
            messages=None,
            initial_prompt="",
            output="",
            tool_calls={tool_name: 0 for tool_name in self.function_map.keys()},
            failed_call=0,
            total_call=0,
            finished=False,
            tokens_num=0,
            history=[]  # to store the raw responses for debugging
        )
        llm_call_n = 0
        while llm_call_n < self.max_llm_call:
            formatted_prompt, raw_response = await self._call_llm(messages, **kwargs)
            # print("formatted_prompt:", formatted_prompt[:1000] + "...")
            # print("raw_response:", raw_response[:1000] + "...")
            seq["history"].append(raw_response) # mainly for debugging purpose, can be removed later

            if llm_call_n == 0:
                seq["initial_prompt"] = formatted_prompt
                seq["tokens_num"] = len(self.tokenizer(formatted_prompt, return_tensors="pt").input_ids[0])

            has_action, action, action_input, proc_response = self._detect_tool(raw_response)

            if not has_action:
                has_answer, proc_response = self._detect_answer(raw_response)
                if has_answer:
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    seq["total_call"] = llm_call_n + 1
                    seq["finished"] = True
                    seq["tokens_num"] += len(self.tokenizer(proc_response, return_tensors="pt").input_ids[0])
                    seq["output"] = proc_response
                    break
                else:
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    messages.append(Message(role=USER, content="[Instruction Reminder] Please follow the format: either return <answer>...</answer> for final answer, or call a tool using <tool_call>{...}</tool_call>."))
                    seq["failed_call"] += 1
                    seq["tokens_num"] += len(self.tokenizer(proc_response, return_tensors="pt").input_ids[0])
                    
            else:
                if action_input:
                    observation = await self._call_tool(action, action_input, **kwargs)
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    if self.bedrock_call:
                        observation = f"<tool_response>\n{observation}\n</tool_response>"
                        messages.append(Message(role=USER, content=observation))
                    else:
                        messages.append(Message(role=TOOL, content=observation))
                    seq["tool_calls"][action] += 1
                    seq["total_call"] += 1
                    seq["tokens_num"] += len(self.tokenizer(proc_response + observation, return_tensors="pt").input_ids[0])
                else:
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    seq["failed_call"] += 1 
                    seq["tokens_num"] += len(self.tokenizer(proc_response, return_tensors="pt").input_ids[0])

            if seq["tokens_num"] > self.max_tokens:
                print(f"[Warning] Tokens number {seq['tokens_num']} exceeds max limit {self.max_tokens}. Stopping further calls.")
                break

            llm_call_n += 1
            # print("[DEBUG] formated_prompt:", formatted_prompt[:1000] + "...") 
            # print("[DEBUG] raw_response:", raw_response[:1000] + "...")
            # print("[DEBUG] proc_response:", proc_response[:1000] + "...")
        seq["messages"] = [msg.to_dict() for msg in messages]
        seq["output"] = next((m.content for m in reversed(messages) if m.role == ASSISTANT), "")

        return seq
    
    async def run_with_soft_tool_plan(self, question: str, **kwargs) -> Dict:
        user_prompt, tool_plans = self.build_prompt_with_tool_plan(question)
        messages = [Message(role=USER, content=user_prompt)]

        seq = dict(
            messages=None,
            initial_prompt="",
            output="",
            tool_calls={tool_name: 0 for tool_name in self.function_map.keys()},
            failed_call=0,
            total_call=0,
            finished=False,
            tokens_num=0,
            history=[]  # to store the raw responses for debugging
        )
        llm_call_n = 0
        while llm_call_n < self.max_llm_call:
            if len(tool_plans) > 0:
                next_tool = tool_plans.pop(0)
                if next_tool not in self.function_map:
                    raise ValueError(f"Tool {next_tool} not found in function map.")
                tool_thinking_prompt = self.tool_to_thinking_prompt[next_tool]
                kwargs['tool_thinking_prompt'] = tool_thinking_prompt
            formatted_prompt, raw_response = await self._call_llm(messages, **kwargs)
            if "tool_thinking_prompt" in kwargs:
                kwargs.pop('tool_thinking_prompt', None)  # clear after use

            seq["history"].append(raw_response) # mainly for debugging purpose, can be removed later

            if llm_call_n == 0:
                seq["initial_prompt"] = formatted_prompt
                seq["tokens_num"] = len(self.tokenizer(formatted_prompt, return_tensors="pt").input_ids[0])

            has_action, action, action_input, proc_response = self._detect_tool(raw_response)

            if not has_action:
                has_answer, proc_response = self._detect_answer(raw_response)
                if has_answer:
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    seq["total_call"] = llm_call_n + 1
                    seq["finished"] = True
                    seq["tokens_num"] += len(self.tokenizer(proc_response, return_tensors="pt").input_ids[0])
                    seq["output"] = proc_response
                    break
                else:
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    messages.append(Message(role=USER, content="[Instruction Reminder] Please follow the format: either return <answer>...</answer> for final answer, or call a tool using <tool_call>{...}</tool_call>."))
                    seq["failed_call"] += 1
                    seq["tokens_num"] += len(self.tokenizer(proc_response, return_tensors="pt").input_ids[0])
                    
            else:
                if action_input:
                    observation = await self._call_tool(action, action_input, **kwargs)
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    if self.bedrock_call:
                        observation = f"<tool_response>\n{observation}\n</tool_response>"
                        messages.append(Message(role=USER, content=observation))
                    else:
                        messages.append(Message(role=TOOL, content=observation))
                    seq["tool_calls"][action] += 1
                    seq["total_call"] += 1
                    seq["tokens_num"] += len(self.tokenizer(proc_response + observation, return_tensors="pt").input_ids[0])
                else:
                    messages.append(Message(role=ASSISTANT, content=proc_response))
                    seq["failed_call"] += 1 
                    seq["tokens_num"] += len(self.tokenizer(proc_response, return_tensors="pt").input_ids[0])

            if seq["tokens_num"] > self.max_tokens:
                print(f"[Warning] Tokens number {seq['tokens_num']} exceeds max limit {self.max_tokens}. Stopping further calls.")
                break

            llm_call_n += 1
            # print("[DEBUG] formated_prompt:", formatted_prompt[:1000] + "...") 
            # print("[DEBUG] raw_response:", raw_response[:1000] + "...")
            # print("[DEBUG] proc_response:", proc_response[:1000] + "...")
        seq["messages"] = [msg.to_dict() for msg in messages]
        seq["output"] = next((m.content for m in reversed(messages) if m.role == ASSISTANT), "")

        return seq

    def _prepend_react_prompt(self, question: str) -> List[Message]:
        if self.build_prompt is not None:
            return self.build_prompt(question)
        
        tool_descs = []
        keep_tool_names = []
        for tool_name, function in self.function_map.items():
            # This is used for search only baseline
            if tool_name in self.tool_to_exclude_in_prompt:
                continue
            name = function.get('name', None)
            name_for_human = function.get('name_for_human', name)
            name_for_model = function.get('name_for_model', name)
            assert name_for_human and name_for_model
            args_format = function.get('args_format', '')
            tool_descs.append(
                TOOL_DESC.format(name_for_human=name_for_human,
                                 name_for_model=name_for_model,
                                 description_for_model=function['description'],
                                 parameters=json.dumps(function['parameters'], ensure_ascii=False),
                                 args_format=args_format).rstrip())
            keep_tool_names.append(function.name)
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(keep_tool_names)
        if tool_names not in  ["web_search,crawl_webpage", "web_search"]:
            raise ValueError(f"Tool names should be 'web_search,crawl_webpage', but got {tool_names}")
        
        prompt = self.prompt_react.format(
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=question,
        )
        
        return prompt
    
    def _detect_answer(self, text: str) -> Tuple[bool, str]:
        answer_match = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
        if answer_match:
            answer_end_idx = text.rfind("</answer>") + len("</answer>")
            return True, text[:answer_end_idx]
        else:
            return False, text

    def _detect_tool(self, text: str) -> Tuple[bool, str, str, str]:
        """
        Detect the first tool call from LLM response formatted with <tool_call>...</tool_call>.

        Returns:
            has_tool: Whether a tool was detected
            tool_name: Tool name if found
            tool_args: JSON of tool arguments if found or error message
            trunc_response
        """
        tool_match = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", text, re.DOTALL)

        if not tool_match:
            return False, "", "", ""

        try:
            # 1. 取第一个 tool_call
            raw_tool_call = tool_match[0]
            tool_call = json.loads(raw_tool_call)

            tool_name = tool_call.get("name", "").strip()
            tool_params = tool_call.get("parameters", None)

            # 2. 检查 tool 是否存在
            if tool_name not in self.function_map:
                err_msg = (
                    f"\nTool `{tool_name}` does not exist. "
                    f"Please use one of the available tools: {list(self.function_map.keys())}\n"
                )
                return True, "", "", text.split("</tool_call>")[0] + " </tool_call>" + err_msg

            # 3. 检查是否提供 parameters
            if tool_params is None:
                err_msg = (
                    f'\nTool call missing "parameters" field. '
                    f'Please format tool call as: {{"name": "tool_name", "parameters": {{...}}}}\n'
                )
                return True, "", "", text.split("</tool_call>")[0] + " </tool_call>" + err_msg

            return True, tool_name, tool_params, text.split("</tool_call>")[0] + " </tool_call>"

        except Exception as e:
            err_msg = f"\n[ERROR] Failed to parse <tool_call>: {e}\n"
            return True, "", "", text.split("</tool_call>")[0] + " </tool_call>" + err_msg


    def _postprocess_web_content(self, result: Dict, content: str, content_extract_method: str, context_char: int = 2000) -> str: 
        if content_extract_method == "snippet_f1":
            success, snippet = extract_snippet_with_context(content, result['snippet'], context_chars=context_char)
            return snippet 
        elif content_extract_method == "first_webpage":
            return content[:context_char] if len(content) > context_char else content
        else:
            print(f"Content extract method `{content_extract_method}` is not implemented in func _postprocess_web_content.") # we do this because the func is called under try, hence won't raise error
 
    async def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
       
        tool = self.function_map[tool_name]
        assert tool.is_async, f'Tool {tool_name} is not async, but called in async mode.' 
        
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
            