# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import concurrent.futures
import os
import re
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
import json
import random

import fastapi
import numpy as np
import ray
import uvicorn
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from starlette.responses import JSONResponse
import pandas as pd
from tensordict import TensorDict

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, ToolCompletionCallback
import torch

from tests.workers.rollout.my_utils import (
    extract_snippet_with_context,
)
from tests.workers.rollout.my_tools_sever import WebSearchToolClient, CrawlWebpageToolClient, WebSearchToolServer, CrawlWebpageToolServer
from tests.workers.rollout.my_tools import format_search_results

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # 只显示 error

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ✅ 输出到文件
file_handler = logging.FileHandler("my_tool_debug.log",  mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



TOOL_DESC = (
    "<tool>\n"
    "Tool Name: {name_for_model}\n"
    "Description: {description_for_model}\n"
    "Usage: This tool lets the agent interact with the {name_for_human} API.\n"
    "Parameters (JSON Schema): {parameters}\n"
    "{args_format}\n"
    "</tool>"
)



PROMPT_REACT = """You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

{tool_descs}

Instructions:
You starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). 
The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.
The tool you can should use is one of the following: {tool_names}.

Example response:
<think> thinking process here </think>
<tool_call>
{{"name": "tool name here", "parameters": {{"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{{"name": "another tool name here", "arguments": {{...}}}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>


Question: {query}
"""

ASSISTANT = "assistant"
TOOL = "tool"
USER = "user"
SYSTEM = "system"


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]
    
def prepend_react_prompt(tool_metadata_map, question: str) -> str:
        tool_descs = []
        for function in tool_metadata_map.values():
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
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_name for tool_name in tool_metadata_map)
        prompt = PROMPT_REACT.format(
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=question,
        )
        return prompt

class Qwen3CustomToolCompletionCallback(ToolCompletionCallback):
    def __init__(self, config: DictConfig, scheduler: ChatCompletionScheduler):
        super().__init__(config, scheduler)

        self.max_assistant_turns = config.actor_rollout_ref.rollout.max_assistant_turns

        if config.tool_server.crawl_webpage_server.server_num == 1:
            self.function_map = {
                config.tool_server.web_search_server.name: WebSearchToolClient(base_url=config.tool_server.web_search_server.url, parameters=config.tool_server.web_search_server.parameters, max_concurrency=config.tool_server.web_search_server.client_max_concurrency),
                config.tool_server.crawl_webpage_server.name: CrawlWebpageToolClient(base_url=config.tool_server.crawl_webpage_server.url, parameters=config.tool_server.crawl_webpage_server.parameters, max_concurrency=config.tool_server.crawl_webpage_server.client_max_concurrency)
            }
        else:
            from tests.workers.rollout.my_tools_sever import CrawlWebpageToolClientV2
            self.function_map = {
                config.tool_server.web_search_server.name: WebSearchToolClient(base_url=config.tool_server.web_search_server.url, parameters=config.tool_server.web_search_server.parameters, max_concurrency=config.tool_server.web_search_server.client_max_concurrency),
                config.tool_server.crawl_webpage_server.name: CrawlWebpageToolClientV2(base_url=config.tool_server.crawl_webpage_server.url, 
                                                                                       parameters=config.tool_server.crawl_webpage_server.parameters,
                                                                                       max_concurrency=config.tool_server.crawl_webpage_server.client_global_concurrency,
                                                                                       per_endpoint_concurrency=config.tool_server.crawl_webpage_server.client_per_endpoint_concurrency)
            }
        print("function_map keys: ", self.function_map.keys())
        self.web_result_config = config.tool_server.web_search_server.web_result_config

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any], filter_convs: List[float]) -> None:
        role, content, finish_reason = (
            completions.choices[0].message.role,
            completions.choices[0].message.content,
            completions.choices[0].finish_reason,
        )
        messages.append({"role": role, "content": content})
        turn = len(messages)
        assert len(filter_convs) == 1, f"filter_convs length {len(filter_convs)} should be 1, but got {filter_convs}"

        # STEP 0: check if we reach max turns
        if len(messages) >= self.max_assistant_turns:
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Reach max turns, done!")
            filter_convs[0] = 0.0
            return

        # STEP 1: check if we reach max tokens
        if finish_reason == "length":
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Reach max tokens, done!")
            filter_convs[0] = 0.0
            return

        # ------------------------------ start to modify the content -------------------------------

        has_action, action, action_input, proc_content = self._detect_tool(content)
        if has_action:
            if action_input:
                observation = await self._call_tool(action, action_input, debug_msg=content) # need to modify later
                messages[-1]["content"] = proc_content
                messages.append({"role": "tool", "content": observation})
            else:
                messages[-1]["content"] = proc_content 
        else:
            has_answer, proc_content = self._detect_answer(content)
            if has_answer:
                messages[-1]["content"] = proc_content 
                filter_convs[0] = 1.0
                print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Got answer: {proc_content}, done!, filter_convs={filter_convs}")
                return
            else:
                # change the logic after 07/25/2025
                # 1. Before that, I add a message role user, and content: [Instruction Reminder] Please follow the format: either return <answer>...</answer> for final answer, or call a tool using <tool_call>{...}</tool_call>."}
                # 2. But I found in RL, Qwen3-8B have increasing failed tool calling rates when training goes on. 
                # 3. First of all, adding the user role in the middle of messages is buggy for Qwen3-8B, as they will remove all thinking contents of assistant before the user role.
                # 4. 
                # messages.append({"role": "user", "content": "[Instruction Reminder] Please follow the format: either return <answer>...</answer> for final answer, or call a tool using <tool_call>{...}</tool_call>."})

                messages[-1]["content"] = proc_content + "\n The response failed to follow the format. Fail to call tool or provide answer."
                print(f"[id={completions.id},turn={turn},finish_reason=['Don't Follow Instruction'] Failed to call tool or provide answer")
                filter_convs[0] = 0.0
                return 


        # STEP 5: resubmit chat completions with code block output
        self.scheduler.submit_chat_completions(
            messages=messages,
            request_id=completions.id,
            info=info,
            filter_convs=filter_convs
        )
    
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
                # 08/21/2025 I don't know why Qwen3-4B often cannot provide correct format. 
                # To prevent the issue, I make True --> False (has_action)
                return False, "", "", text.split("</tool_call>")[0] + " </tool_call>" + err_msg

            # 3. 检查是否提供 parameters
            if tool_params is None:
                err_msg = (
                    f'\nTool call missing "parameters" field. '
                    f'Please format tool call as: {{"name": "tool_name", "parameters": {{...}}}}\n'
                )
                # 08/21/2025 I don't know why Qwen3-4B often cannot provide correct format. 
                # To prevent the issue, I make True --> False (has_action)
                return False, "", "", text.split("</tool_call>")[0] + " </tool_call>" + err_msg

            return True, tool_name, tool_params, text.split("</tool_call>")[0] + " </tool_call>"

        except Exception as e:
            err_msg = f"\n[ERROR] Failed to parse <tool_call>: {e}\n"
            # 08/21/2025 I don't know why Qwen3-4B often cannot provide correct format. 
            # To prevent the issue, I make True --> False (has_action)
            # This will make the conversation be filtered; meaning that：they do involve in advantage computation, but not used to update model parameters.
            return False, "", "", text.split("</tool_call>")[0] + " </tool_call>" + err_msg


    def _postprocess_web_content(self, result: Dict, content: str, content_extract_method: str, context_char: int = 2000) -> str: 
        if content_extract_method == "snippet_f1":
            success, snippet = extract_snippet_with_context(content, result['snippet'], context_chars=context_char)
            return snippet 
        else:
            print(f"Content extract method `{content_extract_method}` is not implemented in func _postprocess_web_content.") # we do this because the func is called under try, hence won't raise error
 
    async def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'

        debug_msg = kwargs.pop('debug_msg', None)
        tool = self.function_map[tool_name]
        # assert tool.is_async, f'Tool {tool_name} is not async, but called in async mode.' 
        # async with self.tool_semaphore:
        try:
            tool_result = await tool.call(tool_args, **kwargs)

            if tool_name == "web_search":
                if self.web_result_config["snippet_only"]:
                    if isinstance(tool_result, str):
                        observation = tool_result
                    else:
                        observation = format_search_results(tool_result)
                        for result in tool_result:
                            self.function_map["crawl_webpage"].save_snippet(result['url'], result["snippet"])
                    # print("Search observatinon:", observation[:500] + "...")
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
                                result, content, self.web_result_config["content_extract_method"])
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
            logger.error(f"😋 [DEBUG] Error happened calling tool: {tool_name} with args: {tool_args}, tool_result: {tool_result}") # , debug_msg: {debug_msg}

            exception_type = type(ex).__name__
            exception_message = str(ex)
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n'
            print(error_message)
            return error_message


    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int, response_filter: List[float]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        # response_filter: response-level filter. 1.0 means keep, 0.0 means discard.

        # prompts: [prompt] from input dataset
        prompts = [
            self.tokenizer.apply_chat_template(
                prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False
            )
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]
        assert len(batch_conversations) == len(prompts) * n
        assert len(response_filter) == len(prompts) * n, f"response_filter length {len(response_filter)} does not match prompts length {len(prompts) * n}"

        # sequences: [prompt + response]
        sequences = [
            self.tokenizer.apply_chat_template(
                conversation, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False
            )
            for conversation in batch_conversations
        ]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(
            batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0),
            batch_conversations,
            responses["input_ids"],
            responses["attention_mask"],
        )

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        original_response_mask = responses["attention_mask"].clone()
        response_filter = torch.tensor(response_filter, dtype=response_mask.dtype, device=response_mask.device)
        response_mask = response_mask * response_filter.unsqueeze(1) 

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
                "original_response_mask":  original_response_mask,  # [bsz, response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        num_tool_turns = np.array(
            [sum(1 for message in conversation if message["role"] == "tool") for conversation in batch_conversations],
            dtype=np.int32,
        )
        extra_info = {f"__num_{tool_name}_turns__": [] for tool_name in self.function_map.keys()}
        for conversation in batch_conversations:
            for tool_name in self.function_map.keys():
                extra_info[f"__num_{tool_name}_turns__"].append(self._get_tool_call_turns(conversation, tool_name))
        # to numpy array
        for tool_name in self.function_map.keys():
            extra_info[f"__num_{tool_name}_turns__"] = np.array(extra_info[f"__num_{tool_name}_turns__"], dtype=np.int32)
        extra_info.update({
            "__num_tool_turns__": num_tool_turns,
        })
        extra_info["response_filter"] = response_filter.cpu().numpy() 
        non_batch_tensor = {
            "__num_turns__": num_turns,
             **extra_info,
        }

        return DataProto(batch=batch, non_tensor_batch=non_batch_tensor)

    def _get_tool_call_turns(self, message, tool_name):
        n_call = 0
        for text in [ mssg for mssg in message if mssg["role"] == ASSISTANT]:
            tool_match = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", text["content"], re.DOTALL)
            if tool_match:
                try:
                    raw_tool_call = tool_match[0]
                    tool_call = json.loads(raw_tool_call)
                    if tool_call.get("name", "").strip() == tool_name:
                        n_call += 1
                except Exception as e:
                    continue

        return n_call

if __name__ == "__main__":
    print("CUDA visible devices:", torch.cuda.device_count())
    
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )


    all_actor_names = ray.util.list_named_actors()
    print("✅ 当前活跃的 Ray Actors:")
    for name in all_actor_names:
        print(name)
 
    # Load config
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/workspace/verl/checkpoints/web_research_async_rl/qwen3-8b-web_research-turns_12_20250717_041110/global_step_10/actor/hf_model" # Qwen/Qwen3-8B" 
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    # config.actor_rollout_ref.rollout.multi_turn.format = "hermes" ###### double check
    config.actor_rollout_ref.rollout.multi_turn.completion_callback = (
        "tests.workers.rollout.my_test_web_research_completion_callback.Qwen3CustomToolCompletionCallback"
    )
    config.actor_rollout_ref.rollout.prompt_length = 32768
    config.actor_rollout_ref.rollout.response_length = 8192
    config.actor_rollout_ref.rollout.n = 1
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.8

    # try:
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    # except Exception as e:
    #     print(f"Read tokenizer from model_path's huggingface subdir.")
    #     tokenizer = hf_tokenizer(os.path.join(config.actor_rollout_ref.model.path, "huggingface"))


    # start server  
    web_search_actor = WebSearchToolServer.remote(
        api_key="005fba2b8daa23f10be87a7d76a5bd37c99627c7",  # ✅ 记得替换成你的 Serper API Key
        cache_file="/workspace/cache/serper_search_cache.json"
    )
    loop = asyncio.get_event_loop()
    web_search_addr = ray.get(web_search_actor.get_server_address.remote())
    print(f"✅ WebSearchToolServer started at {web_search_addr}")
    crawl_actor = CrawlWebpageToolServer.remote(
        semaphore_limit=32,
        cache_file="/workspace/cache/crawl4ai_url_cache.json",
    )
    crawl_addr = ray.get(crawl_actor.get_server_address.remote())
    print(f"✅ CrawlWebpageToolServer started at {crawl_addr}")

    config.tool_server.web_search_server.url = f"http://{web_search_addr}"
    config.tool_server.crawl_webpage_server.url = f"http://{crawl_addr}"


    # format prompt 
    web_search_metadata = ray.get(web_search_actor.get_metadata.remote())
    crawler_metadata = ray.get(crawl_actor.get_metadata.remote())

    tool_metadata_map = {
        config.tool_server.web_search_server.name: web_search_metadata,
        config.tool_server.crawl_webpage_server.name: crawler_metadata,
    }
    
    config.tool_server.web_search_server.parameters = ray.get(web_search_actor.get_parameters.remote())
    config.tool_server.crawl_webpage_server.parameters = ray.get(crawl_actor.get_parameters.remote())
    # Init sandbox and async rollout manager
    async_rollout_manager = init_async_rollout_manager(config)

    # Build dataset
    data_path = "/workspace/data/odqa_gaia_webwalker_gpqa_dev_sample_13K/dev.parquet"
    dataset = pd.read_parquet(data_path).to_dict(orient='records')
    random.seed(42)
    # dataset.shuffle()
    dataset = dataset[:8000]  # Limit to 100 samples for testing

    prompts = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": prepend_react_prompt(tool_metadata_map, question=item["question"])}]
                    for item in dataset
                ]
            ),
        },
    )

    print("check first prompt:", prompts.non_tensor_batch["raw_prompt"][0][0]["content"])
    sampling_params = {
        "chat_template_kwargs": {"enable_thinking": config.qwen3.enable_thinking}
    }

    result = async_rollout_manager.generate_sequences(prompts=prompts)
    assert len(result) == len(dataset) * config.actor_rollout_ref.rollout.n

    # Check max turns that sandbox is called
    num_turns = result.non_tensor_batch["__num_turns__"]
    print(f"num_turns: {num_turns}")
    assert np.max(num_turns) > 2, f"max turns: {np.max(num_turns)}"

    # Check response_mask
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"


    # write to disk
    output_dir = "/workspace/scripts/web_research/latest_verl/outputs"
    os.makedirs(output_dir, exist_ok=True)

    dataset = [item for item in dataset for _ in range(config.actor_rollout_ref.rollout.n)]
    assert len(dataset) == len(result.batch["prompts"]) == len(result.batch["responses"]), f"{len(dataset)} != {len(result.batch['prompts'])} != {len(result.batch['responses'])}"

    for idx, (prompt, response, mask) in enumerate(zip(result.batch["prompts"], result.batch["responses"], result.batch["response_mask"])):
        prompt_str = tokenizer.decode(prompt)
        response_str = tokenizer.decode(response)

        dataset[idx]["prompt_str"] = prompt_str
        dataset[idx]["response_str"] = response_str
        dataset[idx]["response_mask"] = mask.tolist()
        dataset[idx]["response"] = response.tolist()

    output_path =  data_path.split("/")[1] + "_".join(model_path.split("/")) + "web_research_results.parquet"

    if config.tool_server.web_search_server.web_result_config["snippet_only"]:
        output_path = f'snp-ctxchar_{config.tool_server.web_search_server.web_result_config["context_chars"]}-{output_path}'
    output_path = os.path.join(output_dir, output_path)
    pd.DataFrame(dataset).to_parquet(output_path, index=False)
    # Decode responses with response_mask
    # for i in range(len(responses)):
    #     valid_tokens = responses[i][response_mask[i].bool()]
    #     response_str = tokenizer.decode(valid_tokens)
    #     assert "<tool_response>" not in response_str, f"found <tool_response> in response: {response_str}"
    #     assert "</tool_response>" not in response_str, f"found </tool_response> in response: {response_str}"
    #     print(f"response: {response_str}")

    # print("Test passed!")
