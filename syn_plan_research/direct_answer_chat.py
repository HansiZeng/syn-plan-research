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
from tools import BaseTool


TOOL_DESC = (
    "<tool>\n"
    "Tool Name: {name_for_model}\n"
    "Description: {description_for_model}\n"
    "Usage: This tool lets the agent interact with the {name_for_human} API.\n"
    "Parameters (JSON Schema): {parameters}\n"
    "{args_format}\n"
    "</tool>"
)


PROMPT_DIRECT_ANSWER = """You are a helpful and intelligent agent capable of answering complex questions using your own internal reasoning and knowledge, without external tools.
The answer should be concise and directly address the question asked if possible.

When responding, follow this format strictly:
<think>
Your step-by-step reasoning and thought process goes here.
</think>
<answer>
Your final answer goes here.
</answer>


Question: {query}
"""

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



class DirectAnswerChatAgent:
    """This agent use ReAct format to call tools"""

    def __init__(self,
                 client: AsyncOpenAI,
                 tokenizer: AutoTokenizer,
                 model_name: str = "QwQ-32B",
                 name: Optional[str] = "direct-answer-chat-agent",
                 description: Optional[str] = None,
                 concurrent_limit: int = 32,
                 gen_config: Optional[Dict] = None,
                 max_tokens: int = 32768,
                 enable_thinking: bool = True,
                 **kwargs):

        self.client = client
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(concurrent_limit)
        self.gen_config = gen_config 
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        
        self.name = name 
        self.desription = description or "Web Research Agent that can search the web and crawl webpages to answer questions."

    async def _call_llm(self, messages, **kwargs) -> str:
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
            enable_thinking=self.enable_thinking
        )

    async def run(self, question: str, **kwargs) -> Dict:
        messages = [Message(role=USER, content=self._prepend_direct_answer_prompt(question))]
        seq = dict(
            messages=None,
            initial_prompt="",
            output="",
            tokens_num=0,
            finished=False,
        )
        
        formatted_prompt, raw_response = await self._call_llm(messages, **kwargs)
        seq["initial_prompt"] = formatted_prompt
        seq["tokens_num"] = len(self.tokenizer(formatted_prompt, return_tensors="pt").input_ids[0])
        has_answer, proc_response = self._detect_answer(raw_response)
        messages.append(Message(role=ASSISTANT, content=proc_response))
        seq["output"] = proc_response
        seq["messages"] = messages
        seq["messages"] = [msg.to_dict() for msg in messages]
        if has_answer:
            seq["finished"] = True

        return seq

    def _prepend_direct_answer_prompt(self, question: str) -> List[Message]:
        prompt = PROMPT_DIRECT_ANSWER.format(
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
