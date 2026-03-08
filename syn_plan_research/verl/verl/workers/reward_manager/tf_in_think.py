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

from collections import defaultdict
import numpy as np

import torch

from verl import DataProto
from verl.utils.reward_score import tf_in_think_web_research_qa_f1
from verl.workers.reward_manager import register


@register("tf_in_think")
class TFinThinkRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", format_penlty=0.4) -> None:
        """
        Initialize the TFinThinkRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = tf_in_think_web_research_qa_f1.compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.format_penlty = format_penlty  # Coefficient for format reward

        print( "🔧 [DEBUG] successfully initialized TFinThinkRewardManager with format_penlty:", self.format_penlty)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        original_reward_extra_keys = set()

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # currently it only works for Qwen3 
            assert "Qwen/Qwen3" in self.tokenizer.name_or_path, f"Tokenizer {self.tokenizer.name_or_path} is not supported. We only support Qwen3 tokenizer."
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = "<|im_start|>assistant\n" + self.tokenizer.decode(valid_response_ids)  # We assume the response starts with <|im_start|>assistant\n, it is only works for Qwen3 now

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            # num_turns = data_item.non_tensor_batch.get("__num_turns__", None)

            for key, val in data_item.non_tensor_batch.items(): 
                if "__num" in key and "turns__" in key:
                    extra_info[key] = val

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                format_penlty=self.format_penlty,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    if i == 0:
                        original_reward_extra_keys.add(key)
                    if key == "score":
                        continue
                    reward_extra_info[key].append(value)
            else:
                reward = score

            for key, val in extra_info.items():
                if key not in original_reward_extra_keys:
                    reward_extra_info[key].append(val)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str.rstrip(self.tokenizer.pad_token))
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        print(f"🔧 [DEBUG] original_reward_extra_keys: {original_reward_extra_keys}")
        print(f"🔧 [DEBUG] reward_extra_info keys: ")
        for key in reward_extra_info.keys():
            print(f"🔧 [DEBUG] {key}: {len(reward_extra_info[key])}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
