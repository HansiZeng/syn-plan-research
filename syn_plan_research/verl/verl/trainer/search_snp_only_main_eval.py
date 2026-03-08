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
"""
Generate responses given a dataset of prompts
"""

import os
import time  # ← 新增：用于生成唯一后缀，避免命名冲突
import hydra
import numpy as np
import ray
import json

# ↓↓↓ 新增：减少后台进程与僵尸风险（等价于 --include-dashboard=false）
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.trainer.main_ppo import create_rl_dataset
from verl.workers.reward_manager.eval_naive import EvalNaiveRewardManager
from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from tqdm import tqdm

from tests.workers.rollout.my_tools_sever import (
    WebSearchToolClient,
    CrawlWebpageToolClient,
    WebSearchToolServer,
    CrawlWebpageToolServer,
)
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.tool_utils import WebSearchCacheSaver

import pandas as pd


# ↓↓↓ 新增：通用安全杀 Actor 工具（支持 None/单个/列表）
def _kill_actor(a):
    try:
        if a is None:
            return
        if isinstance(a, (list, tuple)):
            for x in a:
                _kill_actor(x)
            return
        ray.kill(a, no_restart=True)
    except Exception:
        pass


def compute_passk_stats(df: pd.DataFrame, score_col: str = "reward", ddof: int = 1):
    """
    - per_pair_df：每个 (id, data_source) 的 pass_k / max / mean / std(基于这组k个score)
    - per_source_df：每个 data_source 上，以上指标在不同 id 上的平均（其中 std 是“先算每题std，再做平均”）
    """
    # 确保分数列是数值型
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # 1) 对每个 (id, data_source) 计算：样本数、max、mean、std(对这组k个score)
    per_pair_df = (
        df.groupby(["id", "data_source"], as_index=False)
        .agg(
            pass_k=(score_col, "size"),  # 只统计非NaN
            score_max=(score_col, "max"),
            score_mean=(score_col, "mean"),
            score_std=(score_col, lambda x: x.std(ddof=ddof)),
        )
    )

    # 单样本/全NaN时 std 会是 NaN，这里把 pass_k<=1 的 std 置为 0，更符合直觉
    per_pair_df.loc[per_pair_df["pass_k"] <= 1, "score_std"] = 0.0

    # 2) 每个 data_source 上，对不同 id 的这些指标再取平均
    per_source_df = (
        per_pair_df.groupby("data_source", as_index=False)
        .agg(
            avg_score_max_over_ids=("score_max", "mean"),
            avg_score_mean_over_ids=("score_mean", "mean"),
            avg_score_std_over_ids=("score_std", "mean"),  # ← 再平均
            avg_pass_k_over_ids=("pass_k", "mean"),
            num_ids=("id", "nunique"),
        )
    )

    return per_pair_df, per_source_df


def merge_result_into_df(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Merge a result dict into a DataFrame.

    Args:
        df (pd.DataFrame): Original DataFrame.
        result (dict): Dictionary containing additional data. Values can be lists or dicts of lists.

    Returns:
        pd.DataFrame: New DataFrame with merged columns.
    """
    df = df.copy()
    n = len(df)

    for key, val in result.items():
        if isinstance(val, list):
            if len(val) != n:
                raise ValueError(f"Length mismatch for key '{key}': expected {n}, got {len(val)}")
            df[key] = val

        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if not isinstance(sub_val, list):
                    raise TypeError(
                        f"Expected list for result['{key}']['{sub_key}'], got {type(sub_val).__name__}"
                    )
                if len(sub_val) != n:
                    raise ValueError(
                        f"Length mismatch for result['{key}']['{sub_key}']: expected {n}, got {len(sub_val)}"
                    )
                df[f"{key}.{sub_key}"] = sub_val

        else:
            raise TypeError(f"Unsupported type for key '{key}': expected list or dict, got {type(val).__name__}")

    return df


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARNING",
                    "VLLM_USE_V1": "1",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    # ===== 新增：为本次 run 生成唯一后缀，避免命名冲突（可选但推荐） =====
    uniq = f"{os.getpid()}_{int(time.time())}"

    # ===== 新增：提前声明，finally 中统一清理 =====
    web_search_actor = None
    cache_saver_actor = None
    async_rollout_manager = None

    try:
        print("✅ output path: ", config.actor_rollout_ref.eval_output_path)
        # pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # init tool servers（限制自动重启，避免 driver 结束后被拉起）
        web_search_actor = WebSearchToolServer.options(
            name=f"web_search_server_{uniq}",
            max_restarts=0,
            max_task_retries=0,
        ).remote(
            api_key=config.tool_server.web_search_server.api_key,
            cache_file=config.tool_server.web_search_server.cache_file,
        )
        web_search_addr = ray.get(web_search_actor.get_server_address.remote())
        print(f"✅ WebSearchToolServer started at {web_search_addr}")

        # === 2. 写入 config.tool_server.xxx.url 和 parameters ===
        config.tool_server.web_search_server.url = f"http://{web_search_addr}"
        config.tool_server.web_search_server.parameters = ray.get(web_search_actor.get_parameters.remote())

        cache_saver_actor = WebSearchCacheSaver.options(
            max_restarts=0,
            max_task_retries=0,
        ).remote(
            base_url=config.tool_server.web_search_server.url,
            parameters=config.tool_server.web_search_server.parameters,
        )

        # === 3. 构造 tool_metadata_map ===
        web_search_metadata = ray.get(web_search_actor.get_metadata.remote())

        tool_metadata_map = {
            config.tool_server.web_search_server.name: web_search_metadata,
        }
        async_rollout_manager = init_async_rollout_manager(config)

        # create rl dataset:
        processor = None
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, tool_metadata_map
        )
        if config.validation.pass_at_k > 1:
            print(f"🔍 Repeating dataset {config.validation.pass_at_k} times for pass@k evaluation.")
            val_dataset.repeat_data(config.validation.pass_at_k)
            if config.validation.data_source:
                print(
                    f"🔍 Filtering dataset by source '{config.validation.data_source}' for pass@k evaluation."
                )
                val_dataset.filter_by_source(config.validation.data_source)
            print(f"🔍 New dataset length after repeat and filter: {len(val_dataset)}")

        batch_size = 4096
        all_dfs = []
        for i in tqdm(
            range(0, len(val_dataset), batch_size),
            total=len(val_dataset) // batch_size,
            desc="Processing batches",
        ):
            batch_dataset_lst = []
            for j in range(i, min(i + batch_size, len(val_dataset))):
                batch_dataset_lst.append(val_dataset[j])

            collated_dataset = collate_fn(batch_dataset_lst)
            gen_inputs = DataProto.from_single_dict(collated_dataset)

            gen_inputs.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print("check first prompt:", gen_inputs.non_tensor_batch["raw_prompt"][0][0]["content"])

            # get rollout manager
            reward_fn = EvalNaiveRewardManager(
                tokenizer=tokenizer,
                num_examine=-1,
                compute_score=None,  # use default compute_score
                reward_fn_key=config.data.reward_fn_key,
            )
            assert (
                config.actor_rollout_ref.rollout.n == 1
            ), "actor_rollout_ref.rollout.n should be 1 for generation task."
            gen_outputs = async_rollout_manager.generate_sequences(gen_inputs)
            gen_inputs = gen_inputs.union(gen_outputs)
            assert len(gen_outputs) == len(batch_dataset_lst)

            # get response and reward function
            result = reward_fn(gen_inputs)

            batch_df = merge_result_into_df(
                val_dataset.dataframe.to_pandas().iloc[i : i + batch_size], result
            )
            all_dfs.append(batch_df)

            # break

        output_dir = os.path.dirname(config.actor_rollout_ref.eval_output_path)
        # write to a new parquet
        makedirs(output_dir, exist_ok=True)
        result_df = pd.concat(all_dfs, ignore_index=True)
        result_df.to_parquet(config.actor_rollout_ref.eval_output_path)

        if config.validation.pass_at_k == 1:
            # compute metric for each domain
            ds_to_score = {}
            if "data_source" in result_df.columns:
                grouped = result_df.groupby("data_source")
                for domain, group in grouped:
                    scores = group["reward"].tolist()
                    ds_to_score[domain] = {
                        "score": round(np.mean(scores), 3),
                        "count": len(scores),
                    }
            with open(os.path.join(output_dir, "metric.json"), "w") as f:
                json.dump(ds_to_score, f, indent=4)

            print("💾 Start to save web search cache...")
            status = ray.get(cache_saver_actor.save_cache.remote())
            print(f"✅ Web search cache saved: {status}")
        else:
            # compute pass@k stats
            per_pair_df, per_source_df = compute_passk_stats(result_df, score_col="reward")
            per_pair_df.to_parquet(os.path.join(output_dir, "per_pair_scores.parquet"))
            per_source_df.to_parquet(os.path.join(output_dir, "per_source_scores.parquet"))
            # save as json
            per_pair_df.to_json(os.path.join(output_dir, "per_pair_scores.json"), orient="records", indent=4)
            per_source_df.to_json(os.path.join(output_dir, "per_source_scores.json"), orient="records", indent=4)

    finally:
        # ====== 新增：尽量先让 rollout manager 自己释放（如关闭 vLLM 引擎等）======
        try:
            if async_rollout_manager is not None and hasattr(async_rollout_manager, "shutdown"):
                async_rollout_manager.shutdown()
        except Exception:
            pass

        # ====== 新增：显式杀掉我们创建的所有 Actors（禁止自动重启）======
        _kill_actor(cache_saver_actor)
        _kill_actor(web_search_actor)


if __name__ == "__main__":
    # 确保无论是否异常都 shutdown（释放本地 auto-start 的 Ray 进程）
    try:
        main()
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass
