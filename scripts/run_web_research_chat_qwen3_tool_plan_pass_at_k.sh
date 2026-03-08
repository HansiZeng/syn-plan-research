#!/bin/bash

set -e

# 默认模型和tokenizer名称
MODEL_NAME="Qwen/Qwen3-32B"
TOKENIZER_NAME=$MODEL_NAME

API_KEY="token-abc123"  # 替换为你的实际值
API_BASE_URL="http://main-model-server:6980/v1"  # 替换为你的部署地址

# 工具API密钥与缓存路径（可选）
SERPER_API_KEY="005fba2b8daa23f10be87a7d76a5bd37c99627c7"  # 替换为你的SERPER key（如果用 web_search）
SERPER_CACHE="/workspace/cache/serper_search_cache.json"
CRAWLER_CACHE="/workspace/cache/crawl4ai_url_cache.json"

# 数据集与运行设置
# SUBSET_NUM=1  # 设置为 -1 表示用全量数据
TOOL_NAMES="web_search crawl_webpage"

# 执行命令

pass_at_ks=(16 16)
data_paths=(
    "/workspace/data/odqa_gpqa_webwalker_2k_train_Qwen3_8B/subset_1.parquet"
    "/workspace/data/odqa_gpqa_webwalker_2k_train_Qwen3_8B/subset_2.parquet"
)

for i in "${!pass_at_ks[@]}"; do
    pass_at_k="${pass_at_ks[$i]}"
    data_path="${data_paths[$i]}"

    echo "pass_at_k: $pass_at_k"
    echo "data_path: $data_path"
    for prompt_type in "original"; do
        echo "prompt_type: $prompt_type"
   
    cmd=(
        python syn_plan_research/run_web_research_chat_tool_plan.py
        --model_name "$MODEL_NAME"
        --tokenizer_name "$TOKENIZER_NAME"
        --api_key "$API_KEY"
        --api_base_url "$API_BASE_URL"
        --serper_api_key "$SERPER_API_KEY"
        --serper_cache_file "$SERPER_CACHE"
        --crawler_cache_file "$CRAWLER_CACHE"
        --tool_names $TOOL_NAMES
        --max_llm_call 15
        --temperature 1.0
        --top_p 1.0
        --max_tokens 8192
        --repetition_penalty 1.05
        --top_k_sampling -1
        --concurrent_limit 50
        --tool_concurrent_limit 50
        --content_extract_method "snippet_f1"
        --work_dir "/workspace/"
        --snippet_only
        --pass_at_k "$pass_at_k"
        --data_path "$data_path"
        --context_chars 5000
    )
     cmd+=(--use_qwen3_suggest_params)
   
    echo "${cmd[@]}"
    "${cmd[@]}"
    done
done
