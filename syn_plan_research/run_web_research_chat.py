import os
import time
import json
import random
import argparse
import asyncio
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import asyncio
import copy
import pandas as pd 

from web_research_chat import AsyncWebResearchChatAgent  # your agent class
from tools import WebSearchTool, CrawlWebpageTool  # assuming these tools are defined in your tools module
from prompts import prompt_react_map  # assuming you have a prompts module with prompt_react_map defined

def load_dataset(args):
    if args.single_question:
        return [{'Question': args.single_question}], 'custom'

    dataset_path_map = {
        'supergpqa': f'{args.work_dir}/data/SuperGPQA/{args.split}.json',
        'webwalker': f'{args.work_dir}/data/WebWalkerQA/{args.split}.json',
        'browsecomp': f'{args.work_dir}/data/BrowseComp/{args.split}.json',
        'openthoughts': f'{args.work_dir}/data/OpenThoughts/{args.split}.json',
        'webthinker': f'{args.work_dir}/data/WebThinker/{args.split}.json',
        'gaia': f'{args.work_dir}/data/GAIA/{args.split}.json',
    }
    data_path = dataset_path_map.get(args.dataset_name,
        f'{args.work_dir}/data/{args.dataset_name.upper()}/{args.split}.json' if args.dataset_name in ["math500", "gpqa"]
        else f'{args.work_dir}/data/{args.dataset_name}/{args.split}.json')

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data, args.dataset_name

async def main_async(args):
    random.seed(args.seed or int(time.time()))
    np.random.seed(args.seed or int(time.time()))

    if args.data_path is not None:
        data = pd.read_parquet(args.data_path).to_dict(orient='records')
        dataset_name = "_".join(args.data_path.split('/')[-2:]).split('.')[0]
    else:
        data, dataset_name = load_dataset(args)
    if args.subset_num > 0:
        data = data[:args.subset_num]
    print("original dataset size:", len(data), "dataset_name:", dataset_name)
    new_data = []
    if args.pass_at_k > 1:
        new_data = [copy.deepcopy(d) for d in data for _ in range(args.pass_at_k)]
        data = new_data
        print("new dataset size after pass_at_k:", len(data))
    questions = [] 
    for item in data:
        q = item["Question"] if "Question" in item else item["question"]
        questions.append(q)

    output_dir = f"{args.work_dir}/outputs/{dataset_name}.{'-'.join(args.model_name.split('/')[-2:])}.webresearch_chat"
    os.makedirs(output_dir, exist_ok=True)

    print("size of questions:", len(questions))
    print("example question: ", questions[0])
    print("output_dir:", output_dir)

    if args.use_bedrock:
        client = None
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base_url)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    tool_maps = {}
    for tool_name in args.tool_names:
        if tool_name == "web_search":
            tool_maps[tool_name] = WebSearchTool(
                api_key=args.serper_api_key,
                cache_file=args.serper_cache_file,
            )
        elif tool_name == "crawl_webpage":
            tool_maps[tool_name] = CrawlWebpageTool(
                cache_file=args.crawler_cache_file,
                semaphore= asyncio.Semaphore(args.tool_concurrent_limit)
            )
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")

    if args.snippet_only and args.web_search_context_chars > 0:
            raise ValueError("When using snippet_only, web_search_context_chars should be 0.")
    agent = AsyncWebResearchChatAgent(
        client=client,
        tokenizer=tokenizer,
        tool_maps=tool_maps,
        model_name=args.model_name,
        concurrent_limit=args.concurrent_limit,
        max_llm_call=args.max_llm_call,
        gen_config={
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
            'top_k_sampling': args.top_k_sampling,
        },
        web_result_config={
            'snippet_only': args.snippet_only,
            'content_extract_method': args.content_extract_method,
            "context_chars": args.context_chars,
            "web_search_context_chars": args.web_search_context_chars
        },
        enable_thinking=not args.no_enable_thinkng,
        prompt_react=prompt_react_map[args.prompt_type],
        bedrock_call=args.use_bedrock,
        tool_to_exclude_in_prompt=args.tool_to_exclude_in_prompt,
    )
    
    print("web_result_config:", agent.web_result_config)
    tasks = [agent.run(q) for q in questions] 
    with tqdm(total=len(tasks)) as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
        
        tracked_tasks = [track_progress(task) for task in tasks]
        completed_sequences = await asyncio.gather(*tracked_tasks)

    out_seqs = []
    assert len(completed_sequences) == len(data)
    for seq, item in zip(completed_sequences, data):
        assert len(set(seq.keys()).intersection(set(item.keys()))) == 0, f"Keys in sequence {seq} overlap with item {item}"
        seq.update(item)
        seq["Output"] = seq["output"]
        seq.pop("output", None) # align with WebThinker eval format 
        seq.pop("prompt", None) # remove prompt to save space

        for k, v in seq.items():
            if isinstance(v, np.ndarray):
                seq[k] = v.tolist()
        out_seqs.append(seq)

   
    # Save
    t = time.localtime()
    random_num = str(random.randint(0, 99)).zfill(2)
    result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}.json'
    if args.snippet_only:
        result_json_name = f'snp-ctxchar_{args.context_chars}-{result_json_name}'
    else:
        if "web_search" in args.tool_names and "crawl_webpage" in args.tool_names:
            result_json_name = f'ws_ctxchar_{args.web_search_context_chars}_crawl_ctxchar_{args.context_chars}-{result_json_name}'
        elif "web_search" in args.tool_names and "crawl_webpage" not in args.tool_names:
            result_json_name = f'ws_ctxchar_{args.web_search_context_chars}-{result_json_name}'
        else:
            raise ValueError("At least one of 'web_search' or 'crawl_webpage' must be in tool_names.")

    if args.pass_at_k > 1:
        result_json_name = f'pass_at_k_{args.pass_at_k}-{result_json_name}'

    if args.prompt_type != "original":
        result_json_name = f'prompt_{args.prompt_type}-{result_json_name}'

    if args.use_bedrock:
        if "claude-3-7" in args.model_name.lower():
            result_json_name = f'claude3-7-{result_json_name}'
        else:
            raise ValueError(f"Unknown model name for Bedrock: {args.model_name}")

    if args.is_baseline:
        output_dir = os.path.join(output_dir, 'baseline')
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, result_json_name), 'w', encoding='utf-8') as f:
        json.dump(out_seqs, f, ensure_ascii=False, indent=2)

    for tool in agent.function_map.values():
        if hasattr(tool, 'save_cache') and callable(getattr(tool, 'save_cache')):
            tool.save_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--tokenizer_name', type=str, default="")
    parser.add_argument('--api_key', type=str, default="")
    parser.add_argument('--api_base_url', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default='webthinker')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--subset_num', type=int, default=-1)
    parser.add_argument('--single_question', type=str, default=None)
    parser.add_argument('--max_search_limit', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--repetition_penalty', type=float, default=1.05)
    parser.add_argument('--top_k_sampling', type=int, default=20)
    parser.add_argument('--snippet_only', action='store_true')
    parser.add_argument('--content_extract_method', type=str, default='snippet_f1')
    parser.add_argument('--concurrent_limit', type=int, default=32)
    parser.add_argument('--tool_concurrent_limit', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--max_llm_call", type=int, default=15,)
    parser.add_argument("--serper_api_key", type=str, default=None, help="API key for SERP API if using web search tools")
    parser.add_argument("--serper_cache_file", type=str, default=None, help="Path to cache file for SERP API results")
    parser.add_argument("--crawler_cache_file", type=str, default=None, help="Path to cache file for web crawler results")
    parser.add_argument("--tool_names", type=str, nargs='+', default=["web_search", "crawl_webpage"],
                        help="List of tool names to use in the agent")
    parser.add_argument('--work_dir', type=str, default='/workspace/', help="Root directory for data files")
    parser.add_argument("--no_enable_thinkng", action="store_true")
    parser.add_argument("--use_qwen3_suggest_params", action="store_true")
    parser.add_argument("--context_chars", type=int, default=2000,)
    parser.add_argument("--pass_at_k", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset file if not using default paths")
    parser.add_argument("--prompt_type", type=str, default="original", choices=["original", "sr_cr1", "sr_cr2", "tf_in_think"],)
    parser.add_argument("--web_search_context_chars", type=int, default=0, help="Context characters for web search results")
    parser.add_argument("--use_bedrock", action="store_true", help="Use AWS Bedrock for model inference")
    parser.add_argument("--tool_to_exclude_in_prompt", type=str, nargs='*', default=[],)
    parser.add_argument("--is_baseline", action="store_true", help="Run as a baseline with only web search tool")



    args = parser.parse_args()

    if "qwen3" in args.model_name.lower() and args.use_qwen3_suggest_params: 
        print("Use Qwen3 suggested parameters for web research chat agent.")
        print("[Warning] The sampling parameters will be overridden by Qwen3's suggested parameters.")
        if not args.no_enable_thinkng:
            args.temperature = 0.6
            args.top_p = 0.95
            args.top_k_sampling = 20
        else: 
            args.temperature = 0.7
            args.top_p = 0.8
            args.top_k_sampling = 20


    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
    time.sleep(120) # Let save cache finished
