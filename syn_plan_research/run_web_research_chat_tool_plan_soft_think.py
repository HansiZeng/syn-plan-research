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
from diverse_prompts import build_prompt_factory, tool_to_thinking_prompt


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
        build_prompt_with_tool_plan=build_prompt_factory(
            function_map=tool_maps,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            with_tool_plan=True),
        tool_to_thinking_prompt=tool_to_thinking_prompt,
    )
    
    print("web_result_config:", agent.web_result_config)
    tasks = [agent.run_with_soft_tool_plan(q) for q in questions] 
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
        result_json_name = f'ws_ctxchar_{args.web_search_context_chars}_crawl_ctxchar_{args.context_chars}-{result_json_name}'

    if args.pass_at_k > 1:
        result_json_name = f'pass_at_k_{args.pass_at_k}-{result_json_name}'

    result_json_name = f"diverse_prompt_tool_soft_think_minstep_{args.min_steps}_maxstep_{args.max_steps}-{result_json_name}"

    with open(os.path.join(output_dir, result_json_name), 'w', encoding='utf-8') as f:
        json.dump(out_seqs, f, ensure_ascii=False, indent=2)

    for tool in agent.function_map.values():
        if hasattr(tool, 'save_cache') and callable(getattr(tool, 'save_cache')):
            tool.save_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--api_base_url', type=str, required=True)
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
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument("--max_llm_call", type=int, default=15,)
    parser.add_argument("--serper_api_key", type=str, default=None, help="API key for SERP API if using web search tools")
    parser.add_argument("--serper_cache_file", type=str, default=None, help="Path to cache file for SERP API results")
    parser.add_argument("--crawler_cache_file", type=str, default=None, help="Path to cache file for web crawler results")
    parser.add_argument("--tool_names", type=str, nargs='+', default=["web_search", "crawl_webpage"],
                        help="List of tool names to use in the agent")
    parser.add_argument('--work_dir', type=str, default='/workspace/', help="Root directory for data files")
    parser.add_argument("--no_enable_thinkng", action="store_true")
    parser.add_argument("--use_qwen3_suggest_params", action="store_true")
    parser.add_argument("--context_chars", type=int, default=5000,)
    parser.add_argument("--pass_at_k", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset file if not using default paths")
    parser.add_argument("--web_search_context_chars", type=int, default=0, help="Context characters for web search results")
    parser.add_argument("--min_steps", type=int, default=3, help="Minimum steps for tool planning")
    parser.add_argument("--max_steps", type=int, default=6, help="Maximum steps for tool planning")



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