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

from direct_answer_chat import DirectAnswerChatAgent


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

    data, dataset_name = load_dataset(args)
    if args.subset_num > 0:
        data = data[:args.subset_num]
    print("original dataset size:", len(data))
    new_data = []
    if args.pass_at_k > 1:
        new_data = [copy.deepcopy(d) for d in data for _ in range(args.pass_at_k)]
        data = new_data
        print("new dataset size after pass_at_k:", len(data))
    questions = [] 
    for item in data:
        q = item["Question"] if "Question" in item else item["question"]
        questions.append(q)

    output_dir = f"{args.work_dir}/outputs/{dataset_name}.{'-'.join(args.model_name.split('/')[-2:])}.direct_answer_chat"
    os.makedirs(output_dir, exist_ok=True)

    print("size of questions:", len(questions))
    print("example question: ", questions[0])
    print("output_dir:", output_dir)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base_url)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)


    agent = DirectAnswerChatAgent(
        client=client,
        tokenizer=tokenizer,
        model_name=args.model_name,
        concurrent_limit=args.concurrent_limit,
        gen_config={
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
            'top_k_sampling': args.top_k_sampling,
        },
        enable_thinking=not args.no_enable_thinkng,
    )
    
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
        out_seqs.append(seq)

   
    # Save
    t = time.localtime()
    random_num = str(random.randint(0, 99)).zfill(2)
    result_json_name = f'direct_ans_{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}.json'

    if args.pass_at_k > 1:
        result_json_name = f'pass_at_k_{args.pass_at_k}-{result_json_name}'

    with open(os.path.join(output_dir, result_json_name), 'w', encoding='utf-8') as f:
        json.dump(out_seqs, f, ensure_ascii=False, indent=2)


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
    parser.add_argument('--work_dir', type=str, default='/workspace/', help="Root directory for data files")
    parser.add_argument("--no_enable_thinkng", action="store_true")
    parser.add_argument("--use_qwen3_suggest_params", action="store_true")
    parser.add_argument("--context_chars", type=int, default=2000,)
    parser.add_argument("--pass_at_k", type=int, default=1)



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
