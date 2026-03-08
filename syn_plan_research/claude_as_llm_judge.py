import pandas as pd 
import os
import json
# import boto3
from datetime import date
import aioboto3
import asyncio
from tqdm import tqdm
from botocore.exceptions import ClientError
import re

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
REGION = "us-east-1"

prompt_template = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to any of the labeled answers.

Question:
{question}

Labeled Answers (one or more possible correct answers):
{labeled_answer}

Predicted Answer:
{pred_answer}

Are these answers equivalent? Please respond with "Correct" if the predicted answer matches any of the labeled answers, or "Incorrect" if it does not. Do not include any other text.
"""

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="LLM as judge using Claude API")
    parser.add_argument("--raw_eval_path", type=str, required=True, help="Path to the raw evaluation data")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--subset_num", type=int, default=-1, help="Number of samples to process (0 for all)")
    return parser.parse_args()

def build_claude_request(messages, system_prompt, stop_tokens=["</tool_call>", "</answer>"]):
    clean_messages = [
        {k: v for k, v in m.items() if k in ["role", "content"]}
        for m in messages
    ]
    print("clean_messages:", clean_messages)
    return json.dumps({
        "messages": clean_messages,
        "max_tokens": 8192,
        "system": system_prompt,
        "anthropic_version": "bedrock-2023-05-31",
        "stop_sequences": stop_tokens,
    })

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
              pass_k=(score_col, "size"),           # 只统计非NaN。如果你想把NaN也算进去就换成 "size"
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
                       avg_score_std_over_ids=("score_std", "mean"),   # ← 你要的“把std再平均”
                       avg_pass_k_over_ids=("pass_k", "mean"),
                       num_ids=("id", "nunique"),
                   )
    )

    return per_pair_df, per_source_df



def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


async def async_call_claude(
    messages,
    semaphore: asyncio.Semaphore,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region: str = "us-west-2",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    retry_limit: int = 5,
):  
    from botocore.config import Config
    import botocore.session
    from aiobotocore.session import get_session

    session = get_session()
    legacy_session = botocore.session.get_session()
    credentials = legacy_session.get_credentials().get_frozen_credentials()

    clean_messages = [
        {k: v for k, v in m.items() if k in ["role", "content"]}
        for m in messages
    ]

    body = json.dumps({
        "messages": clean_messages,
        "system": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "anthropic_version": "bedrock-2023-05-31",
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
                        return parsed["content"][0]["text"]
                    else:
                        return "[Claude Error]: No valid content"
        except Exception as e:
            print(f"[Claude Error] attempt {attempt+1}/{retry_limit}: {e}")
            if "maximum context length" in str(e).lower() and max_tokens > 256:
                max_tokens = max_tokens // 2
                print(f"→ Reducing max_tokens to {max_tokens}")
            if attempt == retry_limit - 1:
                return f"[ERROR] {str(e)}"
            await asyncio.sleep(2 * (attempt + 1))

    return "[Claude Error] Unexpected failure"


async def batch_call_claude(message_list, system_prompt=DEFAULT_SYSTEM_PROMPT, max_concurrency=32):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def track_progress(messages):
        result = await async_call_claude(messages, semaphore=semaphore, system_prompt=system_prompt)
        pbar.update(1)
        return result

    with tqdm(total=len(message_list), desc="Calling Claude", dynamic_ncols=True) as pbar:
        tasks = [track_progress(messages) for messages in message_list]
        results = await asyncio.gather(*tasks)
    return results



if __name__ == "__main__":
    args = parse_arguments()
    df = pd.read_parquet(args.raw_eval_path)
    if not args.out_dir:
        args.out_dir = os.path.join(os.path.dirname(args.raw_eval_path), "llm_as_judge_eval")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    if args.subset_num > 0:
        df = df.sample(n=args.subset_num, random_state=42).reset_index(drop=True)
    prompts = []
    extract_answers = []
    all_messages = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing messages"):
        pred_answer = extract_solution(row["responses"])
        if pred_answer is None:
            extract_answers.append(False)
            pred_answer = "No answer found"
        else:
            extract_answers.append(True)
        
        labeled_answer_str = "\n".join(f"- {ans}" for ans in row["reward_model"]["ground_truth"]["target"])
        prompt = prompt_template.format(
            question=row["question"],
            labeled_answer=labeled_answer_str,
            pred_answer=pred_answer
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        all_messages.append(messages)

    assert len(all_messages) == len(df), "Mismatch in number of messages and dataframe rows"
    print("answer valid rates: ", sum(extract_answers) / len(extract_answers))
    all_judges = asyncio.run(batch_call_claude(all_messages, system_prompt=DEFAULT_SYSTEM_PROMPT))
    assert len(all_judges) == len(df), (len(all_judges), len(df))

    judege_scores = []
    for idx, judge in enumerate(all_judges):
        judge = judge.strip()
        if judge.lower() == "correct":
            judege_scores.append(1.0)
        elif judge.lower() == "incorrect":
            judege_scores.append(0.0)
        else:
            print(f"Unexpected judge response at index {idx}: {judge}")
            judege_scores.append(None)

    df["llm_as_judge"] = judege_scores

    per_pair_df, per_source_df = compute_passk_stats(df, score_col="llm_as_judge")
    per_pair_df.to_parquet(os.path.join(args.out_dir, "per_pair_scores.parquet"))
    per_source_df.to_parquet(os.path.join(args.out_dir, "per_source_scores.parquet"))
    per_pair_df.to_json(os.path.join(args.out_dir, "per_pair_scores.json"), orient="records", indent=4)
    per_source_df.to_json(os.path.join(args.out_dir, "per_source_scores.json"), orient="records", indent=4)
