import pandas as pd 
import os
import json
# import boto3
from datetime import date
import aioboto3
import asyncio
from tqdm import tqdm
from botocore.exceptions import ClientError

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
REGION = "us-east-1"

def build_short_answer_thought_prompt(user_prompt: str, history: list, answer: str) -> str:
    prompt = """You are simulating the internal reasoning process of a capable LLM agent.

You will be given:
1. A user prompt that describes the task and question.
2. A reasoning trajectory for an information-seeking task, consisting of multiple steps in the format:
   Thought: ...
   Action: ...
   Observation: ...
   (repeated for multiple turns)
3. A final answer produced by the agent.

Your task: Rewrite the final thought that naturally leads to the given final answer.

Guidelines:
- The written Thought must summarize the key results from the previous tool calls and logically connect them to the final answer.
- Output only the Thought content itself (no labels, no formatting, no additional text).
- The style should sound natural and coherent, as if the agent is concluding its reasoning before answering.

"""

    prompt += f"User Prompt:\n\"{user_prompt}\"\n\n"

    prompt += "[Start Reasoning trajectory]\n"
    for idx, (thought, action, observation) in enumerate(history):
        prompt += f"Step {idx}:\n"
        prompt += f"Thought: {thought}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Observation: {observation}\n"
    prompt += "[End Trajectory]\n\n"

    prompt += f"Final Answer: {answer}\n\n"
    prompt += f"Thought:\n"

    return prompt

def validate_response_structure(processed_str: str, do_print: bool, answer_turn=False) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # processed_str = '<think> </think>' + processed_str
    
    # Check required tags
    if answer_turn:
        tags = {
            'think_start': ('<think>', 1),
            'think_end': ('</think>', 1),
            'answer_start': ('<answer>', 1),
            'answer_end': ('</answer>', 1)
        }
    else:
        tags = {
            'think_start': ('<think>', 1),
            'think_end': ('</think>', 1),
            'answer_start': ('<tool_call>', 1),
            'answer_end': ('</tool_call>', 1)
        }


    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print("processed_str:", processed_str)
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")

    if not validation_passed:
        return validation_passed
    
    # check if <think> and </think> are not empty
    think_start_idx = positions['think_start'] + len(tags['think_start'][0])
    think_end_idx = positions['think_end']
    assert think_start_idx < think_end_idx, "Think start index should be less than think end index"
    think_content = processed_str[think_start_idx:think_end_idx]

    if think_content.strip() == "":
        if do_print:
            print("  [Error] <think>...</think> is empty or only contains whitespace")
        validation_passed = False
    else:
        if do_print:
            print("  <think>...</think> contains non-whitespace content")

    return validation_passed 

def process_messages_to_triple_and_answer(messages: list) -> list:
    """
    Converts a list of message dictionaries into a list of (thought, action, observation) tuples.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'.

    Returns:
        list: List of tuples (thought, action, observation).
    """
    triples = []
    for i, mssg in enumerate(messages):
        if i == 0:
            assert mssg["role"] == "user", "First message must be from the user" 
            user_prompt = mssg["content"]
        elif i == len(messages) - 1:
            assert mssg["role"] == "assistant", "Last message must be from the assistant"
            final_answer = mssg["content"]
            assert validate_response_structure(final_answer, do_print=False, answer_turn=True), "Final answer does not meet the required structure"
            final_answer = final_answer.split("</answer>")[0].rstrip('\n').split("<answer>")[-1].lstrip('\n')
        else:
            if mssg["role"] == "user":
                continue 
            if mssg["role"] == "assistant":
                assert messages[i+1]["role"] == "tool", "Assistant message must be followed by a tool message"
                content = mssg["content"]
                assert validate_response_structure(content, do_print=False, answer_turn=False), "Assistant message does not meet the required structure"
                thought = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                action = content.split("</tool_call>")[0].rstrip('\n').split("<tool_call>")[-1].lstrip('\n') 
                observation = messages[i+1]["content"]

                triples.append((thought, action, observation))

    return {
        "user_prompt": user_prompt,
        "final_answer": final_answer,
        "triples": triples
    }

def triple_back_to_messages(triples: list, user_prompt: str, final_answer: str, final_thought: str) -> list:
    """
    Converts a list of (thought, action, observation) tuples back into a list of message dictionaries.

    Args:
        triples (list): List of tuples (thought, action, observation).
        user_prompt (str): The original user prompt.
        final_answer (str): The final answer from the assistant.

    Returns:
        list: List of message dictionaries with 'role' and 'content'.
    """
    messages = [{"role": "user", "content": user_prompt}]
    
    for thought, action, observation in triples:
        thought = thought.strip()
        if thought.startswith("Your_Rewrite_Thought:"):
            thought = thought[len("Your_Rewrite_Thought:"):]
        if thought.startswith("Thought:"):
            thought = thought[len("Thought:"):]
        action = action.strip('\n')
        messages.append({"role": "assistant", "content": f"<think>\n{thought}\n</think>\n\n<tool_call>\n{action}\n</tool_call>"})
        messages.append({"role": "tool", "content": observation})

    final_thought = final_thought.strip("\n")
    final_answer = final_answer.strip("\n")
    messages.append({"role": "assistant", "content": f"<think>\n{final_thought}\n</think>\n\n<answer>{final_answer}</answer>"})

    return messages

def message_contain_claude_error(messages: list) -> bool:
    """
    Checks if any message in the list contains a Claude error.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'.

    Returns:
        bool: True if any message contains a Claude error, False otherwise.
    """
    for mssg in messages:
        if mssg["role"] == "assistant" and "[Claude Error]" in mssg["content"]:
            return True
    return False

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Process web research chat arguments.")
    parser.add_argument("--raw_train_path", type=str, required=True, help="Path to the raw training data.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed output data.")
    parser.add_argument("--subset_num", type=int, default=-1, help="Subset number for processing.")
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
    df = pd.read_parquet(args.raw_train_path)
    if args.subset_num > 0:
        df = df.sample(n=args.subset_num, random_state=42).reset_index(drop=True)

    all_messages = [item for item in df["messages"]]
    all_dict_messages = [process_messages_to_triple_and_answer(messages) for messages in all_messages]
    all_triples = [item["triples"] for item in all_dict_messages]
    all_user_prompts = [item["user_prompt"] for item in all_dict_messages]
    all_final_answers = [item["final_answer"] for item in all_dict_messages]

    max_turns = max(len(triple) for triple in all_triples)
    print("The total number of conversations:", len(all_triples))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    assert len(all_user_prompts) == len(all_triples) == len(all_final_answers), \
        (len(all_user_prompts), len(all_triples), len(all_final_answers))

    rewrite_prompts = []
    for user_prompt, triples, answer in zip(all_user_prompts, all_triples, all_final_answers):
        rewrite_prompt = build_short_answer_thought_prompt(
            user_prompt=user_prompt,
            history=triples,
            answer=answer
        )
        rewrite_prompts.append([{ "role": "user", "content": rewrite_prompt}])
    print(f"Total number of rewrite prompts: {len(rewrite_prompts)}")
    all_thoughts = asyncio.run(batch_call_claude(rewrite_prompts, system_prompt=DEFAULT_SYSTEM_PROMPT))
    assert len(all_thoughts) == len(rewrite_prompts) == len(all_final_answers), (len(all_thoughts), len(rewrite_prompts), len(all_final_answers))

    new_all_messages = []
    for idx, (user_prompt, triples, final_answer, final_thought) in enumerate(zip(all_user_prompts, all_triples, all_final_answers, all_thoughts)):
        new_messages = triple_back_to_messages(triples, user_prompt, final_answer, final_thought)
        new_all_messages.append(new_messages)
    assert len(new_all_messages) == len(all_messages), (len(new_all_messages), len(all_messages))

    df["messages"] = new_all_messages
    old_len = len(df)
    print("size of df before clean claude error:", len(df))
    clean_df = [] 
    for idx, row in df.iterrows():
        if not message_contain_claude_error(row["messages"]):
            clean_df.append(row)
    df = pd.DataFrame(clean_df)
    print("size of df after clean claude error:", len(df))
    print("We removed", old_len - len(df), "conversations with Claude errors")
    df.to_parquet(args.output_path, index=False)
    


