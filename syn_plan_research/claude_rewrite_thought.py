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

def build_short_thought_prompt(user_prompt: str, history: list, current_action: str) -> str:
    """
    Constructs a prompt to generate a short, natural 'Your_Rewrite_Thought' for the current action.

    Args:
        user_prompt (str): The original user question or task instruction.
        history (list): A list of tuples (thought, action, observation) from previous steps.
        current_action (str): The action at the current step.
        current_observation (str): The observation resulting from the previous action.

    Returns:
        str: The full prompt for Claude or other LLMs.
    """
    prompt = """You are simulating the internal reasoning process of a strong LLM agent.

You will be given the reasoning trajectory of a **weaker open-source LLM agent** that attempts to complete an information-seeking task via multiple steps. This agent interacts with external tools in a multi-turn format: **Thought → Action → Observation**, aiming to answer a user’s question.

Each step is formatted as:
Thought: ...
Action: ...
Observation: ...

---

Your task is to generate a rewritten version of the current **Thought**, called `Your_Rewrite_Thought`, which explains the rationale behind the **current Action** in a concise and natural way.

You will be provided with:
- The **user prompt** that defines the overall goal.
- The previous history consisting of Your_Rewrite_Thought, Action, and Observation steps.
- The **current Action** taken by the agent.
- The latest **Observation** from the previous step.

---

### Instructions:

1. `Your_Rewrite_Thought` should simulate how an LLM would *think before* taking the current Action.
2. The thought must logically justify the current Action, without diverging from it.
3. Keep it concise and natural—avoid overly verbose, stylistic, or redundant explanations.
4. Do **not** rewrite earlier thoughts.
5. Output ONLY the rewritten thought text — do NOT include any labels or prefixes such as "Your_Rewrite_Thought:", "Thought:", "Action:", or "Observation:".
7. The literal string "Your_Rewrite_Thought" must NOT appear anywhere in your output.

---

"""

    prompt += f"User Prompt:\n\"{user_prompt}\"\n\n"

    prompt += "[Begin Trajectory So Far]\n"
    for idx, (thought, action, observation) in enumerate(history):
        prompt += f"Step {idx}:\n"
        prompt += f"Your_Rewrite_Thought: {thought}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Observation: {observation}\n"
    prompt += "[End Trajectory]\n\n"

    prompt += f"Step {len(history)}:\n"
    prompt += f"Action: {current_action}\n\n"
    prompt += "Your_Rewrite_Thought:"

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

def process_messages_to_triple(messages: list) -> list:
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

def triple_back_to_messages(triples: list, user_prompt: str, final_answer: str) -> list:
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
        messages.append({"role": "assistant", "content": f"<think>\n{thought}\n</think>\n\n<tool_call>\n{action.strip()}\n</tool_call>"})
        messages.append({"role": "tool", "content": observation})

    messages.append({"role": "assistant", "content": final_answer})

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
    all_dict_messages = [process_messages_to_triple(messages) for messages in all_messages]
    all_triples = [item["triples"] for item in all_dict_messages]
    all_user_prompts = [item["user_prompt"] for item in all_dict_messages]
    all_final_answers = [item["final_answer"] for item in all_dict_messages]

    max_turns = max(len(triple) for triple in all_triples)
    print(f"Max turns in any conversation: {max_turns}")
    print("The total number of conversations:", len(all_triples))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    idx_to_new_triples = {idx: [] for idx, _ in enumerate(all_triples)}
    for turn in range(1, max_turns+1):
        keep_idxes = [] 
        rewrite_prompts = []
        keep_triples = []
        for idx, (user_prompt, triples) in enumerate(zip(all_user_prompts, all_triples)):
            if len(triples) < turn:
                continue 
            
            history = triples[:turn-1]
            current_action = triples[turn-1][1]

            rewrite_prompt = build_short_thought_prompt(
                user_prompt=user_prompt,
                history=history,
                current_action=current_action,
            )
            rewrite_prompts.append([{ "role": "user", "content": rewrite_prompt}])
            keep_idxes.append(idx)
            keep_triples.append((None, current_action, triples[turn-1][2]))

        print(f"Turn: {turn}, Number of rewrite prompts: {len(rewrite_prompts)}")

        all_thougths = asyncio.run(batch_call_claude(rewrite_prompts, system_prompt=DEFAULT_SYSTEM_PROMPT))
        assert len(keep_idxes) == len(all_thougths) == len(keep_triples), (len(keep_idxes), len(all_thougths), len(keep_triples))
        for idx, triple, thought in zip(keep_idxes, keep_triples, all_thougths):
            idx_to_new_triples[idx].append((thought, triple[1], triple[2]))

    expected_len = len(all_triples)
    indices = list(idx_to_new_triples.keys())

    assert min(indices) == 0, "Minimum index is not 0"
    assert max(indices) == expected_len - 1, "Maximum index mismatch"
    assert len(idx_to_new_triples) == expected_len, "Length mismatch with triples"

    # 2. 构造有序的 new_triples list
    new_triples = [None] * expected_len
    for idx, triple in idx_to_new_triples.items():
        new_triples[idx] = triple
    new_all_messages = []
    for idx, triples in enumerate(new_triples):
        user_prompt = all_user_prompts[idx]
        final_answer = all_final_answers[idx]
        new_messages = triple_back_to_messages(triples, user_prompt, final_answer)
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
    


