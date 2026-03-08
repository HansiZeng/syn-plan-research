# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from collections import Counter
import jieba


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

def compute_format_reward(full_text: str) -> list[float]:
    # Extract all assistant responses
    assistant_blocks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", full_text, re.DOTALL)
    
    format_rewards = []
    for i, block in enumerate(assistant_blocks): 
        if i == len(assistant_blocks) - 1: 
            format_r = validate_response_structure(block, do_print=False, answer_turn=True)
        else:
            format_r = validate_response_structure(block, do_print=False, answer_turn=False) 

        format_rewards.append(format_r) 

    return all(format_rewards)

def normalize_answer(s, is_multi_choice=False):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if is_multi_choice:
        return white_space_fix(remove_punc(lower(s)))
    else:
        return white_space_fix(remove_articles(remove_punc(lower(s))))

def is_chinese(text: str, threshold: float = 0.1) -> bool:
    # 仅统计非数字字符中的中文占比
    filtered_chars = [ch for ch in text if not ch.isdigit() and not ch.isspace()]
    total_chars = len(filtered_chars)
    if total_chars == 0:
        return False
    chinese_chars = sum(1 for ch in filtered_chars if '\u4e00' <= ch <= '\u9fff')
    return (chinese_chars / total_chars) > threshold

def compute_f1(prediction, golden_answers, possible_chinese=False, is_multi_choice=False):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    f1_score = 0. 
    for golden_answer in golden_answers:
        f1_score = max(f1_score, f1(prediction, golden_answer, possible_chinese=possible_chinese, is_multi_choice=is_multi_choice))
    return f1_score

def f1(prediction, answer, possible_chinese=False, is_multi_choice=False):
    """Compute the F1 score between the prediction and the answer.
    """
    # if only if the data_source is possible_chinese (webwalker) and answer is chinese we compute it as chinese
    if possible_chinese and is_chinese(answer):
        prediction_tokens = list(jieba.cut(normalize_answer(prediction, is_multi_choice=is_multi_choice)))
        ground_truth_tokens = list(jieba.cut(normalize_answer(answer, is_multi_choice=is_multi_choice)))
    else:
        prediction_tokens = normalize_answer(prediction, is_multi_choice=is_multi_choice).split()
        ground_truth_tokens = normalize_answer(answer, is_multi_choice=is_multi_choice).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


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


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(data_source, solution_str, ground_truth, format_score=0.2, good_ans_threshold=0.8):
    """The scoring function for F1.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    possible_chinese = "webwalker" in data_source.lower()
    is_multi_choice = "gpqa" in data_source.lower()

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    ans_score = 0.0 
    if answer is not None:
        ans_score = compute_f1(answer, ground_truth["target"], possible_chinese=possible_chinese, is_multi_choice=is_multi_choice)

    is_format_correct = compute_format_reward(solution_str) 

    if is_format_correct:
        if ans_score > 0.0:
            total_score = ans_score + format_score
        else: 
            total_score = 0. 
    else:
        if ans_score >= good_ans_threshold:
            total_score = 0.0
        else:
            total_score = -format_score

    print(f"🔧 [DEBUG] answer: {answer}, ground_truth: {ground_truth['target']}, score: {total_score}, ans_score: {ans_score}, format_correct: {float(is_format_correct)}")

    return {
        "score": total_score,
        "answer_score": ans_score, 
        "format_correct": float(is_format_correct),
    }
