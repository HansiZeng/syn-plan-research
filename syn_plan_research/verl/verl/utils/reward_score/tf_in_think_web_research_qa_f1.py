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

import re

def is_tool_use_format(text: str) -> bool:
    # Pattern allowing optional whitespace and newlines between tags and content
    tool_pattern = re.compile(
        r"<tool_name>\s*.+?\s*</tool_name>\s*"
        r"<reason>\s*.+?\s*</reason>\s*"
        r"<tool_call>\s*\{.*?\}\s*</tool_call>",
        re.DOTALL
    )
    return bool(tool_pattern.search(text))

def is_final_answer_format(text: str) -> bool:
    answer_pattern = re.compile(
        r"<reason>\s*.+?\s*</reason>\s*"
        r"<answer>\s*.+?\s*</answer>",
        re.DOTALL
    )
    return bool(answer_pattern.search(text))

def compute_format_reward(full_text: str) -> list[float]:
    # Extract all assistant responses
    assistant_blocks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", full_text, re.DOTALL)
    
    format_matches = []
    for i, block in enumerate(assistant_blocks): 
        if i < len(assistant_blocks) - 1: 
            if is_tool_use_format(block):
                format_matches.append(True)
            else:
                format_matches.append(False)
        else:
            if is_final_answer_format(block):
                format_matches.append(True)
            else:
                format_matches.append(False)

    return float(all(format_matches))
    

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    f1_score = 0. 
    for golden_answer in golden_answers:
        f1_score = max(f1_score, f1(prediction, golden_answer))
    return f1_score

def f1(prediction, answer):
    """Compute the F1 score between the prediction and the answer.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(answer).split()
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


def compute_score(data_source, solution_str, ground_truth, format_penlty=0.4):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if answer is None: 
        ans_score = 0.0
    else:
        ans_score = compute_f1(answer, ground_truth["target"]) # not find equal 0. 
    format_score = compute_format_reward(solution_str) 

    if format_score == 1.:
        if ans_score > 0.8:
            final_score = ans_score
        else:
            final_score = format_penlty
    else:
        if ans_score > 0.8:
            final_score = ans_score - format_penlty
        else:
            final_score = 0.0

    print(f"🔧 [DEBUG] answer: {answer}, ground_truth: {ground_truth['target']}, score: {final_score}, format_score: {format_score}, ans_score: {ans_score}")

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str.rstrip('<|endoftext|>')}")

    return {
        "score": final_score,
        "format_score": format_score,
        "ans_score": ans_score,
    }
