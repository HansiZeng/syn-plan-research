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

    f1_score = 0.0
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


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, data_source=None):
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

    possible_chinese = "webwalker" in data_source.lower()
    is_multi_choice = "gpqa" in data_source.lower()

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        score = compute_f1(answer, ground_truth["target"], possible_chinese=possible_chinese, is_multi_choice=is_multi_choice)
        print(f"🔧 [DEBUG] answer: {answer}, ground_truth: {ground_truth['target']}, score: {score}")
        return score
