from json import tool
import random
import json
from typing import Union, List, Dict

def generate_tool_plan(tool_list: List[str], first_tool_name="web_search", min_steps=3, max_steps=6):
    assert first_tool_name in tool_list, f"First tool {first_tool_name} must be in the tool names: {tool_list}"
    plan = [first_tool_name]
    num_steps = random.randint(min_steps, max_steps) - 1  # -1 because we already added the first tool
    for _ in range(num_steps):
        plan.append(random.choice(tool_list))
    return plan

# def generate_example_response(tool_plan, tool_to_parameters: Dict[str, List[str]]) -> str:
#     response_parts = []
#     for i, tool in enumerate(tool_plan):
#         response_parts.append(f"<think> Thinking about which tool to use... Step {i+1} </think>")
#         call = {
#             "name": tool,
#             "parameters": {
#                 "query" if tool == "web_search" else "url": f"parameter_value_{i+1}"
#             }
#         }
#         response_parts.append("<tool_call>")
#         response_parts.append(json.dumps(call, indent=2))
#         response_parts.append("</tool_call>")
#         response_parts.append("<tool_response>")
#         response_parts.append(f"response content from {tool} (step {i+1})")
#         response_parts.append("</tool_response>")
#     response_parts.append("<think> final reasoning before answering </think>")
#     response_parts.append("<answer> final answer here </answer>")
#     return "\n".join(response_parts)

def generate_example_response(tool_plan, tool_to_parameters: Dict[str, List[str]]) -> str:
    response_parts = []

    for i, tool in enumerate(tool_plan):
        response_parts.append(f"<think> Thinking about which tool to use... Step {i+1} </think>")

        # 构造参数
        param_dict = {}
        for param in tool_to_parameters.get(tool, []):
            param_dict[param] = f"parameter_value_{i+1}_{param}"

        call = {
            "name": tool,
            "parameters": param_dict
        }

        response_parts.append("<tool_call>")
        response_parts.append(json.dumps(call, indent=2))
        response_parts.append("</tool_call>")
        response_parts.append("<tool_response>")
        response_parts.append(f"response content from {tool} (step {i+1})")
        response_parts.append("</tool_response>")

    response_parts.append("<think> final reasoning before answering </think>")
    response_parts.append("<answer> final answer here </answer>")
    return "\n".join(response_parts)



def build_prompt_factory(function_map, first_tool_name="web_search", min_steps=3, max_steps=6, with_tool_plan=False):
    def build_prompt(question: str) -> str:
        tool_to_parameters = {
            tool.name: list(tool.parameters["properties"].keys())
            for tool in function_map.values()
        }
        tool_list = list(tool_to_parameters.keys())

        tool_plan = generate_tool_plan(tool_list, first_tool_name, min_steps, max_steps)
        tool_plan_str = " -> ".join(tool_plan)

        # tool descriptions
        tool_descs = []
        for function in function_map.values():
            name = function.get('name', None)
            name_for_human = function.get('name_for_human', name)
            name_for_model = function.get('name_for_model', name)
            assert name_for_human and name_for_model
            args_format = function.get('args_format', '')
            tool_descs.append(
                f"- {name_for_model}: {function['description']}"
            )
        tool_descs_str = '\n'.join(tool_descs)
        tool_names = ', '.join(tool.name for tool in function_map.values())
        example_response = generate_example_response(tool_plan, tool_to_parameters=tool_to_parameters)

        prompt = f"""You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

{tool_descs_str}

Instructions:
You start with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and end with (thinking about the answer -> answer of the question). 
The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.
The tool you can use should be one of the following: {tool_names}.

Below is a tool usage plan you should follow (in order): {tool_plan_str}

When you reach the <answer> tag, your final answer MUST be short and direct — ideally one phrase or one sentence. DO NOT repeat the question or explain your reasoning again. The reasoning belongs only in <think>.
For example, when question is who sings Beat it, your answer should be <answer> Michael Jackson </answer>.

Example response:
{example_response}

Question: {question}
"""
        if with_tool_plan:
            return prompt, tool_plan
        else:
            return prompt
    return build_prompt


# tool_to_thinking_prompt = {
#     "web_search": "I'm not sure I have enough context yet... maybe I should search the web to see what's out there.",
#     "crawl_webpage": "Hmm, there are several promising links from search results. Perhaps I should look into one of them to get more specific information."
# }

tool_to_thinking_prompt = {
    "web_search": (
        "To answer this, I probably need to gather external information first. Let me think about the best way to phrase the query for web_search."
    ),
    "crawl_webpage": (
        "There are several promising links from search results. Perhaps I should look into one of them to get more specific information"
    )
}
