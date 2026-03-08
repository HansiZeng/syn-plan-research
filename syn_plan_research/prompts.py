PROMPT_REACT = """You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

{tool_descs}

Instructions:
You starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). 
The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.
The tool you can should use is one of the following: {tool_names}.
When you reach the <answer> tag, your final answer MUST be short and direct — ideally one phrase or one sentence. DO NOT repeat the question or explain your reasoning again. The reasoning belongs only in <think>.
For example, when question is who sings Beat it, your answer should be <answer> Michael Jackson </answer>.

Example response:
<think> thinking process here </think>
<tool_call>
{{"name": "tool name here", "parameters": {{"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{{"name": "another tool name here", "arguments": {{...}}}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>



Question: {query}
"""

Sr_Cr1_PROMPT_REACT = """You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

{tool_descs}

Instructions:
You starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). 
The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.
The tool you can should use is one of the following: {tool_names}.

Specifically, for tool calling:
After each `web_search` tool call, you should carefully read the corresponding `<tool_response>` to identify useful URLs from the search results. 
Then, you must call the `crawl_webpage` tool using one of those URLs. Repeat this pattern (search → crawl) as needed before producing your final answer.

Example response:
<think> thinking process here </think>
<tool_call>
{{"name": "tool name here", "parameters": {{"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{{"name": "another tool name here", "arguments": {{...}}}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>

Question: {query}
"""

Sr_Cr2_PROMPT_REACT = """You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

{tool_descs}

Instructions:
You starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). 
The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.
The tool you can should use is one of the following: {tool_names}.

Specifically, for tool calling:
After each `web_search` tool call, you should carefully read the corresponding `<tool_response>` to identify useful URLs from the search results. 
Then, you must make two separate `crawl_webpage` tool calls, each using one of the identified URLs—since each `crawl_webpage` call can handle only a single URL. 
Repeat this pattern (search → crawl → crawl) as needed before producing your final answer.

Example response:
<think> thinking process here </think>
<tool_call>
{{"name": "tool name here", "parameters": {{"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(thinking, tool calls and responses continue...)
<think> thinking process here </think>
<answer> answer here </answer>

Question: {query}
"""


TOOL_NAME_FIRST_PROMPT_REACT = """You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

{tool_descs}

## Instructions:
To answer the question, you may go through one or more reasoning cycles. Each cycle consists of:
1. Choosing a tool to use.
2. Thinking about the tool's parameters.
3. Calling the tool with appropriate inputs.
4. Receiving and processing the tool's response.

You can perform multiple such cycles before producing a final answer. Clearly mark each step using the provided tags. The tool you choose **must** be one of the following: {tool_names}.


## Response Format:
<tool_name> name_of_the_tool </tool_name>
<think> your reasoning for choosing this tool </think>
<tool_call>
{{"name": "name_of_the_tool", "parameters": {{"param1": value1, "param2": value2, ...}}}}
</tool_call>
<tool_response>
tool output here
</tool_response>

(repeat the above cycle as needed)

<think> final reasoning before answering </think>
<answer> final answer here </answer>


## Question: 
{query}
"""

TOOL_NAME_FIRST_PROMPT_REACT = """You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

{tool_descs}

## Instructions:
To answer the question, you may go through one or more reasoning cycles. Each cycle consists of:
1. Choosing a tool to use.
2. Reasoning about the tool's parameters.
3. Calling the tool with appropriate inputs.
4. Receiving and processing the tool's response.

You can perform multiple such cycles before producing a final answer. Clearly mark each step using the provided tags. The tool you choose **must** be one of the following: {tool_names}.


## Response Format:
<tool_name> name_of_the_tool </tool_name>
<reason> your reasoning for choosing this tool </reason>
<tool_call>
{{"name": "name_of_the_tool", "parameters": {{"param1": value1, "param2": value2, ...}}}}
</tool_call>
<tool_response>
tool output here
</tool_response>

(repeat the above cycle as needed)

<reason> final reasoning before answering </reason>
<answer> final answer here </answer>


## Question: 
{query} /no_think
"""



prompt_react_map = {
    "original": PROMPT_REACT,
    "sr_cr1": Sr_Cr1_PROMPT_REACT,
    "sr_cr2": Sr_Cr2_PROMPT_REACT,
    "tf_in_think": TOOL_NAME_FIRST_PROMPT_REACT,
}