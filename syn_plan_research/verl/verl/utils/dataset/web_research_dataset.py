import json
from verl.utils.dataset.rl_dataset import RLHFDataset
import verl.utils.torch_functional as verl_F

from verl.utils.dataset.web_research_prompts import PROMPT_MAP

TOOL_DESC = (
    "<tool>\n"
    "Tool Name: {name_for_model}\n"
    "Description: {description_for_model}\n"
    "Usage: This tool lets the agent interact with the {name_for_human} API.\n"
    "Parameters (JSON Schema): {parameters}\n"
    "{args_format}\n"
    "</tool>"
)



# PROMPT_REACT = """You are an intelligent agent that can interact with tools to answer complex questions. Below are the available tools:

# {tool_descs}

# Instructions:
# You starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). 
# The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.
# The tool you can should use is one of the following: {tool_names}.

# Example response:
# <think> thinking process here </think>
# <tool_call>
# {{"name": "tool name here", "parameters": {{"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}}}
# </tool_call>
# <tool_response>
# tool_response here
# </tool_response>
# <think> thinking process here </think>
# <tool_call>
# {{"name": "another tool name here", "arguments": {{...}}}}
# </tool_call>
# <tool_response>
# tool_response here
# </tool_response>
# (more thinking processes, tool calls and tool responses here)
# <think> thinking process here </think>
# <answer> answer here </answer>


# Question: {query}
# """

ASSISTANT = "assistant"
TOOL = "tool"
USER = "user"
SYSTEM = "system"


def prepend_react_prompt(tool_metadata_map, question: str, prompt_type: str = "original") -> str:
        tool_descs = []
        for function in tool_metadata_map.values():
            name = function.get('name', None)
            name_for_human = function.get('name_for_human', name)
            name_for_model = function.get('name_for_model', name)
            assert name_for_human and name_for_model
            args_format = function.get('args_format', '')
            tool_descs.append(
                TOOL_DESC.format(name_for_human=name_for_human,
                                 name_for_model=name_for_model,
                                 description_for_model=function['description'],
                                 parameters=json.dumps(function['parameters'], ensure_ascii=False),
                                 args_format=args_format).rstrip())
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_name for tool_name in tool_metadata_map)
        prompt = PROMPT_MAP[prompt_type].format(
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=question,
        )
        return prompt

class WebResearchRLDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor, tool_metadata_map, prompt_fn=None):
        super().__init__(data_files, tokenizer, config, processor=None)  # 跳过 multimodal/tokenizer
        self.tool_metadata_map = tool_metadata_map
        self.prompt_fn = prompt_fn or prepend_react_prompt

        print("[🔍 DEBUG] WebResearchRLDataset initialized with tool_metadata_map:", self.tool_metadata_map)

    def repeat_data(self, num_repeats):
        """
        Repeat the dataset for a given number of times.
        """
        import datasets
        self.dataframe = datasets.concatenate_datasets([self.dataframe] * num_repeats)

        print(f"[🔍 DEBUG] Data repeated {num_repeats} times, new length: {len(self.dataframe)}")

    def filter_by_source(self, source_name):
        """
        Filter the dataset by a specific source name.
        """
        self.dataframe = self.dataframe.filter(lambda ex: ex["data_source"] == source_name)

    def __getitem__(self, item):
        row_dict = self.dataframe[item]
        question = row_dict.get("question")
        assert question is not None, "Missing 'question' in data row"

        row_dict["raw_prompt"] = [
            {"role": "user", "content": self.prompt_fn(self.tool_metadata_map, question, prompt_type=self.config.web_research.prompt_type)},
        ]

        # we set a dummy_input_ids 
        raw_prompt = self.tokenizer.apply_chat_template(row_dict["raw_prompt"], tokenize=False, add_generation_prompt=True) 
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["dummy_input_ids"] = input_ids[0]
        row_dict["dummy_attention_mask"] = attention_mask[0]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})

       
        row_dict["index"] = index
        row_dict["interaction_kwargs"] = interaction_kwargs
        
        return row_dict