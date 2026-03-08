# -------------------- 07/08 chatgpt 的函数，需要07/09 检查是否正确 --------------------

def postprocess(
    self,
    batch: DataProto,
    batch_conversations: List[List[Dict[str, str]]],
    n: int
) -> DataProto:
    """
    多轮 SFT 后处理逻辑，支持 tool 调用与 prompt/response 分离 tokenization。

    Args:
        batch: 原始 DataProto
        batch_conversations: 每个 raw_prompt 扩展为 n 个 conversation 的完整对话（即 bsz * n）
        n: 每个原始样本生成 n 个 response

    Returns:
        DataProto：包含拼接后的 input_ids、attention_mask、loss_mask 等
    """
    pad_token_id = self.tokenizer.pad_token_id or 0

    prompt_ids_list = []
    response_ids_list = []
    response_loss_masks = []
    response_attention_masks = []

    for conversation in batch_conversations:
        prompt_ids, response_ids, loss_mask, attention_mask = self._process_full_conversation_tokens(
            conversation, tools=self.tool_schemas, enable_thinking=True
        )
        prompt_ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))
        response_ids_list.append(torch.tensor(response_ids, dtype=torch.long))
        response_loss_masks.append(torch.tensor(loss_mask, dtype=torch.long))
        response_attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))

    # 左 padding prompts
    def left_pad(sequences, pad_id):
        max_len = max(seq.size(0) for seq in sequences)
        padded, masks = [], []
        for seq in sequences:
            pad_len = max_len - seq.size(0)
            padded_seq = torch.cat([torch.full((pad_len,), pad_id, dtype=seq.dtype), seq])
            attn_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones_like(seq)])
            padded.append(padded_seq)
            masks.append(attn_mask)
        return torch.stack(padded), torch.stack(masks)

    # 右 padding responses
    def right_pad(sequences, pad_val):
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_val)

    prompts, prompt_attn_masks = left_pad(prompt_ids_list, pad_token_id)
    responses = right_pad(response_ids_list, pad_token_id)
    response_attn_masks = right_pad(response_attention_masks, 0)
    loss_masks = right_pad(response_loss_masks, 0)

    # 拼接 input_ids
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat([prompt_attn_masks, response_attn_masks], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    # 构造 response_mask：将 tool_call 部分 mask 掉
    response_mask = self._mask_out_tools_calling_tokens(
        batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0),
        batch_conversations,
        responses,
        response_attn_masks,
    )

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=len(input_ids),
    )

    num_turns = np.array([len(convo) for convo in batch_conversations], dtype=np.int32)
    return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})