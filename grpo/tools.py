
import torch
import subprocess as sp

def get_section(text, tag):
    """
    从文本中提取指定标签内的内容

    Args:
        text: 源文本
        tag: XML格式的标签名，如 "content", "title" 等

    Returns:
        标签内的文本内容，如果标签不存在则返回空字符串
    """
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return ""
    return text[start + len(tag) + 2 : end].strip()


def flatten_rollout_groups(rollout_groups):
    rollout_samples = []

    for group in rollout_groups:
        prompt_ids = group["prompt_input_ids"]
        prompt_text = group["prompt_text"]

        for resp in group["responses"]:
            response_ids = resp["response_ids"]
            old_log_probs = resp["old_log_probs"]


            if len(response_ids) == 0:
                continue

            rollout_samples.append({
                "prompt_ids": prompt_ids,
                "prompt_text": prompt_text,
                "response_ids": response_ids,
                "response_text": resp["response_text"],
                "old_log_probs": old_log_probs,
                "reward": resp["reward"],
                "advantage": resp["advantage"],
            })

    return rollout_samples



def collate_policy_update_batch(samples, tokenizer, pad_to_multiple_of=8):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    max_len = 0
    for s in samples:
        full_ids = s["prompt_ids"] + s["response_ids"]
        max_len = max(max_len, len(full_ids))

    if pad_to_multiple_of is not None:
        m = pad_to_multiple_of
        max_len = ((max_len + m - 1) // m) * m

    input_ids = []
    attention_mask = []
    response_mask = []
    old_log_probs = []
    advantages = []

    for s in samples:
        prompt_ids = s["prompt_ids"]
        response_ids = s["response_ids"]
        full_ids = prompt_ids + response_ids

        pad_len = max_len - len(full_ids)

        # teacher forcing 阶段先用 right padding 更直观
        padded_ids = full_ids + [pad_token_id] * pad_len
        padded_attn = [1] * len(full_ids) + [0] * pad_len

        # prompt位置=0, response位置=1, pad位置=0
        padded_resp_mask = (
            [0] * len(prompt_ids)
            + [1] * len(response_ids)
            + [0] * pad_len
        )

        # old_log_probs 只对应 response 部分
        padded_old_lp = (
            [0.0] * len(prompt_ids)
            + s["old_log_probs"]
            + [0.0] * pad_len
        )

        input_ids.append(padded_ids)
        attention_mask.append(padded_attn)
        response_mask.append(padded_resp_mask)
        old_log_probs.append(padded_old_lp)
        advantages.append(s["advantage"])

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.float),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float),
        "advantages": torch.tensor(advantages, dtype=torch.float),
    }
    return batch
