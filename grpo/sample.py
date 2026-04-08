import torch
import torch.nn.functional as F


@torch.no_grad()
def rollout_batch(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    prompt_texts,
    num_samples_per_prompt=8,
    max_new_tokens=20,
    temperature=1.0,
    top_p=0.95,
    eos_token_id=None,
):
    """
    对一个 batch 的 prompt 做 rollout 采样。
    
    输入:
        input_ids:      [B, L]
        attention_mask: [B, L]
        prompt_texts:   list[str], 长度=B

    输出:
        rollout_groups: list[dict], 长度=B
            每个元素:
            {
                "prompt_text": str,
                "prompt_input_ids": list[int],
                "responses": [
                    {
                        "response_ids": list[int],
                        "response_text": str,
                        "old_log_probs": list[float],
                    },
                    ...
                ]
            }
    """
    model.eval()
    device = input_ids.device

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    batch_size = input_ids.size(0)
    rollout_groups = []

    for b in range(batch_size):
        # 取出单个 prompt
        prompt_ids = input_ids[b]
        prompt_mask = attention_mask[b]

        # 去掉左侧 padding，只保留真实 prompt
        real_prompt_ids = prompt_ids[prompt_mask.bool()].tolist()

        group = {
            "prompt_text": prompt_texts[b],
            "prompt_input_ids": real_prompt_ids,
            "responses": [],
        }

        # 对同一个 prompt 采样 n 次
        for _ in range(num_samples_per_prompt):
            one_resp = rollout_one_sample(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=real_prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
                device=device,
            )
            group["responses"].append(one_resp)

        rollout_groups.append(group)

    return rollout_groups



@torch.no_grad()
def rollout_one_sample(
    model,
    tokenizer,
    prompt_input_ids,
    max_new_tokens=20,
    temperature=1.0,
    top_p=0.95,
    eos_token_id=None,
    device="cuda",
):
    """
    对单个 prompt 采样一条 response，并保存 old_log_probs。

    输出:
        {
            "response_ids": list[int],
            "response_text": str,
            "old_log_probs": list[float],
        }
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # 当前完整上下文，一开始就是 prompt
    generated_ids = torch.tensor([prompt_input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(generated_ids, device=device)

    response_ids = []
    old_log_probs = []

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            use_cache=False,   # 第一版先别加 cache，逻辑更清楚
        )

        # 取最后一个位置的 logits，预测“下一个 token”
        next_token_logits = outputs.logits[:, -1, :]   # [1, V]

        # temperature
        if temperature is not None and temperature > 0:
            next_token_logits = next_token_logits / temperature

        # top-p 过滤
        filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)

        # 转成 log_probs
        next_token_log_probs = F.log_softmax(filtered_logits, dim=-1)  # [1, V]
        next_token_probs = torch.exp(next_token_log_probs)             # [1, V]

        # 采样一个 token
        sampled_token = torch.multinomial(next_token_probs, num_samples=1)  # [1, 1]
        sampled_token_id = sampled_token.item()

        # 记录这个被采样 token 的 log prob
        sampled_log_prob = next_token_log_probs[0, sampled_token_id].item()

        response_ids.append(sampled_token_id)
        old_log_probs.append(sampled_log_prob)

        # 拼回上下文，继续生成下一步
        generated_ids = torch.cat([generated_ids, sampled_token], dim=1)
        new_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        # 遇到 EOS 就结束
        if eos_token_id is not None and sampled_token_id == eos_token_id:
            break

    # 这里可以选择是否去掉末尾 eos
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return {
        "response_ids": response_ids,
        "response_text": response_text,
        "old_log_probs": old_log_probs,
    }


def top_p_filtering(logits, top_p=0.95):
    """
    logits: [1, V]
    返回做完 nucleus sampling 过滤后的 logits
    """
    if top_p is None or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 标记需要移除的位置
    sorted_indices_to_remove = cumulative_probs > top_p

    # 保留第一个超过 top_p 的 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    filtered_logits = logits.clone()
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    filtered_logits[:, indices_to_remove] = float("-inf")

    return filtered_logits