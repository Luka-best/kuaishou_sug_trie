import torch
import torch.nn.functional as F

def compute_grpo_loss(model, batch, clip_eps=0.2):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # [B, L, V]

    # ===== shift，对齐 causal LM =====
    shift_logits = logits[:, :-1, :]              # [B, L-1, V]
    shift_labels = input_ids[:, 1:]               # [B, L-1]
    shift_response_mask = response_mask[:, 1:]    # [B, L-1]
    shift_old_log_probs = old_log_probs[:, 1:]    # [B, L-1]

    # 新策略下，所有位置对真实 token 的 log prob
    log_probs = F.log_softmax(shift_logits, dim=-1)   # [B, L-1, V]
    new_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [B, L-1]

    # ratio = exp(new - old)
    ratio = torch.exp(new_log_probs - shift_old_log_probs)  # [B, L-1]

    # response-level advantage -> token-level broadcast
    adv = advantages.unsqueeze(1)  # [B, 1]

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv

    token_loss = -torch.min(surr1, surr2)  # [B, L-1]

    # 只在 response token 上算 loss
    token_loss = token_loss * shift_response_mask

    denom = shift_response_mask.sum().clamp_min(1.0)
    loss = token_loss.sum() / denom

    return loss