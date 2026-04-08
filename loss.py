import torch
import torch.nn.functional as F

def _tensor_to_log_scalar(x):
    # x 可能是标量，也可能是多卡聚合后的向量
    if not isinstance(x, torch.Tensor):
        return float(x)
    x_det = x.detach()
    if x_det.numel() == 1:
        return float(x_det.cpu())
    return float(x_det.mean().cpu())  # 多元素时取mean用于日志显示

def compute_nttp_loss_paper_style(
    logits,          # [B, L, V] batch length vocabulary_size
    labels,          # [B, L], 非query位置为-100
    trie,
    loss_type,
):
    """
    论文风格先跑通版 NTTP：
        L_nttp = - sum_{j in S_t}(log(p_{t,j}))
    只看 Trie 合法 next token 集合的概率，不显式记录其他 token 的loss。
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    device = logits.device
    B, L, V = logits.shape

    # 数值稳定：先做 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]

    losses = []
    n_positions = 0
    n_miss = 0  # 统计Trie前缀未命中的位置（调试用）

    for b in range(B):
        # 用 teacher forcing：prefix 来自真实 labels（只在 query 区域）
        prefix = []

        for pos in range(L):
            tgt = labels[b, pos].item()

            # 非query区域（prompt/pad）跳过，不更新prefix
            if tgt == -100:
                continue

            # 当前prefix下合法 next token（children）
            legal_next = trie.get_next_tokens(prefix)

            if legal_next:
                legal_next_tensor = torch.tensor(legal_next, dtype=torch.long, device=device)

                # 当前位置在全词表上的 log_probs
                lp = log_probs[b, pos]  # [V]

                # 只取合法集合
                legal_lp = lp.index_select(0, legal_next_tensor)  # [K]

                # sum 做法
                if loss_type == "sum":
                    log_sum_p_legal = torch.sum(legal_lp, dim=0)
                    losses.append(-log_sum_p_legal)

                # mean 做法
                if loss_type == "mean":
                    log_mean_p_legal = torch.mean(legal_lp, dim=0)
                    losses.append(-log_mean_p_legal)

                # logsumexp 做法
                if loss_type == "logsumexp":
                    log_sum_exp_p_legal = torch.logsumexp(legal_lp, dim=0)
                    losses.append(-log_sum_exp_p_legal)

                n_positions += 1
            else:
                # 前缀不在Trie里（通常是tokenizer不一致/数据清洗差异）
                n_miss += 1

            # teacher forcing: 用真实token更新前缀
            prefix.append(int(tgt))

    if len(losses) == 0:
        loss_nttp = torch.tensor(0.0, device=device)
    else:
        loss_nttp = torch.stack(losses).mean()

    stats = {
        "nttp_positions": n_positions,
        "nttp_miss": n_miss,
        "nttp_miss_rate": (n_miss / max(n_positions + n_miss, 1)),
    }
    return loss_nttp, stats



def compute_total_loss_with_nttp(model, batch, trie, alpha_nttp=0.1, loss_type="logsumexp"):
    """
    batch 需要包含:
      input_ids, attention_mask, labels
    labels中只有query区域是token id，其余是-100（标准SFT做法）
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask", None),
        labels=batch["labels"],  # HF会自动算标准CE (NTP loss)
    )

    loss_ntp = outputs.loss
    logits = outputs.logits  # [B, L, V]

    loss_nttp, nttp_stats = compute_nttp_loss_paper_style(
        logits=logits,
        labels=batch["labels"],
        trie=trie,
        loss_type=loss_type,
    )

    alpha_nttp_loss = alpha_nttp * loss_nttp

    total_loss = loss_ntp + alpha_nttp_loss

    logs = {
        "loss_total": total_loss.detach().cpu().item(),
        "loss_ntp": loss_ntp.detach().cpu().item(),
        "raw_loss_nttp": loss_nttp.detach().cpu().item(),
        "alpha_loss_nttp": alpha_nttp_loss.detach().cpu().item(),
        **nttp_stats,
    }
    return total_loss, logs, outputs