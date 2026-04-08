import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import pandas as pd
from tqdm import tqdm
from trie import load_trie
from dataset import SFTDataCollator, KuaiRSQwenSFTDataset
from torch.utils.data import DataLoader


model_dir = sys.argv[1]
store = sys.argv[2]
use_raw = sys.argv[3].lower() == "true"

MODEL_PATH = f"./outputs/{model_dir}"
TRIE_PATH = "./data/query_trie.pkl"
TEST_PATH = "./data/test.jsonl"
threshold_global = 0.2
threshold_token = 0.05

@torch.no_grad()
def custom_decode_batch(
    model,
    tokenizer,
    input_ids,
    device,
    trie,
    attention_mask=None,
    max_new_tokens=20,
):
    model.eval()

    input_ids = input_ids.to(device)   # [B, T]
    B = input_ids.size(0)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    generated = input_ids
    input_len = input_ids.size(1)   # batch pad 后的统一输入长度
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    failed = torch.zeros(B, dtype=torch.bool, device=device)

    # 每个样本保存每步选中的概率
    generate_probs = [[] for _ in range(B)]

    for step in range(max_new_tokens):
        if step == 0:
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                use_cache=True,
            )
        else:
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values

        # [B, V]
        next_token_logits = outputs.logits[:, -1, :]

        # Trie 约束
        masked_logits = process_logits_batch(
            next_token_logits=next_token_logits,
            generated=generated,
            input_len=input_len,
            trie=trie,
            finished=finished,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 某些样本可能整行都是 -inf，说明无合法后继
        invalid_rows = torch.isneginf(masked_logits).all(dim=1)

        # 标记失败，但为了 batch 能继续跑下去，给它强行填 eos
        failed = failed | invalid_rows
        masked_logits[invalid_rows, tokenizer.eos_token_id] = 0.0

        raw_probs = F.softmax(next_token_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)   # [B, V]
        next_token = torch.argmax(probs, dim=-1, keepdim=True)   # [B, 1]
        next_token_prob = probs.gather(1, next_token).squeeze(1) # [B]
        next_token_raw_prob = raw_probs.gather(1, next_token).squeeze(1)

        for b in range(B):
            if not finished[b] and not failed[b]:
                if use_raw:
                    generate_probs[b].append(next_token_raw_prob[b].item())
                else:
                    generate_probs[b].append(next_token_prob[b].item())

        generated = torch.cat([generated, next_token], dim=1)

        next_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

        # 更新 finished
        finished = finished | (next_token.squeeze(1) == tokenizer.eos_token_id)

        # 全部结束就停
        if finished.all():
            break

    # 组织每个样本结果
    all_gen_ids = []
    all_gen_probs = []

    for b in range(B):
        if failed[b]:
            all_gen_ids.append(None)
            all_gen_probs.append(None)
            continue

        full_seq = generated[b].tolist()
        gen_part = full_seq[input_len:]

        all_gen_ids.append(gen_part)
        all_gen_probs.append(generate_probs[b])

    return all_gen_ids, all_gen_probs, failed.tolist()

def process_logits_batch(next_token_logits, generated, input_len, trie, finished, eos_token_id):
    """
    next_token_logits: [B, V]
    generated: [B, T]
    input_len: list[int] or tensor[B]
    finished: [B] bool tensor，表示该样本是否已经结束
    """
    B, V = next_token_logits.shape
    masked_logits = torch.full_like(next_token_logits, float("-inf"))

    for b in range(B):
        # 已结束样本：只允许继续出 eos，防止后面再乱生成
        if finished[b]:
            masked_logits[b, eos_token_id] = 0.0
            continue

        prefix_ids = generated[b, input_len:].tolist()
        legal_next = trie.get_next_tokens(prefix_ids)

        if not legal_next:
            # 没有合法后继，整个这一行保持 -inf
            # 后面外层可以据此判定该样本失败
            continue

        legal_next_tensor = torch.tensor(
            legal_next,
            dtype=torch.long,
            device=next_token_logits.device
        )
        masked_logits[b, legal_next_tensor] = next_token_logits[b, legal_next_tensor]

    return masked_logits
    


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)
    trie = load_trie(TRIE_PATH)
    base_name = os.path.basename(MODEL_PATH)

    test_dataset = KuaiRSQwenSFTDataset(TEST_PATH, tokenizer, "test")
    data_collator = SFTDataCollator(tokenizer, padding_side = "left")
    
    batch_size = 64
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    results = []
    global_idx = 0

    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        all_gen_ids, all_gen_probs, failed_list = custom_decode_batch(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            trie=trie,
            max_new_tokens=20,
        )

        B = input_ids.size(0)

        for b in range(B):
            gen_ids = all_gen_ids[b]
            gen_probs = all_gen_probs[b]

            # 这里取回原样本，拿 answer 等字段
            sample = test_dataset[global_idx]

            if gen_ids is None:
                pred_query = None
                min_prob = None
                avg_prob = None
                pass_filter = False
                gen_status = False
            else:
                gen_status = (
                    len(gen_ids) > 0 and gen_ids[-1] == tokenizer.eos_token_id
                )
                if gen_status:
                    eos_pos = gen_ids.index(tokenizer.eos_token_id)
                    gen_ids = gen_ids[:eos_pos]
                    gen_probs = gen_probs[:-1]
                pred_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                min_prob = min(gen_probs) if len(gen_probs) > 0 else None
                avg_prob = sum(gen_probs) / len(gen_probs) if len(gen_probs) > 0 else None
                pass_filter = (
                    min_prob is not None
                    and avg_prob is not None
                    and min_prob >= threshold_token
                    and avg_prob >= threshold_global
                )

            results.append({
                "idx": global_idx,
                "gt_query": sample.get("answer", None),
                "prompt_text": tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=True),
                # real_prompt_ids = input_ids[b][attention_mask[b] == 1].tolist(),
                # "prompt_text": tokenizer.decode(real_prompt_ids, skip_special_tokens=True),
                "pred_query": pred_query,
                "pred_ids": str(gen_ids) if gen_ids is not None else None,
                "pred_probs": str(gen_probs) if gen_probs is not None else None,
                "min_prob": min_prob,
                "avg_prob": avg_prob,
                "pass_filter": pass_filter,
                "gen_status": gen_status,
                "is_none": gen_ids is None,
            })

            global_idx += 1

    df = pd.DataFrame(results)
    df.to_csv(f"outputs/{store}_results_{base_name}.csv", index=False)
    print(df.head())
    print(f"saved to outputs/{store}_results_{base_name}.csv")

    

