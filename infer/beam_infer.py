import torch
import torch.nn.functional as F
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import pandas as pd
from tqdm import tqdm
from trie import load_trie
from dataset import SFTDataCollator, KuaiRSQwenSFTDataset

model_dir = sys.argv[1]
store = sys.argv[2]

MODEL_PATH = f"./outputs/{model_dir}/checkpoint-7813"
TRIE_PATH = "./data/query_trie.pkl"
TEST_PATH = "./data/test.jsonl"
threshold_global = 0.2
threshold_token = 0.05


@torch.no_grad()
def custom_decode_beam_single_batched(
    model,
    tokenizer,
    input_ids,
    device,
    trie,
    attention_mask=None,
    max_new_tokens=20,
    beam_size=4,
    length_penalty=1.0,
):
    """
    单条样本，beam search，但当前所有 beam 一起 batch forward
    input_ids: [1, T]
    """
    model.eval()

    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    prompt_len = attention_mask.sum(dim=1).item()

    beams = [{
        "generated": input_ids.clone(),              # [1, cur_len]
        "attention_mask": attention_mask.clone(),    # [1, cur_len]
        "score": 0.0,                                # 累积 log prob
        "finished": False,
        "token_probs": [],
        "past_key_values": None,                     
        "last_token": None,   
    }]

    for step in range(max_new_tokens):
        all_candidates = []

        # 1) 先把未完成的 beam 和已完成的 beam 分开
        active_beams = []
        for beam in beams:
            if beam["finished"]:
                all_candidates.append(beam)   # finished 的 beam 不再扩展，但保留
            else:
                active_beams.append(beam)

        # 如果没有可扩展的 beam，直接停
        if not active_beams:
            break

        # 2) 把 active_beams pad 成 batch，一次 forward
        max_len = max(b["generated"].shape[1] for b in active_beams)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        batch_input_ids = []
        batch_attention_mask = []

        for beam in active_beams:
            cur_ids = beam["generated"][0]          # [cur_len]
            cur_mask = beam["attention_mask"][0]    # [cur_len]
            cur_len = cur_ids.shape[0]
            pad_len = max_len - cur_len

            # left padding
            padded_ids = torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=cur_ids.dtype, device=device),
                cur_ids
            ], dim=0)

            padded_mask = torch.cat([
                torch.zeros((pad_len,), dtype=cur_mask.dtype, device=device),
                cur_mask
            ], dim=0)

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_mask)

        batch_input_ids = torch.stack(batch_input_ids, dim=0)           # [N, max_len]
        batch_attention_mask = torch.stack(batch_attention_mask, dim=0) # [N, max_len]

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )
        next_token_logits = outputs.logits[:, -1, :]   # [N, V]

        # 3) 对 batch 中每条 beam 单独做 Trie 约束和 top-k 扩展
        for row, beam in enumerate(active_beams):
            row_logits = next_token_logits[row:row+1]   # [1, V]

            prefix_ids = beam["generated"][0, prompt_len:].tolist()
            legal_next = trie.get_next_tokens(prefix_ids)

            if not legal_next:
                # 当前 beam 没有合法后继，丢掉
                continue

            legal_next_tensor = torch.tensor(
                legal_next,
                dtype=torch.long,
                device=row_logits.device
            )

            masked_logits = torch.full_like(row_logits, float("-inf"))
            masked_logits[:, legal_next_tensor] = row_logits[:, legal_next_tensor]

            log_probs = F.log_softmax(masked_logits, dim=-1)             # [1, V]
            legal_log_probs = log_probs[0, legal_next_tensor]            # [num_legal]

            k = min(beam_size, legal_next_tensor.numel())
            topk_log_probs, topk_idx = torch.topk(legal_log_probs, k=k)

            for j in range(k):
                token_id = legal_next_tensor[topk_idx[j]].item()
                token_log_prob = topk_log_probs[j].item()
                token_prob = topk_log_probs[j].exp().item()

                next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)

                new_generated = torch.cat([beam["generated"], next_token], dim=1)
                new_mask = torch.cat([
                    beam["attention_mask"],
                    torch.ones((1, 1), dtype=beam["attention_mask"].dtype, device=device)
                ], dim=1)

                new_beam = {
                    "generated": new_generated,
                    "attention_mask": new_mask,
                    "score": beam["score"] + token_log_prob,
                    "finished": (token_id == tokenizer.eos_token_id),
                    "token_probs": beam["token_probs"] + [token_prob],
                }
                all_candidates.append(new_beam)

        if not all_candidates:
            break
        
        def rank_score(x):
            gen_len = x["generated"].shape[1] - prompt_len
            if gen_len <= 0:
                return -1e30
            return x["score"] / (gen_len ** length_penalty)

        # 4) 从所有候选里选 top beam_size
        all_candidates = sorted(all_candidates, key=rank_score, reverse=True)
        beams = all_candidates[:beam_size]

        # 如果保留下来的 beam 都 finished，就提前停
        if all(b["finished"] for b in beams):
            break

    # 5) 最终从剩余 beams 里选最优
    beams = sorted(
        beams,
        key=lambda x: x["score"] / max((x["generated"].shape[1] - prompt_len) ** length_penalty, 1.0),
        reverse=True
    )

    all_ids = [beam["generated"][0, prompt_len:].tolist() for beam in beams]
    all_probs = [beam["token_probs"] for beam in beams]

    return all_ids, all_probs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)
    trie = load_trie(TRIE_PATH)
    base_name = os.path.basename(MODEL_PATH)

    test_dataset = KuaiRSQwenSFTDataset(TEST_PATH, tokenizer, "test")
    data_collator = SFTDataCollator(tokenizer)

    results = []

    for i in tqdm(range(len(test_dataset))):
        sample = test_dataset[i]

        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0)

        if "attention_mask" in sample:
            attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long).unsqueeze(0)
        else:
            attention_mask = None

        all_ids, all_probs = custom_decode_beam_single_batched(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            trie=trie,
            max_new_tokens=20,
        )
        for idx, (gen_ids, gen_probs) in enumerate(zip(all_ids, all_probs)):
            if gen_ids is None:
                pred_query = None
                min_prob = None
                avg_prob = None
                pass_filter = False
                gen_status = False
            else:
                last_token = gen_ids[-1]
                gen_status = last_token == tokenizer.eos_token_id
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
                "idx": i,
                "beam_idx":idx,
                "gt_query": sample.get("answer", None),
                "prompt_text": tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True),
                "pred_query": pred_query,
                "pred_ids": str(gen_ids) if gen_ids is not None else None,
                "pred_probs": str(gen_probs) if gen_probs is not None else None,
                "min_prob": min_prob,
                "avg_prob": avg_prob,
                "pass_filter": pass_filter,
                "gen_status": gen_status,
                "is_none": gen_ids is None,
            })

    df = pd.DataFrame(results)
    df.to_csv(f"outputs/{store}_results_{base_name}.csv", index=False)
    print(df.head())
    print(f"saved to outputs/{store}_results_{base_name}.csv")
