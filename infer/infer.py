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

model_dir = sys.argv[1]
store = sys.argv[2]

MODEL_PATH = f"./outputs/{model_dir}/checkpoint-7813"
TRIE_PATH = "./data/query_trie.pkl"
TEST_PATH = "./data/test.jsonl"
threshold_global = 0.2
threshold_token = 0.05

@torch.no_grad()
def custom_decode(
    model,
    tokenizer,
    input_ids,
    device,
    trie,
    attention_mask=None,
    max_new_tokens=20,
):
    model.eval()

    input_ids = input_ids.to(device)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    generated = input_ids
    prompt_len = input_ids.shape[1]

    generate_probs = []

    for step in range(max_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
        )

        # [B, V]，当前最后一个位置预测“下一个 token”的分布
        next_token_logits = outputs.logits[:, -1, :]

        # ===== 你自己的特殊处理写在这里 =====
        next_token_logits = process_logits(
            next_token_logits,
            generated[0, prompt_len:].tolist(),
            trie,
        )
        if next_token_logits is None:
            return None, None

        # 转概率
        probs = F.softmax(next_token_logits, dim=-1)
        # max_prob = probs.max()
        # if max_prob<threshold_token:
        #     break
        # generate_probs.append(max_prob)

        next_token = torch.argmax(probs, dim=-1, keepdim=True)   # [1, 1]
        next_token_prob = probs.gather(1, next_token).squeeze(1) # [1]
        generate_probs.append(next_token_prob.squeeze(0))

        # greedy
        # next_token = torch.argmax(probs, dim=-1, keepdim=True)   # [B, 1]

        # 拼接回输入
        generated = torch.cat([generated, next_token], dim=1)

        # attention_mask 也要同步补 1
        next_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

        # 如果 batch=1，可以这样简单停
        if next_token[0, 0].item() == tokenizer.eos_token_id:
            break
    # avg_prob = torch.stack(generate_probs).mean()    
    # if avg_prob < threshold_global:
    #     return None
    # if generated[0, -1].item() != tokenizer.eos_token_id:
    #     return None, None
    return generated[0, prompt_len:-1].tolist(), [p.item() for p in generate_probs]

def process_logits(next_token_logits, prefix_ids, trie):
    """
    next_token_logits: [1, vocab_size]
    prefix_ids: list[int]，当前已生成输出前缀
    trie: 你的 Trie
    """
    legal_next = trie.get_next_tokens(prefix_ids)   # list[int]
    if not legal_next:
        return None

    masked_logits = torch.full_like(next_token_logits, float("-inf"))
 
    legal_next_tensor = torch.tensor(
        legal_next,
        dtype=torch.long,
        device=next_token_logits.device
    )

    masked_logits[:, legal_next_tensor] = next_token_logits[:, legal_next_tensor]
    return masked_logits
    


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

        gen_ids, gen_probs = custom_decode(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            trie=trie,
            max_new_tokens=20,
        )
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

    

