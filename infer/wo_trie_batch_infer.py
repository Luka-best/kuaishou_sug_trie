import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SFTDataCollator, KuaiRSQwenSFTDataset

model_dir = sys.argv[1]
store = sys.argv[2]

MODEL_PATH = f"./outputs/{model_dir}"
TEST_PATH = "./data/test.jsonl"

batch_size = 128
max_new_tokens = 20


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    base_name = os.path.basename(MODEL_PATH)

    test_dataset = KuaiRSQwenSFTDataset(TEST_PATH, tokenizer, "test")
    data_collator = SFTDataCollator(tokenizer, padding_side = "left")

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
        else:
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,                 # greedy
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # prompt_lens = attention_mask.sum(dim=1).tolist()
        input_len = input_ids.shape[1]
        B = input_ids.size(0)

        for b in range(B):
            sample = test_dataset[global_idx]
            
            gen_ids = sequences[b, input_len:].tolist()
            pred_query = tokenizer.decode(gen_ids, skip_special_tokens=True)

            real_prompt_ids = input_ids[b][attention_mask[b] == 1].tolist()
            prompt_text = tokenizer.decode(real_prompt_ids, skip_special_tokens=True)

            gen_status = (
                len(gen_ids) > 0 and gen_ids[-1] == tokenizer.eos_token_id
            )

            results.append({
                "idx": global_idx,
                "gt_query": sample.get("answer", None),
                "prompt_text": prompt_text,
                "pred_query": pred_query,
                "pred_ids": str(gen_ids),
                "gen_status": gen_status,
                "is_none": False,
            })

            global_idx += 1

    df = pd.DataFrame(results)
    out_path = f"outputs/{store}_results_{base_name}.csv"
    df.to_csv(out_path, index=False)

    print(df.head())
    print(f"saved to {out_path}")