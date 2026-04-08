import sys
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from dataset import GRPOPromptCollator, KuaiRSQwenGRPODataset, collate_policy_update_batch
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from sample import rollout_batch
from prompts import prompt_judge
from model_helper import run_v3
from tools import flatten_rollout_groups
from loss import compute_grpo_loss

TRAIN_PATH = "./data/train.jsonl"
MODEL_PATH = "./outputs/qwen_nttp_logsumexp_sft_prompt_v2/checkpoint-7813"

def compute_group_advantages(rewards, eps=1e-6):
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    mean = rewards_t.mean()
    std = rewards_t.std(unbiased=False)
    adv = (rewards_t - mean) / (std + eps)
    return adv.tolist()





if __name__ == "__main__":
    output_dir = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model.train()


    train_dataset = KuaiRSQwenGRPODataset(TRAIN_PATH ,tokenizer)
    data_collator = GRPOPromptCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4, 
        shuffle=True,
        collate_fn=data_collator,
    )
    optimizer = AdamW(model.parameters(), lr=1e-6)
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_texts = batch["prompt_texts"]

        # 1. rollout
        model.eval()
        rollout_groups = rollout_batch(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_texts=prompt_texts,
            num_samples_per_prompt=8,
            max_new_tokens=20,
            temperature=1.0,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )


        # 2. judge reward
        for group in rollout_groups:
            rewards = []
            for resp in group["responses"]:
                prompt = prompt_judge.format(context=group["prompt_text"], sug_words=resp["response_text"])
                status_code = -1
                score = None
                while status_code!=200 or not score:
                    score, status_code = run_v3(prompt)
                score = int(score)
                rewards.append(score)

        # 3. advantage
            advantages = compute_group_advantages(rewards)

            for resp, reward, adv in zip(group["responses"], rewards, advantages):
                resp["reward"] = reward
                resp["advantage"] = adv

        # 4. optimize
        model.train()
        rollout_samples = flatten_rollout_groups(rollout_groups)
        policy_batch = collate_policy_update_batch(
            rollout_samples,
            tokenizer,
        )
        policy_batch = {k: v.to(device) for k, v in policy_batch.items()}
        loss = compute_grpo_loss(model,policy_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step={step}, loss={loss.item():.4f}")

        



        

        

