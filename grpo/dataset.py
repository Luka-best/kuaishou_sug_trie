import json
import torch 
from torch.utils.data import Dataset
from prompts import prompt_gen

class KuaiRSQwenGRPODataset(Dataset):
    def __init__(
        self,
        jsonl_path, 
        tokenizer,
        max_prompt_length=1024,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def _build_prompt(self, x):
        caption = (x.get("caption") or "").strip()
        ocr_cover = (x.get("ocr_cover") or "").strip()
        hetu_tag = (x.get("hetu_tag") or "").strip()
        entity = (x.get("entity") or "").strip()
        other_category = (x.get("other_category") or "").strip()
        body_info = (x.get("body_info") or "").strip()
        prompt = prompt_gen.format(caption=caption, ocr_cover=ocr_cover, hetu_tag=hetu_tag, entity=entity, other_category=other_category, body_info=body_info)
        return prompt

    def __getitem__(self, idx):
        x = self.samples[idx]
        prompt = self._build_prompt(x)

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )

        attention_mask = [1] * len(prompt_ids)

        return {
            "prompt_text": prompt,
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
        }


class GRPOPromptCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                self.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer has no pad_token_id and no eos_token_id.")

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, attention_mask, prompt_texts = [], [], []

        for f in features:
            l = len(f["input_ids"])
            pad_len = max_len - l
            input_ids.append([self.pad_token_id] * pad_len + f["input_ids"])
            attention_mask.append([0] * pad_len + f["attention_mask"])
            prompt_texts.append(f["prompt_text"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_texts": prompt_texts
        }



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