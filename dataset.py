import json
import torch 
from torch.utils.data import Dataset
from prompts import prompt_v1, prompt_v2

class KuaiRSQwenSFTDataset(Dataset):
    """
    生成 prompt + answer 训练样本，并构造 labels:
      - prompt 区域 labels = -100
      - answer 区域 labels = token_ids
    """
    def __init__(
        self,
        jsonl_path, 
        tokenizer,
        mode,
        max_length=1200,
        max_prompt_length=1024,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # 至少要有query
                if not obj.get("query"):
                    continue
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
        prompt = prompt_v2.format(caption=caption, ocr_cover=ocr_cover, hetu_tag=hetu_tag, entity=entity, other_category=other_category, body_info=body_info)
        return prompt

    def __getitem__(self, idx):
        x = self.samples[idx]
        query = (x.get("query") or "").strip()

        prompt = self._build_prompt(x)
        answer = query  # 先直接用query作为answer

        # ====== 1) 分开tokenize，方便构造labels mask ======
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )

        # answer末尾建议带 eos（如果 tokenizer 有 eos_token_id）
        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.max_length - len(prompt_ids) - 1),
        )

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        # ====== 2) 拼接并截断 ======
        input_ids = prompt_ids + answer_ids if self.mode != "test" else prompt_ids
        input_ids = input_ids[: self.max_length]

        attention_mask = [1] * len(input_ids)

        # ====== 3) labels: prompt部分 -100, answer部分 GT token ======
        # 注意如果截断了，要同步截断labels
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]

        # 可选：返回一些调试信息
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer": query
            # "meta": {
            #     "photo_id": x.get("photo_id", ""),
            #     "query": query,
            #     "prompt_text": prompt,   # 调试用，正式训练可去掉减少开销
            # }
        }


class SFTDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8, padding_side="right"):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding_side = padding_side

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

        input_ids, attention_mask, labels = [], [], []

        for f in features:
            l = len(f["input_ids"])
            pad_len = max_len - l

            # input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            # attention_mask.append(f["attention_mask"] + [0] * pad_len)
            # labels.append(f["labels"] + [-100] * pad_len)

            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_len + f["input_ids"])
                attention_mask.append([0] * pad_len + f["attention_mask"])
                labels.append([-100] * pad_len + f["labels"])
            else:
                input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
                attention_mask.append(f["attention_mask"] + [0] * pad_len)
                labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }