import pickle
import sys
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from dataset import SFTDataCollator, KuaiRSQwenSFTDataset
from loss import compute_total_loss_with_nttp
from trie import load_trie

TRIE_PATH = "./data/query_trie.pkl"
TRAIN_PATH = "./data/train.jsonl"
EVAL_PATH = "./data/val.jsonl"
MODEL_PATH = "./models/Qwen3-1.7B"

class NTTPTrainer(Trainer):
    def __init__(self, *args, trie=None, alpha_nttp=0.1, use_nttp=True, loss_type="logsumexp", **kwargs):
        super().__init__(*args, **kwargs)
        self.trie = trie
        self.alpha_nttp = alpha_nttp
        self.use_nttp = use_nttp
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.use_nttp:
            total_loss, logs, outputs = compute_total_loss_with_nttp(
                model=model,
                batch=inputs,
                trie=self.trie,
                alpha_nttp=self.alpha_nttp,
                loss_type=self.loss_type,
            )

            if model.training and self.state.global_step % self.args.logging_steps == 0:
                self.log(logs)

            if return_outputs:
                return total_loss, outputs
            else:
                return total_loss
        else:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=inputs["labels"],
            )
            loss = outputs.loss

            if model.training and self.state.global_step % self.args.logging_steps == 0:
                self.log({"ntp_loss": loss.detach().float().item()})

            if return_outputs:
                return loss, outputs
            else:
                return loss



if __name__ == "__main__":
    output_dir = sys.argv[1]
    loss_type = output_dir.split("nttp_")[1].split("_sft")[0]
    use_nttp = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    trie = load_trie(TRIE_PATH)

    train_dataset = KuaiRSQwenSFTDataset(TRAIN_PATH ,tokenizer,"train")
    eval_dataset = KuaiRSQwenSFTDataset(EVAL_PATH ,tokenizer,"eval")
    data_collator = SFTDataCollator(tokenizer, padding_side="right")
    alpha_nttp = 0.1

    model.config.use_cache = False

    training_args = TrainingArguments(
        # output_dir="./outputs/qwen_wo_nttp_sft",
        output_dir = output_dir,
        overwrite_output_dir=True,

        # ===== 训练轮数 / 步数 =====
        num_train_epochs=1,                 # 先跑1轮看loss和流程
        # max_steps=1000,                   # 如果你想按步数控制，开这个并注释掉epochs也行

        # ===== batch相关 =====
        per_device_train_batch_size=4,      # 先保守一点，1.7B + 自定义loss会慢
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,      # 等效batch变大（2*4=16）
        
        # ===== 学习率 =====
        learning_rate=5e-5,                 # 全参SFT常见起点（LoRA可更大如1e-4~2e-4）
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # lr_scheduler_type="constant_with_warmup",

        # ===== 日志 / 保存 / 评估 =====
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",        # 如果暂时不想eval可改为 "no"
        save_strategy="steps",
        save_total_limit=3,
        
        # ===== 精度 =====
        bf16=True,                          # 如果你的GPU支持bf16（A800/A100一般支持）
        fp16=False,                         # bf16和fp16二选一
        # 如果bf16不支持，改成：bf16=False, fp16=True

        # ===== 性能相关 =====
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,        # 显存友好，速度会慢一点
        # torch_compile=False,              # 新环境可试，先别开，先跑通

        # ===== 序列 / 日志 =====
        report_to="none",                   # 不上wandb先关掉
        remove_unused_columns=False,        # 自定义collator/inputs时建议关掉，防止字段被删
        logging_first_step=True,

        # ===== 训练稳定性 =====
        max_grad_norm=1.0,

        # ===== 评估/预测时生成无关 =====
        do_train=True,
        do_eval=True,                       # 没有eval_dataset就改False
    )

    trainer = NTTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        trie=trie,
        alpha_nttp=alpha_nttp, # 建议先小一点
        use_nttp=use_nttp,
        loss_type=loss_type,
    )
    
    trainer.train()


