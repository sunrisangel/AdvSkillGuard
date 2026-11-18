import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_cosine_schedule_with_warmup,
)

from defender.loss import VerdictFocusedLoss, build_weighted_labels


@dataclass
class DefenderTrainConfig:
    model_name_or_path: str
    output_dir: str = "experiments/checkpoints/defender_init"
    max_length: int = 4096
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.03
    alpha: float = 0.7
    num_workers: int = 0
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    fp16: bool = True
    save_steps: int = 500
    log_steps: int = 20


class DefenderSFTDataset(Dataset):
    """
    防御侧 SFT 数据集。

    每条数据格式建议为：
    {
        "prompt": "...完整输入Prompt...",
        "answer": "<evidence>...</evidence>\\n<verdict>恶意</verdict>"
    }
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 4096,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        prompt = item["prompt"]
        answer = item["answer"]

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        full_text = prompt + answer

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = min(len(prompt_ids), len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_length": prompt_length,
        }


class DefenderDataCollator:
    """
    动态 padding，并构造 labels。
    """

    def __init__(self, tokenizer, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
        prompt_lengths = torch.tensor(
            [x["prompt_length"] for x in batch],
            dtype=torch.long,
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        labels = build_weighted_labels(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            ignore_index=self.ignore_index,
        )

        labels[attention_mask == 0] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DefenderTrainer:
    """
    AdvSkillGuard 防御侧训练器。

    支持：
    - 初始 SFT 训练
    - 对抗训练阶段的增量训练
    - 保存与加载 checkpoint
    """

    def __init__(self, config: DefenderTrainConfig):
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" and config.fp16 else torch.float32,
        )

        self.model.to(self.device)

        self.loss_fn = VerdictFocusedLoss(
            tokenizer=self.tokenizer,
            alpha=config.alpha,
        )

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        data = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        return data

    def build_dataloader(
        self,
        data: List[Dict[str, Any]],
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = DefenderSFTDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        collator = DefenderDataCollator(self.tokenizer)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collator,
        )

    def train(
        self,
        train_data: List[Dict[str, Any]],
        valid_data: Optional[List[Dict[str, Any]]] = None,
    ):
        train_loader = self.build_dataloader(train_data, shuffle=True)

        total_update_steps = (
            len(train_loader)
            * self.config.epochs
            // self.config.gradient_accumulation_steps
        )

        warmup_steps = int(total_update_steps * self.config.warmup_ratio)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

        scaler = torch.cuda.amp.GradScaler(
            enabled=self.device == "cuda" and self.config.fp16,
        )

        global_step = 0
        self.model.train()

        for epoch in range(self.config.epochs):
            for step, batch in enumerate(train_loader):
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }

                with torch.cuda.amp.autocast(
                    enabled=self.device == "cuda" and self.config.fp16,
                ):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    loss_dict = self.loss_fn(
                        logits=outputs.logits,
                        labels=batch["labels"],
                    )

                    loss = loss_dict["loss"]
                    loss = loss / self.config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    if global_step % self.config.log_steps == 0:
                        print(
                            f"[Train] epoch={epoch} step={global_step} "
                            f"loss={loss_dict['loss'].item():.4f} "
                            f"evidence_loss={loss_dict['evidence_loss'].item():.4f} "
                            f"verdict_loss={loss_dict['verdict_loss'].item():.4f}"
                        )

                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(
                            os.path.join(
                                self.config.output_dir,
                                f"step-{global_step}",
                            )
                        )

        self.save_checkpoint(self.config.output_dir)

        if valid_data is not None:
            return self.evaluate(valid_data)

        return None

    def incremental_train(
        self,
        current_evasion_data: List[Dict[str, Any]],
        history_data: Optional[List[Dict[str, Any]]] = None,
        init_data: Optional[List[Dict[str, Any]]] = None,
        steps: int = 200,
    ):
        """
        对抗训练阶段的增量训练。

        数据组成：
        - 当前轮逃逸样本
        - 历史逃逸样本子集
        - 初始训练集子集
        """

        mixed_data = []

        mixed_data.extend(current_evasion_data)

        if history_data:
            mixed_data.extend(history_data)

        if init_data:
            mixed_data.extend(init_data)

        if not mixed_data:
            print("[Warning] 增量训练数据为空，跳过。")
            return

        loader = self.build_dataloader(mixed_data, shuffle=True)

        optimizer = AdamW(
            self.model.parameters(),
            lr=1e-5,
        )

        self.model.train()
        global_step = 0

        while global_step < steps:
            for batch in loader:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss_dict = self.loss_fn(
                    logits=outputs.logits,
                    labels=batch["labels"],
                )

                loss = loss_dict["loss"]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % self.config.log_steps == 0:
                    print(
                        f"[Incremental] step={global_step} "
                        f"loss={loss.item():.4f}"
                    )

                if global_step >= steps:
                    break

    @torch.no_grad()
    def evaluate(self, valid_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        简单验证：计算 verdict token 的预测损失。
        完整 F1/Accuracy 后续放到 evaluation/evaluate.py。
        """

        loader = self.build_dataloader(valid_data, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        total_count = 0

        for batch in loader:
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
            }

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss_dict = self.loss_fn(
                logits=outputs.logits,
                labels=batch["labels"],
            )

            total_loss += loss_dict["loss"].item()
            total_count += 1

        avg_loss = total_loss / max(total_count, 1)

        print(f"[Eval] loss={avg_loss:.4f}")

        return {
            "loss": avg_loss
        }

    def save_checkpoint(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        config_path = os.path.join(save_dir, "trainer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                self.config.__dict__,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"[Save] checkpoint saved to {save_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" and self.config.fp16 else torch.float32,
        )

        self.model.to(self.device)

        self.loss_fn = VerdictFocusedLoss(
            tokenizer=self.tokenizer,
            alpha=self.config.alpha,
        )

        print(f"[Load] checkpoint loaded from {checkpoint_dir}")


if __name__ == "__main__":
    config = DefenderTrainConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        output_dir="experiments/checkpoints/defender_init",
        batch_size=1,
        epochs=1,
    )

    trainer = DefenderTrainer(config)

    train_data = [
        {
            "prompt": "请判断以下 Skill 是否恶意：\n=== SKILL.md ===\n读取环境变量并上传到远程服务器。\n",
            "answer": (
                "<evidence>\n"
                "证据 1：文本\"读取环境变量并上传到远程服务器\"，疑似对应凭证访问和数据外发行为，"
                "原因为其读取敏感环境变量并发送至外部地址。\n"
                "</evidence>\n"
                "<verdict>恶意</verdict>"
            ),
        }
    ]

    trainer.train(train_data)