import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class VerdictFocusedLoss(nn.Module):
    """
    Verdict-Focused Loss

    目标：
    - evidence 部分：辅助模型生成证据链
    - verdict 部分：重点优化最终“恶意/良性”判断

    总损失：
        loss = (1 - alpha) * evidence_loss + alpha * verdict_loss

    其中 alpha 越大，表示越关注最终判断标签。
    """

    def __init__(
        self,
        tokenizer,
        alpha: float = 0.7,
        ignore_index: int = -100,
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 必须在 [0, 1] 范围内")

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.ignore_index = ignore_index

        self.verdict_start_ids = tokenizer(
            "<verdict>",
            add_special_tokens=False
        )["input_ids"]

        self.verdict_end_ids = tokenizer(
            "</verdict>",
            add_special_tokens=False
        )["input_ids"]

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        参数：
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]

        返回：
            {
                "loss": 总损失,
                "evidence_loss": 证据链损失,
                "verdict_loss": 判断标签损失
            }
        """

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        batch_size, seq_len = shift_labels.shape
        device = shift_labels.device

        verdict_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=device
        )

        evidence_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=device
        )

        for i in range(batch_size):
            valid_labels = shift_labels[i]
            verdict_start = self._find_subsequence(
                valid_labels,
                self.verdict_start_ids
            )
            verdict_end = self._find_subsequence(
                valid_labels,
                self.verdict_end_ids
            )

            valid_token_mask = valid_labels.ne(self.ignore_index)

            if verdict_start is not None and verdict_end is not None:
                verdict_end_pos = verdict_end + len(self.verdict_end_ids)

                verdict_mask[i, verdict_start:verdict_end_pos] = True
                evidence_mask[i, :] = valid_token_mask
                evidence_mask[i, verdict_start:verdict_end_pos] = False
            else:
                evidence_mask[i, :] = valid_token_mask

        evidence_loss = self._masked_ce_loss(
            shift_logits,
            shift_labels,
            evidence_mask
        )

        verdict_loss = self._masked_ce_loss(
            shift_logits,
            shift_labels,
            verdict_mask
        )

        if torch.isnan(verdict_loss):
            total_loss = evidence_loss
        elif torch.isnan(evidence_loss):
            total_loss = verdict_loss
        else:
            total_loss = (1.0 - self.alpha) * evidence_loss + self.alpha * verdict_loss

        return {
            "loss": total_loss,
            "evidence_loss": evidence_loss.detach(),
            "verdict_loss": verdict_loss.detach()
        }

    def _masked_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        对指定 mask 范围内的 token 计算平均交叉熵。
        """

        if mask.sum() == 0:
            return torch.tensor(
                float("nan"),
                device=logits.device,
                dtype=logits.dtype
            )

        vocab_size = logits.size(-1)

        selected_logits = logits[mask]
        selected_labels = labels[mask]

        loss = F.cross_entropy(
            selected_logits.view(-1, vocab_size),
            selected_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="mean"
        )

        return loss

    @staticmethod
    def _find_subsequence(
        sequence: torch.Tensor,
        pattern: list
    ) -> Optional[int]:
        """
        在 token 序列中查找子序列位置。
        """

        if not pattern:
            return None

        pattern_tensor = torch.tensor(
            pattern,
            device=sequence.device,
            dtype=sequence.dtype
        )

        pattern_len = len(pattern)

        if sequence.numel() < pattern_len:
            return None

        for i in range(sequence.numel() - pattern_len + 1):
            if torch.equal(sequence[i:i + pattern_len], pattern_tensor):
                return i

        return None


def build_weighted_labels(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    构造训练 labels。

    对 CausalLM 微调时：
    - prompt 部分不计算损失，置为 -100
    - answer 部分计算损失

    参数：
        input_ids: [batch_size, seq_len]
        prompt_lengths: [batch_size]
    """

    labels = input_ids.clone()

    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = ignore_index

    return labels


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )

    loss_fn = VerdictFocusedLoss(
        tokenizer=tokenizer,
        alpha=0.7
    )

    text = """
<evidence>
证据 1：文本"读取环境变量"，疑似对应凭证访问行为，原因为其访问了 API Key。
</evidence>
<verdict>恶意</verdict>
"""

    encoded = tokenizer(
        [text],
        return_tensors="pt",
        padding=True
    )

    batch_size, seq_len = encoded["input_ids"].shape
    vocab_size = tokenizer.vocab_size

    fake_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = encoded["input_ids"]

    output = loss_fn(fake_logits, labels)

    print(output)