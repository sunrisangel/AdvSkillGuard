from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GRPOConfig:
    model_name_or_path: str
    output_dir: str = "experiments/checkpoints/attacker_grpo"
    group_size: int = 8
    max_length: int = 4096  # maximum total sequence length, prompt + completion
    max_new_tokens: int = 1024
    learning_rate: float = 1e-6
    epochs: int = 1
    max_train_steps: Optional[int] = 120  # across this optimize() call
    ppo_epochs: int = 2  # >1 lets clipping constrain changes on a fixed rollout
    beta_kl: float = 0.01
    clip_eps: float = 0.2
    entropy_coef_max: float = 0.05
    smoothing_delta: float = 1e-6
    temperature: float = 0.8
    top_p: float = 0.9
    fp16: bool = True
    device: Optional[str] = None
    seed: int = 42
    shuffle: bool = True
    grad_clip: float = 1.0
    entropy_stop_threshold: Optional[float] = None
    reward_patience: Optional[int] = None
    min_reward_improvement: float = 1e-4
    trust_remote_code: bool = False
    # Full-size models should instead use FSDP/DeepSpeed/PEFT in a separate setup.

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if not 0 < self.temperature:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        if self.ppo_epochs < 1 or self.epochs < 1:
            raise ValueError("epochs and ppo_epochs must be positive")
        if not 0 < self.clip_eps < 1:
            raise ValueError("clip_eps must be in (0,1)")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")
        if self.max_length <= self.max_new_tokens + 1:
            raise ValueError("max_length must exceed max_new_tokens + 1")


class GRPOAttackTrainer:
    """Group-relative PPO over validated skill rewrite candidates.

    Each optimize() call starts a new GRPO stage and snapshots the *current*
    attacker as the frozen reference. The sample-time policy log-probabilities
    are fixed across its PPO epochs.

    This is a single-device correctness-oriented implementation, not a
    performance-optimized 27B multi-GPU trainer.
    """

    def __init__(
        self,
        config: GRPOConfig,
        defender_classifier: Any,
        validator: Any,
        prompt_path: str = "prompts/attacker_prompt.txt",
    ) -> None:
        self.config = config
        self.defender_classifier = defender_classifier
        self.validator = validator
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = random.Random(config.seed)
        self.prompt_template = self.load_prompt(prompt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs pad_token or eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device.type == "cuda" and config.fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        ).to(self.device)
        self.ref_model = None  # created only during optimize() stage
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.model.config.use_cache = False
        self.total_optimizer_steps = 0

    @staticmethod
    def load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def format_scripts(scripts: Dict[str, str]) -> str:
        if not scripts:
            return "无"
        return "\n\n".join(
            f"### 文件: {name}\n```text\n{body}\n```"
            for name, body in scripts.items()
        )

    def build_prompt(
        self,
        skill_md: str,
        scripts: Dict[str, str],
        surface_patterns: Optional[List[str]] = None,
    ) -> str:
        return (self.prompt_template
                .replace("{original_skill_md}", skill_md)
                .replace("{script_files}", self.format_scripts(scripts))
                .replace("{selected_strategies}", "自适应组合")
                .replace("{detected_surface_patterns}", "、".join(surface_patterns or []) or "无"))

    @staticmethod
    def safe_parse_candidate(text: str) -> Dict[str, Any]:
        empty = {"rewritten_skill_md": "", "rewritten_scripts": {}, "modification_summary": []}
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            try:
                obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
            except (ValueError, TypeError):
                return empty
        if not isinstance(obj, dict):
            return empty
        if not isinstance(obj.get("rewritten_skill_md"), str):
            return empty
        scripts = obj.get("rewritten_scripts", {})
        if not isinstance(scripts, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in scripts.items()
        ):
            return empty
        summary = obj.get("modification_summary", [])
        if not isinstance(summary, (list, dict)):
            return empty
        return {"rewritten_skill_md": obj.get("rewritten_skill_md", ""),
                "rewritten_scripts": scripts, "modification_summary": summary}

    @classmethod
    def format_rewritten_skill(cls, candidate: Dict[str, Any]) -> str:
        return ("=== SKILL.md ===\n" + candidate["rewritten_skill_md"]
                + "\n\n=== 脚本文件 ===\n"
                + cls.format_scripts(candidate["rewritten_scripts"]))

    @staticmethod
    def _is_benign(verdict: Any) -> bool:
        return isinstance(verdict, str) and verdict.strip().lower() in {"benign", "良性"}

    def compute_rewards(self, original_skill: Dict[str, Any], responses: List[str]) -> List[float]:
        rewards = []
        for response in responses:
            candidate = self.safe_parse_candidate(response)
            if not candidate["rewritten_skill_md"]:
                rewards.append(0.0)
                continue
            # The injected validator MUST sandbox all execution and network access.
            try:
                checked = self.validator.validate(
                    original_skill=original_skill, rewritten_candidate=candidate
                )
                passed = bool(checked.get("passed", checked.get("fpv_passed", False)))
                if not passed:
                    rewards.append(0.0)
                    continue
                pred = self.defender_classifier.classify(self.format_rewritten_skill(candidate))
                rewards.append(float(self._is_benign(pred.get("verdict"))))
            except Exception as exc:
                # Fail closed: evaluation errors can never become positive rewards.
                print(f"[GRPO] reward evaluation failed: {exc}")
                rewards.append(0.0)
        return rewards

    @staticmethod
    def compute_advantages(rewards: List[float], eps: float = 1e-6) -> torch.Tensor:
        r = torch.as_tensor(rewards, dtype=torch.float32)
        return (r - r.mean()) / (r.std(unbiased=False) + eps)

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        max_prompt = self.config.max_length - self.config.max_new_tokens
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_prompt,
            add_special_tokens=True,
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        if ids.shape[1] < 1:
            raise ValueError("Empty tokenized prompt")
        return ids, mask

    def _generate_rollouts(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor):
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            sequences = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_return_sequences=self.config.group_size,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=False,
            )
        if was_training:
            self.model.train()
        prompt_len = prompt_ids.shape[1]
        raw_completion = sequences[:, prompt_len:]
        if raw_completion.shape[1] == 0:
            raise RuntimeError("Model generated zero tokens")
        # EOS is included in likelihood; trailing padding after EOS is not.
        mask = torch.ones_like(raw_completion, dtype=torch.float32)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            eos_hits = (raw_completion == eos).long()
            # Include the first EOS token, exclude all tokens after it.
            mask = ((eos_hits.cumsum(dim=1) - eos_hits) == 0).float()
        texts = []
        for i in range(raw_completion.shape[0]):
            n = int(mask[i].sum().item())
            texts.append(self.tokenizer.decode(raw_completion[i, :n], skip_special_tokens=True).strip())
        return sequences, mask, texts

    def _token_stats(self, model, sequences: torch.Tensor, response_mask: torch.Tensor,
                     prompt_len: int, *, entropy: bool = False):
        """Exact response-token log-probs from *generated IDs*, not retokenized text.

        For token with absolute position t, the predicting logit is at t-1.
        Masking excludes post-EOS padding. Entropy is averaged over response
        positions only, and remains differentiable when model is trainable.
        """
        seq_mask = torch.ones_like(sequences, dtype=torch.long)
        # Padding after EOS must not attend as real content.
        seq_mask[:, prompt_len:] = response_mask.long()
        outputs = model(input_ids=sequences, attention_mask=seq_mask, use_cache=False)
        start = prompt_len - 1
        n = response_mask.shape[1]
        logits = outputs.logits[:, start: start + n, :].float()
        labels = sequences[:, prompt_len:prompt_len + n]
        log_all = F.log_softmax(logits, dim=-1)
        log_probs = log_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        if not entropy:
            return log_probs, None
        probs = log_all.exp()
        entropies = -(probs * log_all).sum(-1)
        return log_probs, entropies

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _kl_surrogate(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
        delta = (ref_logp - logp).clamp(min=-20.0, max=20.0)
        return delta.exp() - delta - 1.0

    def _snapshot_reference(self) -> None:
        # Snapshot once per GRPO stage; may require significant extra memory.
        if self.ref_model is not None:
            del self.ref_model
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        self.ref_model.eval()
        self.ref_model.config.use_cache = False
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def train_step(self, original_skill: Dict[str, Any],
                   surface_patterns: Optional[List[str]] = None,
                   *, max_updates: Optional[int] = None) -> Dict[str, Any]:
        if self.ref_model is None:
            self._snapshot_reference()
        prompt = self.build_prompt(
            original_skill.get("skill_md", ""), original_skill.get("scripts", {}),
            surface_patterns=surface_patterns,
        )
        prompt_ids, prompt_mask = self._encode_prompt(prompt)
        sequences, response_mask, responses = self._generate_rollouts(prompt_ids, prompt_mask)
        rewards = self.compute_rewards(original_skill, responses)
        advantages = self.compute_advantages(rewards).to(self.device)
        n_success = sum(v > 0 for v in rewards)
        weights = torch.tensor(
            [self.config.group_size / (n_success + self.config.smoothing_delta) if r > 0 else 1.0
             for r in rewards], device=self.device,
        )
        prompt_len = prompt_ids.shape[1]
        with torch.no_grad():
            old_logp, _ = self._token_stats(self.model, sequences, response_mask, prompt_len)
            ref_logp, _ = self._token_stats(self.ref_model, sequences, response_mask, prompt_len)
        # KL and entropy still provide gradients for uniform-reward groups; the
        # policy-gradient component vanishes as expected.
        num_updates = min(self.config.ppo_epochs, max_updates or self.config.ppo_epochs)
        last = {}
        for _ in range(num_updates):
            self.model.train()
            current_logp, entropy_per_token = self._token_stats(
                self.model, sequences, response_mask, prompt_len, entropy=True
            )
            ratio = torch.exp((current_logp - old_logp).clamp(-20.0, 20.0))
            clipped = ratio.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
            adv = advantages[:, None]
            weight = weights[:, None]
            ppo_obj = torch.minimum(ratio * adv, clipped * adv) * weight
            policy_loss = -self._masked_mean(ppo_obj, response_mask)
            kl_loss = self._masked_mean(self._kl_surrogate(current_logp, ref_logp), response_mask)
            entropy = self._masked_mean(entropy_per_token, response_mask)
            mean_reward = sum(rewards) / len(rewards)
            entropy_coef = self.config.entropy_coef_max * (1.0 - mean_reward)
            loss = policy_loss + self.config.beta_kl * kl_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GRPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.total_optimizer_steps += 1
            last = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "kl_loss": float(kl_loss.detach()),
                "entropy": float(entropy.detach()),
                "entropy_coef": entropy_coef,
                "avg_reward": mean_reward,
            }
        last.update({"rewards": rewards, "num_success": n_success,
                     "optimizer_steps": num_updates})
        return last

    def train(self, malicious_samples: List[Dict[str, Any]],
              surface_patterns_map: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """Backward-compatible training interface; starts a GRPO stage."""
        return self._train_stage(malicious_samples, surface_patterns_map)

    def _train_stage(self, samples: List[Dict[str, Any]],
                     patterns: Optional[Dict[str, List[str]]] = None,
                     group_size: Optional[int] = None) -> List[Dict[str, Any]]:
        if not samples:
            raise ValueError("GRPO requires nonempty malicious_samples")
        if group_size is not None and group_size != self.config.group_size:
            raise ValueError("group_size must match GRPOConfig.group_size")
        self._snapshot_reference()
        stage_start_steps = self.total_optimizer_steps
        max_steps = self.config.max_train_steps
        logs = []
        best_reward = -float("inf")
        stale = 0
        for epoch in range(self.config.epochs):
            order = list(range(len(samples)))
            if self.config.shuffle:
                self.rng.shuffle(order)
            for idx in order:
                remaining = None if max_steps is None else max_steps - (self.total_optimizer_steps - stage_start_steps)
                if remaining is not None and remaining <= 0:
                    break
                sample = samples[idx]
                key = str(sample.get("skill_dir", idx))
                log = self.train_step(sample, (patterns or {}).get(key), max_updates=remaining)
                log.update({"epoch": epoch, "sample_index": idx, "skill_dir": sample.get("skill_dir")})
                logs.append(log)
                print(f"[GRPO] epoch={epoch} sample={idx+1}/{len(samples)} "
                      f"steps={self.total_optimizer_steps-stage_start_steps} "
                      f"reward={log['avg_reward']:.4f} loss={log['loss']:.4f}")
                if log["avg_reward"] > best_reward + self.config.min_reward_improvement:
                    best_reward = log["avg_reward"]
                    stale = 0
                else:
                    stale += 1
                if self.config.entropy_stop_threshold is not None and \
                   log["entropy"] <= self.config.entropy_stop_threshold:
                    break
                if self.config.reward_patience is not None and stale >= self.config.reward_patience:
                    break
            if max_steps is not None and self.total_optimizer_steps - stage_start_steps >= max_steps:
                break
            if logs and (
                (self.config.entropy_stop_threshold is not None and
                 logs[-1]["entropy"] <= self.config.entropy_stop_threshold) or
                (self.config.reward_patience is not None and stale >= self.config.reward_patience)
            ):
                break
        self.save_model()
        self.save_logs(logs)
        return logs

    def optimize(self, malicious_samples: List[Dict[str, Any]],
                 defender: Optional[Any] = None, validator: Optional[Any] = None,
                 group_size: Optional[int] = None,
                 surface_patterns_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """API expected by the revised adversarial-training main loop."""
        if defender is not None:
            self.defender_classifier = defender
        if validator is not None:
            self.validator = validator
        before = self.total_optimizer_steps
        logs = self._train_stage(malicious_samples, surface_patterns_map, group_size)
        updated_steps = self.total_optimizer_steps - before
        return {
            "updated": updated_steps > 0,
            "optimizer_steps": updated_steps,
            "samples_processed": len(logs),
            "avg_reward": (sum(x["avg_reward"] for x in logs) / len(logs)) if logs else None,
            "last_entropy": logs[-1]["entropy"] if logs else None,
            "checkpoint_path": self.config.output_dir,
            # The caller must refresh its ContextBridgeRewriter using this path
            # if it does not share the exact same live self.model object.
        }

    def save_model(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

    def save_logs(self, logs: List[Dict[str, Any]]) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "grpo_logs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in logs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = ["GRPOConfig", "GRPOAttackTrainer"]
