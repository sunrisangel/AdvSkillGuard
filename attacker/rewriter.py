import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STRATEGY_CONFIGS: List[List[str]] = [
    ["S1"], ["S2"], ["S3"], ["S4"],
    ["S1", "S2"], ["S1", "S3"], ["S1", "S4"],
    ["S2", "S3"], ["S2", "S4"], ["S3", "S4"],
]


@dataclass
class RewriteConfig:
    model_name_or_path: str
    prompt_path: str = "prompts/attacker_prompt.txt"
    max_length: int = 4096
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.9
    candidates_per_strategy: int = 2
    device: Optional[str] = None
    fp16: bool = True
    device_map: Optional[str] = None  # GPU 上默认 auto 分片
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path 不能为空")
        if self.max_length < 1 or self.max_new_tokens < 1:
            raise ValueError("max_length、max_new_tokens 必须为正整数")
        if self.candidates_per_strategy < 1:
            raise ValueError("candidates_per_strategy 必须至少为 1")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature 必须大于 0；top_p 必须属于 (0,1]")


class ContextBridgeRewriter:
    """固定策略改写器；GRPO 更新后的模型可热同步。"""

    def __init__(self, config: RewriteConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_template = self._load_prompt(config.prompt_path)
        self._shared_grpo_model = False
        self.tokenizer = None
        self.model = None
        self.load_checkpoint(config.model_name_or_path)

    @staticmethod
    def _load_prompt(prompt_path: str) -> str:
        path = Path(prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"攻击侧 Prompt 文件不存在: {prompt_path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def format_scripts(scripts: Dict[str, str]) -> str:
        if not scripts:
            return "无"
        if not isinstance(scripts, dict):
            raise TypeError("scripts 必须为字典")
        return "\n\n".join(
            f"### 文件: {name}\n```text\n{content}\n```"
            for name, content in scripts.items()
        )

    @staticmethod
    def _model_input_device(model: Any) -> torch.device:
        """取得输入嵌入层所在设备，兼容 device_map='auto' 的多卡模型。"""
        try:
            return model.get_input_embeddings().weight.device
        except (AttributeError, TypeError):
            return next(model.parameters()).device

    def load_checkpoint(self, model_path: str) -> None:
        """从本地检查点/模型路径载入生成模型；适用于与 GRPO 分离的进程。"""
        if not model_path:
            raise ValueError("model_path 不能为空")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=self.config.trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("tokenizer 缺少 pad_token 和 eos_token")
            tokenizer.pad_token = tokenizer.eos_token

        cuda_mode = str(self.device).startswith("cuda")
        dtype = torch.float16 if cuda_mode and self.config.fp16 else torch.float32
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype,
        }
        device_map = self.config.device_map
        if cuda_mode and device_map is None and self.config.device == "cuda":
            device_map = "auto"
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device_map is None:
            model = model.to(self.device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.config.model_name_or_path = str(model_path)
        self._shared_grpo_model = False

    def sync_from_grpo(self, trainer: Any) -> None:
        """GRPO 后共享同一份已更新模型权重，避免额外加载完整大模型。

        要求 GRPOAttackTrainer 实例暴露 model/tokenizer 属性。该方式适合同进程顺序
        执行训练与生成；不要在两者之间进行并发训练与推理。
        """
        if getattr(trainer, "model", None) is None or getattr(trainer, "tokenizer", None) is None:
            raise TypeError("GRPO trainer 必须包含 model 和 tokenizer 属性")
        self.model = trainer.model
        self.tokenizer = trainer.tokenizer
        self.model.eval()
        self._shared_grpo_model = True

    # 为外部调用提供语义明确的别名
    refresh_from_grpo = sync_from_grpo

    def build_prompt(
        self,
        original_skill_md: str,
        script_files: Dict[str, str],
        strategies: List[str],
        detected_surface_patterns: Optional[List[str]] = None,
    ) -> str:
        substitutions = {
            "{original_skill_md}": original_skill_md or "",
            "{script_files}": self.format_scripts(script_files),
            "{selected_strategies}": "、".join(strategies),
            "{detected_surface_patterns}": (
                "、".join(map(str, detected_surface_patterns))
                if detected_surface_patterns else "无"
            ),
        }
        prompt = self.prompt_template
        for placeholder, value in substitutions.items():
            prompt = prompt.replace(placeholder, value)
        return prompt

    @staticmethod
    def _invalid_candidate(strategies: List[str], error: str) -> Dict[str, Any]:
        return {
            "selected_strategies": list(strategies),
            "requested_strategies": list(strategies),
            "rewritten_skill_md": "",
            "rewritten_scripts": {},
            "modification_summary": [],
            "raw_response": "",
            "valid_format": False,
            "error": error,
        }

    @staticmethod
    def parse_response(response: str) -> Dict[str, Any]:
        """提取首个有效 JSON 对象（支持代码围栏和额外说明文字）。"""
        decoder = json.JSONDecoder()
        value: Any = None
        for index, char in enumerate(response):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(response[index:])
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            value = {
                "selected_strategies": [],
                "rewritten_skill_md": "",
                "rewritten_scripts": {},
                "modification_summary": [],
                "parse_error": True,
            }
        value.setdefault("selected_strategies", [])
        value.setdefault("rewritten_skill_md", "")
        value.setdefault("rewritten_scripts", {})
        value.setdefault("modification_summary", [])
        return value

    @staticmethod
    def is_valid_candidate(candidate: Dict[str, Any]) -> bool:
        if not isinstance(candidate, dict):
            return False
        if not isinstance(candidate.get("rewritten_skill_md"), str):
            return False
        if not candidate["rewritten_skill_md"].strip():
            return False
        scripts = candidate.get("rewritten_scripts")
        if not isinstance(scripts, dict):
            return False
        if any(not isinstance(name, str) or not isinstance(content, str)
               for name, content in scripts.items()):
            return False
        if not isinstance(candidate.get("modification_summary"), list):
            return False
        return True

    def generate_once(
        self,
        original_skill_md: str,
        script_files: Dict[str, str],
        strategies: List[str],
        detected_surface_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """对一个策略配置生成一个候选，不运行其脚本。"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("尚未加载攻击模型")
        prompt = self.build_prompt(
            original_skill_md, script_files, strategies, detected_surface_patterns
        )
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        )
        input_device = self._model_input_device(self.model)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        length = encoded["input_ids"].shape[-1]

        # GRPO 训练完成后共享的模型可能处于 train 模式；固定策略生成期间切换 eval。
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if was_training:
                self.model.train()
        response_ids = generated[0][length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        result = self.parse_response(response)
        result["raw_response"] = response
        result["requested_strategies"] = list(strategies)
        result["valid_format"] = self.is_valid_candidate(result)
        return result

    def generate_candidates(
        self,
        original_skill_md: str,
        script_files: Dict[str, str],
        detected_surface_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """10 个策略配置 × 2 次采样，返回 20 个固定槽位。"""
        result: List[Dict[str, Any]] = []
        for strategies in STRATEGY_CONFIGS:
            for _ in range(self.config.candidates_per_strategy):
                try:
                    candidate = self.generate_once(
                        original_skill_md, script_files, strategies, detected_surface_patterns
                    )
                except Exception as exc:
                    candidate = self._invalid_candidate(strategies, str(exc))
                result.append(candidate)
        return result

    def rewrite_sample(
        self,
        skill_sample: Dict[str, Any],
        detected_surface_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """兼容对抗训练主循环中的 rewrite_sample 调用。"""
        if not isinstance(skill_sample, dict):
            raise TypeError("skill_sample 必须是字典")
        candidates = self.generate_candidates(
            skill_sample.get("skill_md", ""),
            skill_sample.get("scripts", {}),
            detected_surface_patterns,
        )
        for candidate in candidates:
            candidate["source_skill_dir"] = skill_sample.get("skill_dir")
            candidate["source_label"] = skill_sample.get("label", "malicious")
            candidate["valid_format"] = self.is_valid_candidate(candidate)
        return candidates


if __name__ == "__main__":
    from data.data_utils import load_skill

    skill = load_skill("data/raw/example_malicious_skill")
    config = RewriteConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        candidates_per_strategy=2,
    )
    rewriter = ContextBridgeRewriter(config)
    candidates = rewriter.rewrite_sample(skill)
    print(f"候选槽位数量: {len(candidates)}")
    print(json.dumps(candidates[0], ensure_ascii=False, indent=2))
