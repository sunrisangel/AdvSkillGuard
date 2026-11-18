import re
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SkillDefenderClassifier:
    """
    基于 LLM 的恶意 Skill 证据链分类器。

    输入：
        Skill.md + 脚本文件内容

    输出：
        <evidence>...</evidence>
        <verdict>恶意/良性</verdict>
    """

    def __init__(
        self,
        model_name_or_path: str,
        prompt_path: str = "prompts/defender_prompt.txt",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_name_or_path = model_name_or_path
        self.prompt_template = self._load_prompt(prompt_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def _load_prompt(prompt_path: str) -> str:
        path = Path(prompt_path)

        if not path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")

        return path.read_text(encoding="utf-8")

    def build_prompt(self, skill_content: str) -> str:
        """
        将 Skill 内容填入 Prompt 模板。
        """
        return self.prompt_template.replace("{skill_content}", skill_content)

    def generate(self, skill_content: str) -> str:
        """
        调用 LLM 生成完整审查结果。
        """
        prompt = self.build_prompt(skill_content)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        do_sample = self.temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return response.strip()

    @staticmethod
    def parse_evidence(response: str) -> str:
        """
        解析 <evidence>...</evidence> 内容。
        """
        match = re.search(
            r"<evidence>(.*?)</evidence>",
            response,
            flags=re.DOTALL
        )

        if match:
            return match.group(1).strip()

        return ""

    @staticmethod
    def parse_verdict(response: str) -> str:
        """
        解析 <verdict>恶意/良性</verdict>。
        """
        match = re.search(
            r"<verdict>\s*(恶意|良性)\s*</verdict>",
            response
        )

        if match:
            return match.group(1)

        if "恶意" in response and "良性" not in response:
            return "恶意"

        if "良性" in response and "恶意" not in response:
            return "良性"

        return "未知"

    def classify(self, skill_content: str) -> Dict[str, Any]:
        """
        对单个 Skill 内容进行分类。
        """
        response = self.generate(skill_content)

        evidence = self.parse_evidence(response)
        verdict = self.parse_verdict(response)

        return {
            "verdict": verdict,
            "evidence": evidence,
            "raw_response": response
        }

    def classify_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        对 data_utils.py 构造出的 sample 进行分类。
        """
        skill_content = sample.get("model_input", "")

        if not skill_content:
            return {
                "verdict": "未知",
                "evidence": "",
                "raw_response": "",
                "error": "empty model_input"
            }

        result = self.classify(skill_content)
        result["skill_dir"] = sample.get("skill_dir")
        result["label"] = sample.get("label")

        return result


def load_classifier(
    model_name_or_path: str,
    prompt_path: str = "prompts/defender_prompt.txt"
) -> SkillDefenderClassifier:
    """
    快速加载分类器。
    """
    return SkillDefenderClassifier(
        model_name_or_path=model_name_or_path,
        prompt_path=prompt_path
    )


if __name__ == "__main__":
    from data.data_utils import load_skill, build_defender_input

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    skill_dir = "data/raw/example_skill"

    skill_data = load_skill(skill_dir)
    skill_content = build_defender_input(skill_data)

    classifier = SkillDefenderClassifier(
        model_name_or_path=model_path,
        prompt_path="prompts/defender_prompt.txt"
    )

    result = classifier.classify(skill_content)

    print("判断结果:", result["verdict"])
    print("证据链:")
    print(result["evidence"])
    print("原始输出:")
    print(result["raw_response"])