import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from defender.classifier import SkillDefenderClassifier
from evaluation.evaluate import evaluate_predictions, print_metrics, save_json


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_skill_content(sample: Dict[str, Any]) -> str:
    """
    兼容不同测试集格式：
    1. 如果已有 model_input，直接使用；
    2. 否则使用 skill_md + scripts 构造输入。
    """
    if sample.get("model_input"):
        return sample["model_input"]

    skill_md = sample.get("skill_md", "")
    scripts = sample.get("scripts", {})

    if not scripts:
        script_text = "无"
    else:
        blocks = []

        for file_name, content in scripts.items():
            blocks.append(
                f"### 文件: {file_name}\n"
                f"```text\n{content}\n```"
            )

        script_text = "\n\n".join(blocks)

    return f"""=== SKILL.md ===
{skill_md}

=== 脚本文件 ===
{script_text}
"""


def run_prediction(
    classifier: SkillDefenderClassifier,
    samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    predictions = []

    for idx, sample in enumerate(samples):
        print(f"[Predict] {idx + 1}/{len(samples)}")

        skill_content = build_skill_content(sample)

        try:
            result = classifier.classify(skill_content)

            record = dict(sample)
            record["verdict"] = result.get("verdict")
            record["evidence"] = result.get("evidence")
            record["raw_response"] = result.get("raw_response")
            record["predict_error"] = None

        except Exception as e:
            record = dict(sample)
            record["verdict"] = "未知"
            record["evidence"] = ""
            record["raw_response"] = ""
            record["predict_error"] = str(e)

        predictions.append(record)

    return predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="experiments/checkpoints/defender_init",
        help="训练好的防御模型路径"
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test.jsonl",
        help="测试集 jsonl 文件"
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/defender_prompt.txt",
        help="防御侧 Prompt 路径"
    )

    parser.add_argument(
        "--pred_save_path",
        type=str,
        default="experiments/predictions/test_predictions.jsonl",
        help="预测结果保存路径"
    )

    parser.add_argument(
        "--report_save_path",
        type=str,
        default="experiments/reports/test_eval_result.json",
        help="评估报告保存路径"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    test_data = load_jsonl(args.test_file)

    print(f"[Data] test samples: {len(test_data)}")

    classifier = SkillDefenderClassifier(
        model_name_or_path=args.model_name_or_path,
        prompt_path=args.prompt_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    predictions = run_prediction(
        classifier=classifier,
        samples=test_data,
    )

    save_jsonl(predictions, args.pred_save_path)

    result = evaluate_predictions(
        records=predictions,
        label_key="label",
        pred_key="verdict",
    )

    print_metrics(result)
    save_json(result, args.report_save_path)

    print(f"[Save] 预测结果已保存至: {args.pred_save_path}")
    print(f"[Save] 评估报告已保存至: {args.report_save_path}")


if __name__ == "__main__":
    main()