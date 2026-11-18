import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from data.data_utils import load_skill, build_defender_input
from defender.classifier import SkillDefenderClassifier


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scan_skill_dirs(root_dir: str) -> List[str]:
    """
    递归扫描真实 Skill 快照目录。
    只要某个目录下存在 SKILL.md，就认为它是一个 Skill。
    """
    skill_dirs = []

    for current_root, dirs, files in os.walk(root_dir):
        if "SKILL.md" in files:
            skill_dirs.append(current_root)

    return skill_dirs


def classify_realworld_skills(
    classifier: SkillDefenderClassifier,
    skill_dirs: List[str],
    max_samples: int = None,
) -> List[Dict[str, Any]]:
    """
    对真实场景 Skill 逐个进行检测。
    """

    if max_samples is not None:
        skill_dirs = skill_dirs[:max_samples]

    results = []

    for idx, skill_dir in enumerate(skill_dirs):
        print(f"[RealWorld] {idx + 1}/{len(skill_dirs)} {skill_dir}")

        skill_data = load_skill(skill_dir)

        record = {
            "skill_dir": skill_dir,
            "valid": skill_data.get("valid", False),
            "error": skill_data.get("error"),
            "skill_md_path": skill_data.get("skill_md_path"),
        }

        if not skill_data.get("valid"):
            record.update({
                "verdict": "未知",
                "evidence": "",
                "raw_response": "",
                "predict_error": skill_data.get("error"),
            })
            results.append(record)
            continue

        skill_content = build_defender_input(skill_data)

        try:
            pred = classifier.classify(skill_content)

            record.update({
                "verdict": pred.get("verdict"),
                "evidence": pred.get("evidence"),
                "raw_response": pred.get("raw_response"),
                "predict_error": None,
            })

        except Exception as e:
            record.update({
                "verdict": "未知",
                "evidence": "",
                "raw_response": "",
                "predict_error": str(e),
            })

        results.append(record)

    return results


def summarize_realworld_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    汇总真实场景检测结果。
    """

    total = len(results)
    valid = sum(1 for x in results if x.get("valid"))
    invalid = total - valid

    malicious = sum(1 for x in results if x.get("verdict") == "恶意")
    benign = sum(1 for x in results if x.get("verdict") == "良性")
    unknown = sum(1 for x in results if x.get("verdict") == "未知")

    with_evidence = sum(
        1 for x in results
        if x.get("evidence") and x.get("evidence") != "无"
    )

    prediction_errors = sum(
        1 for x in results
        if x.get("predict_error")
    )

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "malicious": malicious,
        "benign": benign,
        "unknown": unknown,
        "with_evidence": with_evidence,
        "prediction_errors": prediction_errors,
        "malicious_rate": malicious / valid if valid else 0.0,
        "benign_rate": benign / valid if valid else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n========== Real World Evaluation ==========")
    print(f"Total Skills     : {summary['total']}")
    print(f"Valid Skills     : {summary['valid']}")
    print(f"Invalid Skills   : {summary['invalid']}")
    print("-------------------------------------------")
    print(f"Pred Malicious   : {summary['malicious']}")
    print(f"Pred Benign      : {summary['benign']}")
    print(f"Pred Unknown     : {summary['unknown']}")
    print(f"With Evidence    : {summary['with_evidence']}")
    print(f"Prediction Errors: {summary['prediction_errors']}")
    print("-------------------------------------------")
    print(f"Malicious Rate   : {summary['malicious_rate']:.4f}")
    print(f"Benign Rate      : {summary['benign_rate']:.4f}")
    print(f"Unknown Rate     : {summary['unknown_rate']:.4f}")
    print("===========================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="真实 Skill 快照根目录"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="experiments/checkpoints/defender_init",
        help="防御模型路径"
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/defender_prompt.txt",
        help="防御侧 Prompt 路径"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多评估多少个 Skill，默认全部"
    )

    parser.add_argument(
        "--pred_save_path",
        type=str,
        default="experiments/predictions/realworld_predictions.jsonl"
    )

    parser.add_argument(
        "--summary_save_path",
        type=str,
        default="experiments/reports/realworld_summary.json"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512
    )

    args = parser.parse_args()

    skill_dirs = scan_skill_dirs(args.root_dir)

    print(f"[Scan] 发现 Skill 数量: {len(skill_dirs)}")

    classifier = SkillDefenderClassifier(
        model_name_or_path=args.model_name_or_path,
        prompt_path=args.prompt_path,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    results = classify_realworld_skills(
        classifier=classifier,
        skill_dirs=skill_dirs,
        max_samples=args.max_samples,
    )

    summary = summarize_realworld_results(results)

    save_jsonl(results, args.pred_save_path)
    save_json(summary, args.summary_save_path)

    print_summary(summary)

    print(f"[Save] 预测结果保存至: {args.pred_save_path}")
    print(f"[Save] 汇总结果保存至: {args.summary_save_path}")


if __name__ == "__main__":
    main()