import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def load_round_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_adversarial_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    基于候选样本级别计算对抗评估指标。

    每条记录建议包含：
    {
        "valid_format": true,
        "fpv_passed": true,
        "defender_verdict": "良性"
    }
    """

    total = len(records)

    valid_format = sum(
        1 for x in records
        if x.get("valid_format", False)
    )

    fpv_passed = sum(
        1 for x in records
        if x.get("fpv_passed", False)
    )

    evasion_success = sum(
        1 for x in records
        if x.get("fpv_passed", False)
        and x.get("defender_verdict") == "良性"
    )

    invalid_format = total - valid_format
    fpv_failed = valid_format - fpv_passed

    format_valid_rate = valid_format / total if total else 0.0
    fpv_pass_rate = fpv_passed / valid_format if valid_format else 0.0
    evasion_rate_all = evasion_success / total if total else 0.0
    evasion_rate_after_fpv = (
        evasion_success / fpv_passed
        if fpv_passed else 0.0
    )

    return {
        "total_candidates": total,
        "valid_format": valid_format,
        "invalid_format": invalid_format,
        "fpv_passed": fpv_passed,
        "fpv_failed": fpv_failed,
        "evasion_success": evasion_success,
        "format_valid_rate": format_valid_rate,
        "fpv_pass_rate": fpv_pass_rate,
        "evasion_rate_all": evasion_rate_all,
        "evasion_rate_after_fpv": evasion_rate_after_fpv,
    }


def compute_round_metrics(round_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    直接基于 adversarial/training_loop.py 保存的 round_x.json 计算指标。
    """

    total_candidates = round_result.get("total_candidates", 0)
    valid_format_candidates = round_result.get("valid_format_candidates", 0)
    fpv_passed = round_result.get("fpv_passed", 0)
    evasion_count = round_result.get("evasion_count", 0)

    return {
        "round_id": round_result.get("round_id"),
        "total_candidates": total_candidates,
        "valid_format_candidates": valid_format_candidates,
        "fpv_passed": fpv_passed,
        "evasion_count": evasion_count,
        "format_valid_rate": (
            valid_format_candidates / total_candidates
            if total_candidates else 0.0
        ),
        "fpv_pass_rate": (
            fpv_passed / valid_format_candidates
            if valid_format_candidates else 0.0
        ),
        "evasion_rate_all": (
            evasion_count / total_candidates
            if total_candidates else 0.0
        ),
        "evasion_rate_after_fpv": (
            evasion_count / fpv_passed
            if fpv_passed else 0.0
        ),
        "defender_updated": round_result.get("defender_updated", False),
    }


def evaluate_round_dir(round_dir: str) -> Dict[str, Any]:
    """
    读取 experiments/adversarial/ 下所有 round_x.json，
    汇总多轮对抗训练指标。
    """

    round_dir = Path(round_dir)
    round_files = sorted(
        round_dir.glob("round_*.json"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    rounds = []

    for path in round_files:
        # 跳过 round_x_train_samples.jsonl
        if path.suffix != ".json":
            continue

        data = load_round_json(str(path))
        rounds.append(compute_round_metrics(data))

    if not rounds:
        return {
            "rounds": [],
            "best_round": None,
            "final_round": None,
        }

    best_round = min(
        rounds,
        key=lambda x: x["evasion_rate_all"]
    )

    final_round = rounds[-1]

    return {
        "rounds": rounds,
        "best_round": best_round,
        "final_round": final_round,
    }


def save_json(data: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_adversarial_report(result: Dict[str, Any]) -> None:
    print("\n========== Adversarial Evaluation ==========")

    if "rounds" in result:
        for r in result["rounds"]:
            print(
                f"Round {r['round_id']}: "
                f"total={r['total_candidates']}, "
                f"fpv={r['fpv_passed']}, "
                f"evasion={r['evasion_count']}, "
                f"evasion_rate={r['evasion_rate_all']:.4f}, "
                f"updated={r['defender_updated']}"
            )

        if result.get("final_round"):
            f = result["final_round"]
            print("--------------------------------------------")
            print(f"Final Round: {f['round_id']}")
            print(f"Final Evasion Rate: {f['evasion_rate_all']:.4f}")

        if result.get("best_round"):
            b = result["best_round"]
            print(f"Best Round: {b['round_id']}")
            print(f"Best Evasion Rate: {b['evasion_rate_all']:.4f}")

    else:
        print(f"Total Candidates : {result['total_candidates']}")
        print(f"Valid Format     : {result['valid_format']}")
        print(f"FPV Passed       : {result['fpv_passed']}")
        print(f"Evasion Success  : {result['evasion_success']}")
        print(f"Format Valid Rate: {result['format_valid_rate']:.4f}")
        print(f"FPV Pass Rate    : {result['fpv_pass_rate']:.4f}")
        print(f"Evasion Rate All : {result['evasion_rate_all']:.4f}")
        print(f"Evasion Rate FPV : {result['evasion_rate_after_fpv']:.4f}")

    print("============================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate_file",
        type=str,
        default=None,
        help="候选攻击样本 jsonl 文件"
    )

    parser.add_argument(
        "--round_dir",
        type=str,
        default=None,
        help="对抗训练 round_x.json 所在目录"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/reports/adversarial_eval.json",
    )

    args = parser.parse_args()

    if args.candidate_file:
        records = load_jsonl(args.candidate_file)
        result = compute_adversarial_metrics(records)

    elif args.round_dir:
        result = evaluate_round_dir(args.round_dir)

    else:
        raise ValueError("必须指定 --candidate_file 或 --round_dir")

    print_adversarial_report(result)
    save_json(result, args.save_path)

    print(f"[Save] 对抗评估结果已保存至: {args.save_path}")


if __name__ == "__main__":
    main()