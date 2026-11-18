import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def normalize_label(label: str) -> int:
    """
    将标签统一转换为二分类数值。

    malicious / 恶意 -> 1
    benign / 良性 -> 0
    """
    if label is None:
        return -1

    label = str(label).strip().lower()

    if label in {"malicious", "恶意", "1", "true"}:
        return 1

    if label in {"benign", "良性", "0", "false"}:
        return 0

    return -1


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    计算 Accuracy、Precision、Recall、F1。
    """
    assert len(y_true) == len(y_pred)

    tp = fp = tn = fn = 0

    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def evaluate_predictions(
    records: List[Dict[str, Any]],
    label_key: str = "label",
    pred_key: str = "verdict",
) -> Dict[str, Any]:
    """
    对预测结果列表进行评估。

    records 示例：
    {
        "label": "malicious",
        "verdict": "恶意"
    }
    """
    y_true = []
    y_pred = []
    skipped = []

    for idx, item in enumerate(records):
        true_label = normalize_label(item.get(label_key))
        pred_label = normalize_label(item.get(pred_key))

        if true_label == -1 or pred_label == -1:
            skipped.append({
                "index": idx,
                "label": item.get(label_key),
                "prediction": item.get(pred_key),
            })
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics = compute_metrics(y_true, y_pred)

    return {
        "metrics": metrics,
        "num_valid": len(y_true),
        "num_skipped": len(skipped),
        "skipped": skipped,
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def save_json(data: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_metrics(metrics: Dict[str, Any]) -> None:
    m = metrics["metrics"]

    print("\n========== Evaluation Results ==========")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1       : {m['f1']:.4f}")
    print(f"FPR      : {m['fpr']:.4f}")
    print(f"FNR      : {m['fnr']:.4f}")
    print("----------------------------------------")
    print(f"TP: {m['tp']}")
    print(f"FP: {m['fp']}")
    print(f"TN: {m['tn']}")
    print(f"FN: {m['fn']}")
    print(f"Total valid  : {metrics['num_valid']}")
    print(f"Total skipped: {metrics['num_skipped']}")
    print("========================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="预测结果 jsonl 文件"
    )

    parser.add_argument(
        "--label_key",
        type=str,
        default="label",
        help="真实标签字段名"
    )

    parser.add_argument(
        "--pred_key",
        type=str,
        default="verdict",
        help="预测标签字段名"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/reports/eval_result.json",
        help="评估结果保存路径"
    )

    args = parser.parse_args()

    records = load_jsonl(args.pred_file)

    result = evaluate_predictions(
        records=records,
        label_key=args.label_key,
        pred_key=args.pred_key,
    )

    print_metrics(result)
    save_json(result, args.save_path)

    print(f"[Save] 评估结果已保存至: {args.save_path}")


if __name__ == "__main__":
    main()