import argparse
import json
from pathlib import Path

from defender.trainer import DefenderTrainer, DefenderTrainConfig


def load_jsonl(path: str):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="基础模型路径"
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train_sft.jsonl",
        help="训练数据 jsonl 文件"
    )

    parser.add_argument(
        "--valid_file",
        type=str,
        default=None,
        help="验证数据 jsonl 文件"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/checkpoints/defender_init",
        help="模型保存路径"
    )

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    train_path = Path(args.train_file)

    if not train_path.exists():
        raise FileNotFoundError(f"训练文件不存在: {args.train_file}")

    train_data = load_jsonl(args.train_file)

    valid_data = None
    if args.valid_file is not None:
        valid_path = Path(args.valid_file)
        if valid_path.exists():
            valid_data = load_jsonl(args.valid_file)

    config = DefenderTrainConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        alpha=args.alpha,
        gradient_accumulation_steps=args.grad_accum,
        fp16=args.fp16,
    )

    trainer = DefenderTrainer(config)

    trainer.train(
        train_data=train_data,
        valid_data=valid_data,
    )


if __name__ == "__main__":
    main()