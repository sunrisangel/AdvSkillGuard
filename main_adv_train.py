import argparse
import json
from pathlib import Path

from data.data_utils import load_dataset_from_jsonl
from attacker.rewriter import ContextBridgeRewriter, RewriteConfig
from attacker.gradient_attack import GradientAttributionAttack, GradientAttackConfig
from attacker.validator import FunctionalityPreservingValidator, ValidatorConfig
from defender.classifier import SkillDefenderClassifier
from defender.trainer import DefenderTrainer, DefenderTrainConfig
from defender.supervision import AttackAwareSupervisionBuilder
from adversarial.replay_buffer import ReplayBuffer
from adversarial.stopping import StoppingCriteria, StoppingConfig
from adversarial.training_loop import AdversarialTrainingLoop, AdversarialTrainConfig


def load_jsonl(path):
    data = []

    if path is None:
        return data

    path = Path(path)

    if not path.exists():
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--malicious_file", type=str, default="data/processed/malicious_samples.jsonl")
    parser.add_argument("--init_train_file", type=str, default="data/processed/train_sft.jsonl")
    parser.add_argument("--valid_file", type=str, default=None)

    parser.add_argument("--defender_model", type=str, default="experiments/checkpoints/defender_init")
    parser.add_argument("--attacker_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    parser.add_argument("--defender_prompt", type=str, default="prompts/defender_prompt.txt")
    parser.add_argument("--attacker_prompt", type=str, default="prompts/attacker_prompt.txt")

    parser.add_argument("--save_dir", type=str, default="experiments/adversarial")
    parser.add_argument("--replay_path", type=str, default="experiments/adversarial/replay_buffer.jsonl")

    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--evasion_threshold", type=float, default=0.05)
    parser.add_argument("--incremental_steps", type=int, default=200)

    parser.add_argument("--top_k_patterns", type=int, default=20)
    parser.add_argument("--candidates_per_strategy", type=int, default=2)

    parser.add_argument("--history_sample_ratio", type=float, default=0.2)
    parser.add_argument("--init_sample_ratio", type=float, default=0.2)

    parser.add_argument("--overlap_threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    malicious_samples = load_dataset_from_jsonl(args.malicious_file)
    init_train_data = load_jsonl(args.init_train_file)
    valid_data = load_jsonl(args.valid_file)

    print(f"[Data] malicious_samples={len(malicious_samples)}")
    print(f"[Data] init_train_data={len(init_train_data)}")
    print(f"[Data] valid_data={len(valid_data)}")

    rewriter = ContextBridgeRewriter(
        RewriteConfig(
            model_name_or_path=args.attacker_model,
            prompt_path=args.attacker_prompt,
            candidates_per_strategy=args.candidates_per_strategy,
            fp16=args.fp16,
        )
    )

    gradient_attacker = GradientAttributionAttack(
        GradientAttackConfig(
            model_name_or_path=args.defender_model,
            top_k=args.top_k_patterns,
            fp16=args.fp16,
        )
    )

    validator = FunctionalityPreservingValidator(
        ValidatorConfig(
            overlap_threshold=args.overlap_threshold,
        )
    )

    classifier = SkillDefenderClassifier(
        model_name_or_path=args.defender_model,
        prompt_path=args.defender_prompt,
        temperature=0.0,
    )

    defender_trainer = DefenderTrainer(
        DefenderTrainConfig(
            model_name_or_path=args.defender_model,
            output_dir=f"{args.save_dir}/defender_latest",
            batch_size=args.batch_size,
            fp16=args.fp16,
            learning_rate=1e-5,
            epochs=1,
        )
    )

    prompt_template = Path(args.defender_prompt).read_text(encoding="utf-8")

    supervision_builder = AttackAwareSupervisionBuilder(
        defender_prompt_template=prompt_template,
    )

    replay_buffer = ReplayBuffer(
        max_size=10000,
        save_path=args.replay_path,
    )

    stopper = StoppingCriteria(
        StoppingConfig(
            patience=5,
            min_improvement=0.005,
            evasion_threshold=args.evasion_threshold,
        )
    )

    adv_loop = AdversarialTrainingLoop(
        config=AdversarialTrainConfig(
            max_rounds=args.max_rounds,
            evasion_threshold=args.evasion_threshold,
            history_sample_ratio=args.history_sample_ratio,
            init_sample_ratio=args.init_sample_ratio,
            incremental_steps=args.incremental_steps,
            save_dir=args.save_dir,
        ),
        rewriter=rewriter,
        gradient_attacker=gradient_attacker,
        validator=validator,
        classifier=classifier,
        defender_trainer=defender_trainer,
        supervision_builder=supervision_builder,
        replay_buffer=replay_buffer,
        stopper=stopper,
    )

    result = adv_loop.run(
        malicious_samples=malicious_samples,
        init_train_data=init_train_data,
        valid_data=valid_data,
    )

    save_path = Path(args.save_dir) / "adv_train_summary.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Done] 对抗训练完成，结果保存至: {save_path}")


if __name__ == "__main__":
    main()