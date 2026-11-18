import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from attacker.rewriter import ContextBridgeRewriter
from attacker.gradient_attack import GradientAttributionAttack
from attacker.validator import FunctionalityPreservingValidator
from defender.classifier import SkillDefenderClassifier
from defender.trainer import DefenderTrainer
from defender.supervision import AttackAwareSupervisionBuilder
from adversarial.replay_buffer import ReplayBuffer
from adversarial.stopping import StoppingCriteria


@dataclass
class AdversarialTrainConfig:
    max_rounds: int = 20       # 上限，不等于实际收敛轮数
    candidates_per_skill: int = 20
    grpo_group_size: int = 8
    evasion_threshold: float = 0.05
    history_sample_ratio: float = 0.2
    init_sample_ratio: float = 0.2
    incremental_steps: int = 200
    convergence_patience: int = 5
    convergence_delta: float = 0.005  # 按 0~1 比例表示 0.50 个百分点
    validation_metric_key: str = "f1"
    save_dir: str = "experiments/adversarial"
    save_each_round: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.max_rounds <= 0 or self.candidates_per_skill <= 0 or self.grpo_group_size <= 1:
            raise ValueError("轮数和候选数必须为正，GRPO 组大小至少为 2")
        if not 0 <= self.evasion_threshold < 1:
            raise ValueError("evasion_threshold 必须在 [0, 1) 内")
        for key in ("history_sample_ratio", "init_sample_ratio"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} 必须在 [0, 1] 内")
        if self.convergence_patience < 1 or self.convergence_delta < 0:
            raise ValueError("收敛窗口和阈值配置无效")


class AdversarialTrainingLoop:
    """固定策略候选生成 -> 计算 ER -> 更新防御者或执行 GRPO -> 联合停止判定。"""

    def __init__(
        self,
        config: AdversarialTrainConfig,
        rewriter: ContextBridgeRewriter,
        gradient_attacker: GradientAttributionAttack,
        validator: FunctionalityPreservingValidator,
        classifier: SkillDefenderClassifier,
        defender_trainer: DefenderTrainer,
        supervision_builder: AttackAwareSupervisionBuilder,
        replay_buffer: ReplayBuffer,
        stopper: Optional[StoppingCriteria] = None,
        grpo_trainer: Optional[Any] = None,
        refresh_attacker: Optional[Callable[[Any, ContextBridgeRewriter], None]] = None,
        refresh_defender: Optional[Callable[[DefenderTrainer, SkillDefenderClassifier, GradientAttributionAttack], None]] = None,
    ):
        self.config = config
        self.rewriter = rewriter
        self.gradient_attacker = gradient_attacker
        self.validator = validator
        self.classifier = classifier
        self.defender_trainer = defender_trainer
        self.supervision_builder = supervision_builder
        self.replay_buffer = replay_buffer
        self.stopper = stopper  # 兼容已有配置；停止判定使用本文独立实现的双条件
        self.grpo_trainer = grpo_trainer
        self.refresh_attacker = refresh_attacker
        self.refresh_defender = refresh_defender
        self.rng = random.Random(config.seed)
        self._validation_metrics: List[float] = []
        self._best_validation_metric: Optional[float] = None
        self._stale_validation_rounds = 0
        os.makedirs(config.save_dir, exist_ok=True)

    def run(
        self,
        malicious_samples: List[Dict[str, Any]],
        init_train_data: Optional[List[Dict[str, Any]]] = None,
        valid_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not malicious_samples:
            raise ValueError("malicious_samples 为空")
        if valid_data is None or len(valid_data) == 0:
            raise ValueError("需要独立验证集才能依据论文双条件判断收敛")
        # 只接受可解析输入；过滤掉明确无效的记录，防止分母口径失真。
        train_set = [s for s in malicious_samples if s.get("valid", True) and
                     (s.get("model_input") or s.get("skill_md") or s.get("scripts"))]
        if not train_set:
            raise ValueError("没有有效的恶意 Skill")
        print(f"[Data] 有效恶意 Skill: {len(train_set)} / {len(malicious_samples)}")
        history: List[Dict[str, Any]] = []
        for round_id in range(self.config.max_rounds):
            print(f"\n========== Adversarial Round {round_id} ==========")
            result = self.run_one_round(round_id, train_set, init_train_data, valid_data)
            history.append(result)
            if self.config.save_each_round:
                self.save_round_result(round_id, result)
            if result["should_stop"]:
                print("[Stop] 防御者收敛且 GRPO 优化后攻击者仍不能突破阈值")
                break
        return {"rounds": history, "replay_buffer_stats": self.replay_buffer.stats()}

    def _evaluate_fixed_strategy(
        self,
        round_id: int,
        malicious_samples: List[Dict[str, Any]],
        phase: str = "before_grpo",
    ) -> Dict[str, Any]:
        evasion_candidates: List[Dict[str, Any]] = []
        generated = 0
        valid_format_count = 0
        fpv_passed_count = 0
        for idx, sample in enumerate(malicious_samples, 1):
            print(f"[Round {round_id} {phase}] skill {idx}/{len(malicious_samples)}")
            skill_content = sample.get("model_input") or self._build_skill_content(sample)
            surface_patterns = self.gradient_attacker.get_surface_patterns(skill_content)
            candidates = self.rewriter.rewrite_sample(
                skill_sample=sample, detected_surface_patterns=surface_patterns
            )
            if not isinstance(candidates, list):
                raise TypeError("rewrite_sample 必须返回 list[dict]")
            if len(candidates) > self.config.candidates_per_skill:
                raise ValueError("单个 Skill 候选超过固定预算；请检查 rewriter 策略数量")
            generated += len(candidates)
            formatted = [c for c in candidates if c.get("valid_format", False)]
            valid_format_count += len(formatted)
            if not formatted:
                continue
            checked = self.validator.validate_batch(original_skill=sample, candidates=formatted)
            if not isinstance(checked, list):
                raise TypeError("validate_batch 必须返回 list[dict]")
            passed = [c for c in checked if c.get("fpv_passed", False)]
            fpv_passed_count += len(passed)
            evasion_candidates.extend(self.filter_evasion_candidates(passed))

        theoretical = self.config.candidates_per_skill * len(malicious_samples)
        rate = self.compute_evasion_rate(len(evasion_candidates), theoretical)
        print(f"[{phase}] theoretical={theoretical}, generated={generated}, "
              f"FPV passed={fpv_passed_count}, escaped={len(evasion_candidates)}, ER={rate:.4f}")
        return {
            "total_candidates": theoretical,
            "generated_candidates": generated,
            "valid_format_candidates": valid_format_count,
            "fpv_passed": fpv_passed_count,
            "evasion_count": len(evasion_candidates),
            "evasion_rate": rate,
            "evasion_candidates": evasion_candidates,
        }

    def run_one_round(
        self,
        round_id: int,
        malicious_samples: List[Dict[str, Any]],
        init_train_data: Optional[List[Dict[str, Any]]] = None,
        valid_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not malicious_samples:
            raise ValueError("本轮无有效恶意 Skill")
        before = self._evaluate_fixed_strategy(round_id, malicious_samples)
        er_before = before["evasion_rate"]
        defender_updated = False
        attacker_optimized = False
        grpo_result: Optional[Dict[str, Any]] = None
        er_after_grpo: Optional[float] = None
        train_samples: List[Dict[str, Any]] = []

        if er_before > self.config.evasion_threshold:
            train_samples = self.supervision_builder.build_batch(before["evasion_candidates"])
            if not train_samples:
                raise RuntimeError("有逃逸样本但无法生成攻击感知监督数据，请检查 supervision_builder")
            # 回放样本必须在本轮数据入池之前抽取，避免当前样本重复出现。
            historical = self.replay_buffer.sample(ratio=self.config.history_sample_ratio)
            initial = self.sample_init_data(init_train_data)
            self.defender_trainer.incremental_train(
                current_evasion_data=train_samples,
                history_data=historical,
                init_data=initial,
                steps=self.config.incremental_steps,
            )
            defender_updated = True
            if self.refresh_defender is not None:
                # 如果 trainer 与 classifier / gradient_attacker 不共享模型，此回调必须刷新它们。
                self.refresh_defender(self.defender_trainer, self.classifier, self.gradient_attacker)
            self.defender_trainer.save_checkpoint(
                os.path.join(self.config.save_dir, f"defender_round_{round_id}")
            )
            self.replay_buffer.add_batch(train_samples)
            self.replay_buffer.save()
        else:
            print("[GRPO] 固定策略逃逸率未超过阈值，开始优化攻击者")
            if self.grpo_trainer is None:
                raise RuntimeError("缺少 grpo_trainer，不能用打印日志代替 GRPO 参数更新")
            result = self.grpo_trainer.optimize(
                malicious_samples=malicious_samples,
                defender=self.classifier,
                validator=self.validator,
                group_size=self.config.grpo_group_size,
            )
            if not isinstance(result, dict):
                raise TypeError("grpo_trainer.optimize 必须返回包含优化结果的 dict")
            if result.get("updated") is not True:
                raise RuntimeError("GRPO 没有确认完成攻击者参数更新")
            grpo_result = result
            attacker_optimized = True
            if self.refresh_attacker is not None:
                self.refresh_attacker(self.grpo_trainer, self.rewriter)
            # 再运行同一固定策略评估以验证新攻击者能否突破阈值。
            # 此处花费额外检测开销，且如果 rewriter 未加载新权重，结果不可信。
            after = self._evaluate_fixed_strategy(round_id, malicious_samples, phase="after_grpo")
            er_after_grpo = after["evasion_rate"]

        metric = self._evaluate_validation(valid_data)
        converged = self._update_validation_convergence(metric)
        saturated = attacker_optimized and er_after_grpo is not None and (
            er_after_grpo <= self.config.evasion_threshold
        )
        should_stop = converged and saturated
        stopping_state = {
            "defender_converged": converged,
            "attacker_saturated": saturated,
            "validation_metric": metric,
            "stale_validation_rounds": self._stale_validation_rounds,
            "should_stop": should_stop,
        }
        return {
            "round_id": round_id,
            "total_candidates": before["total_candidates"],
            "generated_candidates": before["generated_candidates"],
            "valid_format_candidates": before["valid_format_candidates"],
            "fpv_passed": before["fpv_passed"],
            "evasion_count": before["evasion_count"],
            "evasion_rate": er_before,
            "evasion_rate_after_grpo": er_after_grpo,
            "defender_updated": defender_updated,
            "attacker_optimized": attacker_optimized,
            "grpo_result": grpo_result,
            "train_samples": train_samples,
            "stopping_state": stopping_state,
            "should_stop": should_stop,
        }

    @staticmethod
    def _normalize_verdict(verdict: Any) -> str:
        if not isinstance(verdict, str):
            raise ValueError(f"无效的 verdict: {verdict!r}")
        val = verdict.strip().lower()
        mapping = {"良性": "benign", "benign": "benign", "恶意": "malicious", "malicious": "malicious"}
        if val not in mapping:
            raise ValueError(f"无法识别的 verdict: {verdict!r}")
        return mapping[val]

    def filter_evasion_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        escaped = []
        for candidate in candidates:
            skill_content = self.supervision_builder.format_skill_content(
                skill_md=candidate.get("rewritten_skill_md", ""),
                scripts=candidate.get("rewritten_scripts", {}),
            )
            result = self.classifier.classify(skill_content)
            if not isinstance(result, dict):
                raise TypeError("classifier.classify 应返回字典")
            normalized = self._normalize_verdict(result.get("verdict"))
            item = dict(candidate)
            item["defender_verdict"] = normalized
            item["defender_evidence"] = result.get("evidence")
            item["defender_raw_response"] = result.get("raw_response")
            if normalized == "benign":
                escaped.append(item)
        return escaped

    @staticmethod
    def compute_evasion_rate(num_evasion: int, total_candidates: int) -> float:
        if total_candidates <= 0:
            raise ValueError("候选总数必须大于零")
        if not 0 <= num_evasion <= total_candidates:
            raise ValueError("逃逸数量超出候选总数")
        return num_evasion / total_candidates

    def _evaluate_validation(self, valid_data: Optional[List[Dict[str, Any]]]) -> Optional[float]:
        if not valid_data:
            return None
        result = self.defender_trainer.evaluate(valid_data)
        key = self.config.validation_metric_key
        if key not in result:
            raise KeyError(f"验证结果缺少 {key!r}，不能用负损失冒充 F1")
        score = float(result[key])
        if not 0 <= score <= 1:
            raise ValueError(f"{key} 应为 [0,1] 之间的比例值")
        return score

    def _update_validation_convergence(self, metric: Optional[float]) -> bool:
        if metric is None:
            return False
        self._validation_metrics.append(metric)
        if self._best_validation_metric is None:
            self._best_validation_metric = metric
            self._stale_validation_rounds = 0
        elif metric > self._best_validation_metric + self.config.convergence_delta:
            self._best_validation_metric = metric
            self._stale_validation_rounds = 0
        else:
            self._stale_validation_rounds += 1
        return self._stale_validation_rounds >= self.config.convergence_patience

    def sample_init_data(self, init_train_data: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not init_train_data:
            return []
        count = int(len(init_train_data) * self.config.init_sample_ratio)
        return self.rng.sample(init_train_data, count) if count else []

    @staticmethod
    def _build_skill_content(sample: Dict[str, Any]) -> str:
        skill_md = sample.get("skill_md", "")
        scripts = sample.get("scripts") or {}
        blocks = [f"### 文件: {name}\n```text\n{content}\n```" for name, content in scripts.items()]
        script_text = "\n\n".join(blocks) if blocks else "无"
        return f"=== SKILL.md ===\n{skill_md}\n\n=== 脚本文件 ===\n{script_text}\n"

    def save_round_result(self, round_id: int, result: Dict[str, Any]) -> None:
        path = os.path.join(self.config.save_dir, f"round_{round_id}.json")
        data = dict(result)
        train_samples = data.pop("train_samples", [])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        train_path = os.path.join(self.config.save_dir, f"round_{round_id}_train_samples.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for item in train_samples:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        print(f"[Save] {path}")
