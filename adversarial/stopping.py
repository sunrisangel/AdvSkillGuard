import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StoppingConfig:
    patience: int = 5
    min_improvement: float = 0.005
    evasion_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.patience, int) or self.patience < 1:
            raise ValueError("patience 必须为正整数")
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement 必须为有限的非负数")
        if not math.isfinite(self.evasion_threshold) or not 0 <= self.evasion_threshold <= 1:
            raise ValueError("evasion_threshold 必须在 [0, 1] 内")


class StoppingCriteria:
    """根据验证集收敛和 GRPO 后攻击饱和这两个条件决定是否停止。"""

    def __init__(self, config: Optional[StoppingConfig] = None):
        self.config = config if config is not None else StoppingConfig()
        self.best_metric: Optional[float] = None
        self.no_improve_rounds = 0
        self.history: List[Dict[str, Any]] = []
        self._last_round_id: Optional[int] = None

    @staticmethod
    def _check_rate(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须是 [0, 1] 内的有限数")
        return value

    def update(
        self,
        round_id: int,
        valid_metric: Optional[float],
        evasion_rate: float,
        attacker_optimized: bool = False,
        *,
        post_grpo_evasion_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """更新一轮收敛状态。

        Args:
            round_id: 从 0 开始且严格递增的训练轮次编号。
            valid_metric: 独立验证集指标，范围 [0, 1]，越高越好；
                若为 None，则不能判定防御者收敛。
            evasion_rate: 本轮固定策略的**优化前**逃逸率，用于记录。
            attacker_optimized: 本轮是否确实执行 GRPO 参数更新。
            post_grpo_evasion_rate: GRPO 更新攻击者并同步到改写器之后，
                使用规定的评估协议重新测得的逃逸率。未优化时传 None。

        Returns:
            记录当前进度和双重停止判定的字典。

        兼容旧调用形式，但旧调用未提供优化后逃逸率时不会触发攻击饱和，
        防止把优化前的低逃逸率误当成训练终止证据。
        """
        if not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id 必须为非负整数")
        if self._last_round_id is not None and round_id <= self._last_round_id:
            raise ValueError("round_id 必须严格递增，避免重复累计收敛轮次")
        before_rate = self._check_rate(evasion_rate, "evasion_rate")
        after_rate = None
        if post_grpo_evasion_rate is not None:
            after_rate = self._check_rate(post_grpo_evasion_rate, "post_grpo_evasion_rate")
            if not attacker_optimized:
                raise ValueError("未执行 GRPO 更新时，不应传入优化后逃逸率")

        metric: Optional[float] = None
        if valid_metric is not None:
            metric = self._check_rate(valid_metric, "valid_metric")

        # 避免验证集未评估时错误地累加停滞计数。
        if metric is None:
            improvement: Optional[float] = None
            self.no_improve_rounds = 0
            defender_converged = False
        elif self.best_metric is None:
            improvement = None  # 第一次评估仅建立基线
            self.best_metric = metric
            self.no_improve_rounds = 0
            defender_converged = False
        else:
            improvement = metric - self.best_metric
            # 与原实现一致：只有严格超过阈值才算显著改善。
            if improvement > self.config.min_improvement:
                self.best_metric = metric
                self.no_improve_rounds = 0
            else:
                self.no_improve_rounds += 1
            defender_converged = self.no_improve_rounds >= self.config.patience

        # GRPO 必须已优化、已重新评估；边界等于阈值时同样视为饱和。
        attacker_saturated = (
            attacker_optimized
            and after_rate is not None
            and after_rate <= self.config.evasion_threshold
        )
        should_stop = bool(defender_converged and attacker_saturated)

        state: Dict[str, Any] = {
            "round_id": round_id,
            "valid_metric": metric,
            "best_metric": self.best_metric,
            "improvement": improvement,
            "evasion_rate": before_rate,
            "post_grpo_evasion_rate": after_rate,
            "attacker_optimized": bool(attacker_optimized),
            "no_improve_rounds": self.no_improve_rounds,
            "defender_converged": bool(defender_converged),
            "attacker_saturated": bool(attacker_saturated),
            "should_stop": should_stop,
        }
        self.history.append(state)
        self._last_round_id = round_id
        return state

    def reset(self) -> None:
        self.best_metric = None
        self.no_improve_rounds = 0
        self.history.clear()
        self._last_round_id = None

    def summary(self) -> Dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "no_improve_rounds": self.no_improve_rounds,
            "last_round_id": self._last_round_id,
            "history": list(self.history),
            "should_stop": bool(self.history and self.history[-1]["should_stop"]),
        }


if __name__ == "__main__":
    stopper = StoppingCriteria(
        StoppingConfig(patience=3, min_improvement=0.005, evasion_threshold=0.05)
    )
    records = [
        # round, validation metric, pre-GRPO evasion rate, optimized, post-GRPO rate
        (0, 0.820, 0.30, False, None),
        (1, 0.850, 0.24, False, None),
        (2, 0.851, 0.12, False, None),
        (3, 0.852, 0.06, False, None),
        (4, 0.853, 0.04, True, 0.07),  # 优化后的攻击已恢复逃逸能力，不停止
        (5, 0.853, 0.03, True, 0.04),  # 两个条件均满足，停止
    ]
    for round_id, metric, before, optimized, after in records:
        result = stopper.update(
            round_id=round_id,
            valid_metric=metric,
            evasion_rate=before,
            attacker_optimized=optimized,
            post_grpo_evasion_rate=after,
        )
        print(result)
