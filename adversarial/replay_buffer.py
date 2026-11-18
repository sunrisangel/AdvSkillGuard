import os
import json
import random
from typing import List, Dict, Any, Optional


class ReplayBuffer:
    """
    历史逃逸样本池。

    用途：
    - 保存每一轮攻击侧成功逃逸的样本；
    - 在防御侧增量训练时，从历史样本中采样一部分，防止灾难性遗忘；
    - 支持保存到 jsonl 文件，方便中断后继续训练。
    """

    def __init__(
        self,
        max_size: int = 10000,
        save_path: Optional[str] = None,
        seed: int = 42,
    ):
        self.max_size = max_size
        self.save_path = save_path
        self.buffer: List[Dict[str, Any]] = []

        random.seed(seed)

        if save_path is not None and os.path.exists(save_path):
            self.load(save_path)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, sample: Dict[str, Any]) -> None:
        """
        添加单条样本。
        """
        self.buffer.append(sample)

        if len(self.buffer) > self.max_size:
            overflow = len(self.buffer) - self.max_size
            self.buffer = self.buffer[overflow:]

    def add_batch(self, samples: List[Dict[str, Any]]) -> None:
        """
        添加一批样本。
        """
        for sample in samples:
            self.add(sample)

    def sample(
        self,
        ratio: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        从历史样本中采样。

        参数：
            ratio: 按比例采样，例如 0.2 表示采样 20%
            num_samples: 指定采样数量

        二者都为空时，返回全部样本。
        """

        if not self.buffer:
            return []

        if num_samples is None:
            if ratio is None:
                num_samples = len(self.buffer)
            else:
                num_samples = int(len(self.buffer) * ratio)

        num_samples = max(0, min(num_samples, len(self.buffer)))

        if num_samples == 0:
            return []

        return random.sample(self.buffer, num_samples)

    def all(self) -> List[Dict[str, Any]]:
        """
        返回全部历史样本。
        """
        return list(self.buffer)

    def clear(self) -> None:
        """
        清空样本池。
        """
        self.buffer.clear()

    def save(self, path: Optional[str] = None) -> None:
        """
        保存为 jsonl 文件。
        """
        save_path = path or self.save_path

        if save_path is None:
            raise ValueError("save_path 未设置，无法保存 ReplayBuffer")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            for item in self.buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load(self, path: Optional[str] = None) -> None:
        """
        从 jsonl 文件加载样本。
        """
        load_path = path or self.save_path

        if load_path is None:
            raise ValueError("load_path 未设置，无法加载 ReplayBuffer")

        self.buffer = []

        with open(load_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.buffer.append(json.loads(line))

        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def stats(self) -> Dict[str, Any]:
        """
        返回样本池统计信息。
        """
        strategy_count = {}
        source_count = {}

        for item in self.buffer:
            strategies = item.get("selected_strategies", [])
            if isinstance(strategies, list):
                key = "+".join(strategies)
            else:
                key = str(strategies)

            strategy_count[key] = strategy_count.get(key, 0) + 1

            source = item.get("source_skill_dir", "unknown")
            source_count[source] = source_count.get(source, 0) + 1

        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "strategy_count": strategy_count,
            "source_count": source_count,
        }


if __name__ == "__main__":
    buffer = ReplayBuffer(
        max_size=5,
        save_path="experiments/adversarial/replay_buffer.jsonl",
    )

    for i in range(10):
        buffer.add({
            "id": i,
            "prompt": f"sample prompt {i}",
            "answer": "<verdict>恶意</verdict>",
            "selected_strategies": ["S1"],
            "source_skill_dir": f"skill_{i}",
        })

    print(buffer.stats())

    samples = buffer.sample(ratio=0.4)
    print("采样数量:", len(samples))

    buffer.save()