import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


SCRIPT_EXTENSIONS = {
    ".py", ".sh", ".bash", ".js", ".ts", ".java", ".go",
    ".rs", ".cpp", ".c", ".cs", ".php", ".rb",
    ".json", ".yaml", ".yml", ".toml", ".xml"
}

IGNORE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode",
    "node_modules", "dist", "build", ".venv", "venv"
}


def read_text_file(file_path: str) -> str:
    """
    读取文本文件，自动尝试多种编码。
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin-1"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法解码文件: {file_path}"
    )


def find_skill_md(skill_dir: str) -> Optional[str]:
    """
    在 Skill 目录中查找 SKILL.md。
    优先查找根目录下的 SKILL.md；
    如果不存在，则在一级子目录中查找。
    """
    root = Path(skill_dir)

    direct_path = root / "SKILL.md"
    if direct_path.exists():
        return str(direct_path)

    for child in root.iterdir():
        if child.is_dir() and child.name not in IGNORE_DIRS:
            candidate = child / "SKILL.md"
            if candidate.exists():
                return str(candidate)

    return None


def collect_script_files(
    skill_dir: str,
    max_files: int = 30,
    max_chars_per_file: int = 8000
) -> Dict[str, str]:
    """
    收集 Skill 目录下的脚本文件和配置文件内容。
    """
    skill_dir_path = Path(skill_dir)
    scripts = {}

    for root, dirs, files in os.walk(skill_dir_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file_name in files:
            file_path = Path(root) / file_name

            if file_path.name == "SKILL.md":
                continue

            if file_path.suffix.lower() not in SCRIPT_EXTENSIONS:
                continue

            try:
                content = read_text_file(str(file_path))
            except Exception:
                continue

            relative_path = str(file_path.relative_to(skill_dir_path))
            scripts[relative_path] = content[:max_chars_per_file]

            if len(scripts) >= max_files:
                return scripts

    return scripts


def load_skill(skill_dir: str) -> Dict[str, Any]:
    """
    加载单个 Skill，返回统一结构。
    """
    skill_md_path = find_skill_md(skill_dir)

    if skill_md_path is None:
        return {
            "skill_dir": skill_dir,
            "skill_md_path": None,
            "skill_md": "",
            "scripts": {},
            "valid": False,
            "error": "SKILL.md not found"
        }

    try:
        skill_md = read_text_file(skill_md_path)
    except Exception as e:
        return {
            "skill_dir": skill_dir,
            "skill_md_path": skill_md_path,
            "skill_md": "",
            "scripts": {},
            "valid": False,
            "error": str(e)
        }

    scripts = collect_script_files(skill_dir)

    return {
        "skill_dir": skill_dir,
        "skill_md_path": skill_md_path,
        "skill_md": skill_md,
        "scripts": scripts,
        "valid": True,
        "error": None
    }


def format_scripts_for_prompt(scripts: Dict[str, str]) -> str:
    """
    将脚本文件内容格式化为 Prompt 中可使用的字符串。
    """
    if not scripts:
        return "无"

    blocks = []

    for file_name, content in scripts.items():
        block = (
            f"### 文件: {file_name}\n"
            f"```text\n"
            f"{content}\n"
            f"```"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_defender_input(skill_data: Dict[str, Any]) -> str:
    """
    构造防御侧模型输入文本。
    """
    skill_md = skill_data.get("skill_md", "")
    scripts = skill_data.get("scripts", {})

    script_text = format_scripts_for_prompt(scripts)

    prompt_input = f"""=== SKILL.md ===
{skill_md}

=== 脚本文件 ===
{script_text}
"""
    return prompt_input


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    读取 jsonl 格式数据集。
    每行格式示例：
    {
      "skill_dir": "...",
      "label": "malicious"
    }
    """
    dataset = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dataset.append(json.loads(line))

    return dataset


def save_jsonl(data: List[Dict[str, Any]], save_path: str) -> None:
    """
    保存 jsonl 文件。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_dataset(
    skill_dirs: List[str],
    label: str
) -> List[Dict[str, Any]]:
    """
    根据 Skill 文件夹列表构造数据集。
    label 可取：
    - malicious
    - benign
    """
    samples = []

    for skill_dir in skill_dirs:
        skill_data = load_skill(skill_dir)

        sample = {
            "skill_dir": skill_dir,
            "label": label,
            "valid": skill_data["valid"],
            "error": skill_data["error"],
            "skill_md_path": skill_data["skill_md_path"],
            "skill_md": skill_data["skill_md"],
            "scripts": skill_data["scripts"],
            "model_input": build_defender_input(skill_data)
            if skill_data["valid"] else ""
        }

        samples.append(sample)

    return samples


def scan_skill_dirs(root_dir: str) -> List[str]:
    """
    扫描某个根目录下的所有 Skill 文件夹。
    只要目录本身或一级子目录中包含 SKILL.md，就认为是一个 Skill。
    """
    root = Path(root_dir)
    skill_dirs = []

    for child in root.iterdir():
        if not child.is_dir():
            continue

        if find_skill_md(str(child)) is not None:
            skill_dirs.append(str(child))

    return skill_dirs


if __name__ == "__main__":
    # 简单测试
    example_dir = "../data/raw"

    if os.path.exists(example_dir):
        skill_dirs = scan_skill_dirs(example_dir)
        print(f"发现 Skill 数量: {len(skill_dirs)}")

        if skill_dirs:
            sample = load_skill(skill_dirs[0])
            print("是否有效:", sample["valid"])
            print("SKILL.md 路径:", sample["skill_md_path"])
            print("脚本文件数量:", len(sample["scripts"]))
            print(build_defender_input(sample)[:1000])