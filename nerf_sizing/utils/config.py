"""YAML 配置加载工具"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """加载 YAML 配置文件并返回字典。

    Parameters
    ----------
    path : str | Path
        配置文件路径，默认为 ``configs/default.yaml``。

    Returns
    -------
    dict[str, Any]
        解析后的配置字典。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

