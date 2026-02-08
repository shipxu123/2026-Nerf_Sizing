"""LHS 采样器 & PVT Corner 生成器"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy.stats.qmc import LatinHypercube


class LatinHypercubeSampler:
    """拉丁超立方采样器，用于在设计参数空间中生成均匀分布的样本。

    Parameters
    ----------
    param_specs : list[dict]
        参数规格列表，每个元素包含 ``name``, ``min``, ``max``。
    seed : int | None
        随机种子。
    """

    def __init__(self, param_specs: list[dict[str, Any]], seed: int | None = 42) -> None:
        self.param_specs = param_specs
        self.dim = len(param_specs)
        self.seed = seed

        self.names = [p["name"] for p in param_specs]
        self.lower = np.array([p["min"] for p in param_specs], dtype=np.float64)
        self.upper = np.array([p["max"] for p in param_specs], dtype=np.float64)

    def sample(self, n: int) -> np.ndarray:
        """生成 ``n`` 个 LHS 样本。

        Parameters
        ----------
        n : int
            样本数量。

        Returns
        -------
        np.ndarray
            形状 ``(n, dim)`` 的样本数组，值在各参数的 [min, max] 范围内。
        """
        sampler = LatinHypercube(d=self.dim, seed=self.seed)
        unit_samples = sampler.random(n=n)  # (n, dim) in [0, 1]
        # 线性映射到实际范围
        samples = self.lower + unit_samples * (self.upper - self.lower)
        return samples


class PVTCornerGenerator:
    """PVT Corner 网格生成器。

    Parameters
    ----------
    pvt_config : dict
        PVT 配置字典，包含 ``process``, ``voltage``, ``temperature`` 列表。
    """

    def __init__(self, pvt_config: dict[str, list[float]]) -> None:
        self.process_vals = pvt_config["process"]
        self.voltage_vals = pvt_config["voltage"]
        self.temperature_vals = pvt_config["temperature"]

    def generate(self) -> np.ndarray:
        """生成所有 PVT Corner 组合。

        Returns
        -------
        np.ndarray
            形状 ``(num_corners, 3)`` 的数组，列为 [process, voltage, temperature]。
        """
        corners = list(product(
            self.process_vals,
            self.voltage_vals,
            self.temperature_vals,
        ))
        return np.array(corners, dtype=np.float64)

    @property
    def num_corners(self) -> int:
        """PVT Corner 总数。"""
        return len(self.process_vals) * len(self.voltage_vals) * len(self.temperature_vals)

