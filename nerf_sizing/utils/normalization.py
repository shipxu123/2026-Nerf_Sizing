"""输入/输出归一化工具 (Min-Max & Z-score)"""

from __future__ import annotations

import torch


class Normalizer:
    """支持 Min-Max 和 Z-score 两种归一化方式。

    Parameters
    ----------
    method : str
        归一化方法，``"minmax"`` 或 ``"zscore"``。
    """

    def __init__(self, method: str = "minmax") -> None:
        if method not in ("minmax", "zscore"):
            raise ValueError(f"不支持的归一化方法: {method}")
        self.method = method
        self._fitted = False

        # MinMax 参数
        self.data_min: torch.Tensor | None = None
        self.data_max: torch.Tensor | None = None

        # Z-score 参数
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "Normalizer":
        """根据数据计算归一化参数。

        Parameters
        ----------
        data : torch.Tensor
            形状 ``(N, D)`` 的数据张量。
        """
        if self.method == "minmax":
            self.data_min = data.min(dim=0).values
            self.data_max = data.max(dim=0).values
            # 防止除零
            diff = self.data_max - self.data_min
            diff[diff == 0] = 1.0
            self.data_max = self.data_min + diff
        else:
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            self.std[self.std == 0] = 1.0
        self._fitted = True
        return self

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """将数据归一化。"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 方法")
        if self.method == "minmax":
            return (data - self.data_min) / (self.data_max - self.data_min)
        else:
            return (data - self.mean) / self.std

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """将归一化数据还原。"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 方法")
        if self.method == "minmax":
            return data * (self.data_max - self.data_min) + self.data_min
        else:
            return data * self.std + self.mean

    def state_dict(self) -> dict:
        """导出归一化参数。"""
        state = {"method": self.method, "_fitted": self._fitted}
        if self.method == "minmax":
            state["data_min"] = self.data_min
            state["data_max"] = self.data_max
        else:
            state["mean"] = self.mean
            state["std"] = self.std
        return state

    def load_state_dict(self, state: dict) -> None:
        """加载归一化参数。"""
        self.method = state["method"]
        self._fitted = state["_fitted"]
        if self.method == "minmax":
            self.data_min = state["data_min"]
            self.data_max = state["data_max"]
        else:
            self.mean = state["mean"]
            self.std = state["std"]

