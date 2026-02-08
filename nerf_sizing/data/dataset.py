"""PyTorch Dataset / DataLoader 用于电路仿真数据"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CircuitDataset(Dataset):
    """电路仿真数据集。

    Parameters
    ----------
    params : np.ndarray
        设计参数，形状 ``(N_samples, dim_p)``。
    pvt : np.ndarray
        PVT 条件，形状 ``(N_samples, dim_v)``。
    metrics : np.ndarray
        性能指标，形状 ``(N_samples, dim_y)``。
    feasibility : np.ndarray
        可行性标量，形状 ``(N_samples,)``。
    """

    def __init__(
        self,
        params: np.ndarray,
        pvt: np.ndarray,
        metrics: np.ndarray,
        feasibility: np.ndarray,
    ) -> None:
        self.params = torch.tensor(params, dtype=torch.float32)
        self.pvt = torch.tensor(pvt, dtype=torch.float32)
        self.metrics = torch.tensor(metrics, dtype=torch.float32)
        self.feasibility = torch.tensor(feasibility, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "params": self.params[idx],
            "pvt": self.pvt[idx],
            "metrics": self.metrics[idx],
            "feasibility": self.feasibility[idx],
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """自定义 collate 函数。"""
        return {
            "params": torch.stack([b["params"] for b in batch]),
            "pvt": torch.stack([b["pvt"] for b in batch]),
            "metrics": torch.stack([b["metrics"] for b in batch]),
            "feasibility": torch.stack([b["feasibility"] for b in batch]),
        }

    def get_dataloader(self, batch_size: int = 256, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """创建 DataLoader。"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

