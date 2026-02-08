"""训练损失函数: MSE (性能) + BCE (可行性)"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrainingLoss(nn.Module):
    """训练阶段的组合损失函数。

    L_train = MSE(y_GT, y_hat) + lambda * BCE(sigma_GT, sigma_hat)

    Parameters
    ----------
    lambda_feasibility : float
        可行性损失的平衡系数。
    """

    def __init__(self, lambda_feasibility: float = 1.0) -> None:
        super().__init__()
        self.lambda_feasibility = lambda_feasibility
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        y_hat: torch.Tensor,
        y_gt: torch.Tensor,
        sigma_hat: torch.Tensor,
        sigma_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """计算训练损失。

        Parameters
        ----------
        y_hat : torch.Tensor
            性能预测，形状 ``(B, M)``。
        y_gt : torch.Tensor
            性能真值，形状 ``(B, M)``。
        sigma_hat : torch.Tensor
            可行性预测，形状 ``(B, 1)``。
        sigma_gt : torch.Tensor
            可行性真值，形状 ``(B,)`` 或 ``(B, 1)``。

        Returns
        -------
        dict[str, torch.Tensor]
            包含 ``loss``, ``loss_perf``, ``loss_feas``。
        """
        loss_perf = self.mse(y_hat, y_gt)

        sigma_gt = sigma_gt.view_as(sigma_hat)
        loss_feas = self.bce(sigma_hat, sigma_gt)

        total = loss_perf + self.lambda_feasibility * loss_feas

        return {
            "loss": total,
            "loss_perf": loss_perf.detach(),
            "loss_feas": loss_feas.detach(),
        }

