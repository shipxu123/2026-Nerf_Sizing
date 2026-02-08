"""设计损失函数: Spec 约束 + 可行性惩罚"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DesignLoss(nn.Module):
    """设计阶段损失函数。

    L_design = sum_specs [ ReLU(target - predicted)^2  或  ReLU(predicted - target)^2 ]
               + mu * ReLU(-sigma + threshold)^2

    Parameters
    ----------
    specs : list[dict]
        Spec 列表，每个元素包含 ``name``, ``target``, ``direction`` (">=" 或 "<=")。
    feasibility_threshold : float
        可行性阈值。
    feasibility_weight : float
        可行性惩罚系数 mu。
    """

    def __init__(
        self,
        specs: list[dict[str, Any]],
        feasibility_threshold: float = 0.8,
        feasibility_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.feasibility_threshold = feasibility_threshold
        self.feasibility_weight = feasibility_weight

        # 预解析 spec 方向
        self.targets: list[float] = []
        self.directions: list[str] = []
        for spec in specs:
            self.targets.append(spec["target"])
            self.directions.append(spec["direction"])

    def forward(
        self,
        y_hat: torch.Tensor,
        sigma_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """计算设计损失。

        Parameters
        ----------
        y_hat : torch.Tensor
            性能预测 (反归一化后)，形状 ``(B, M)``。
        sigma_hat : torch.Tensor
            可行性预测，形状 ``(B, 1)``。

        Returns
        -------
        dict[str, torch.Tensor]
            包含 ``loss``, ``loss_spec``, ``loss_feas``, 以及每个 spec 的单独损失。
        """
        device = y_hat.device
        loss_spec = torch.tensor(0.0, device=device)
        spec_details: dict[str, torch.Tensor] = {}

        for i, (target_val, direction) in enumerate(zip(self.targets, self.directions)):
            target = torch.tensor(target_val, device=device, dtype=y_hat.dtype)
            predicted = y_hat[:, i]

            if direction == ">=":
                # 当 predicted < target 时有惩罚
                violation = torch.relu(target - predicted)
            else:  # "<="
                # 当 predicted > target 时有惩罚
                violation = torch.relu(predicted - target)

            spec_loss = (violation ** 2).mean()
            loss_spec = loss_spec + spec_loss
            spec_details[f"loss_spec_{self.specs[i]['name']}"] = spec_loss.detach()

        # 可行性惩罚
        threshold = torch.tensor(self.feasibility_threshold, device=device, dtype=sigma_hat.dtype)
        feas_violation = torch.relu(threshold - sigma_hat.squeeze(-1))
        loss_feas = self.feasibility_weight * (feas_violation ** 2).mean()

        total = loss_spec + loss_feas

        return {
            "loss": total,
            "loss_spec": loss_spec.detach(),
            "loss_feas": loss_feas.detach(),
            **spec_details,
        }

