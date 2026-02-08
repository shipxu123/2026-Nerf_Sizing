"""设计参数优化器 (阶段二): 冻结网络权重，优化输入参数"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from ..model.circuit_field_net import CircuitFieldNet
from ..utils.normalization import Normalizer
from .design_loss import DesignLoss

logger = logging.getLogger(__name__)


class DesignOptimizer:
    """设计参数优化器。

    冻结网络权重，将设计参数 p 设为可优化变量，通过梯度下降寻找满足 Spec 的参数。

    Parameters
    ----------
    model : CircuitFieldNet
        已训练的代理模型。
    config : dict
        优化配置 (optimization 部分)。
    specs : list[dict]
        设计指标列表。
    param_bounds : list[dict]
        参数边界列表，每个元素包含 ``name``, ``min``, ``max``。
    norm_p : Normalizer
        参数归一化器 (训练时拟合的)。
    norm_v : Normalizer
        PVT 归一化器。
    norm_y : Normalizer
        性能归一化器。
    device : str
        计算设备。
    """

    def __init__(
        self,
        model: CircuitFieldNet,
        config: dict[str, Any],
        specs: list[dict[str, Any]],
        param_bounds: list[dict[str, Any]],
        norm_p: Normalizer,
        norm_v: Normalizer,
        norm_y: Normalizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        # 冻结所有网络权重
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.config = config
        self.device = device

        self.design_loss = DesignLoss(
            specs=specs,
            feasibility_threshold=config.get("feasibility_threshold", 0.8),
            feasibility_weight=config.get("feasibility_weight", 10.0),
        )

        self.param_bounds = param_bounds
        self.lower = torch.tensor(
            [p["min"] for p in param_bounds], dtype=torch.float32, device=device
        )
        self.upper = torch.tensor(
            [p["max"] for p in param_bounds], dtype=torch.float32, device=device
        )

        self.norm_p = norm_p
        self.norm_v = norm_v
        self.norm_y = norm_y

    def _random_init(self, num_starts: int) -> torch.Tensor:
        """随机初始化多组设计参数。"""
        # 在 [lower, upper] 范围内均匀采样
        rand = torch.rand(num_starts, len(self.param_bounds), device=self.device)
        params = self.lower + rand * (self.upper - self.lower)
        return params

    def _clamp_params(self, params: torch.Tensor) -> torch.Tensor:
        """将参数裁剪到合法范围。"""
        return torch.clamp(params, min=self.lower, max=self.upper)

    def optimize(
        self,
        pvt_corners: torch.Tensor,
        num_starts: int | None = None,
        max_iters: int | None = None,
        lr: float | None = None,
    ) -> dict[str, Any]:
        """执行多起点并行优化。

        Parameters
        ----------
        pvt_corners : torch.Tensor
            PVT Corner 矩阵，形状 ``(C, K)``。
        num_starts : int | None
            起点数量 (默认使用配置值)。
        max_iters : int | None
            最大迭代次数。
        lr : float | None
            学习率。

        Returns
        -------
        dict[str, Any]
            包含 ``best_params``, ``best_loss``, ``all_params``, ``history``。
        """
        num_starts = num_starts or self.config.get("num_starts", 32)
        max_iters = max_iters or self.config.get("max_iters", 1000)
        lr = lr or self.config.get("learning_rate", 0.01)

        pvt_corners = pvt_corners.to(self.device)
        num_corners = pvt_corners.shape[0]

        # 初始化参数 (num_starts, dim_p)
        params = self._random_init(num_starts)
        params = params.detach().requires_grad_(True)

        optimizer = torch.optim.Adam([params], lr=lr)
        history: list[dict[str, float]] = []

        for step in range(1, max_iters + 1):
            optimizer.zero_grad()

            # PVT 聚合: 在所有 Corner 上平均损失
            total_loss = torch.tensor(0.0, device=self.device)

            for c in range(num_corners):
                v = pvt_corners[c].unsqueeze(0).expand(num_starts, -1)  # (S, K)

                # 归一化输入
                p_norm = self.norm_p.normalize(params)
                v_norm = self.norm_v.normalize(v)

                sigma_hat, y_hat_norm = self.model(p_norm, v_norm)

                # 反归一化性能输出
                y_hat = self.norm_y.denormalize(y_hat_norm)

                losses = self.design_loss(y_hat, sigma_hat)
                total_loss = total_loss + losses["loss"]

            avg_loss = total_loss / num_corners
            avg_loss.backward()
            optimizer.step()

            # 裁剪到合法范围
            with torch.no_grad():
                params.data = self._clamp_params(params.data)

            if step % 100 == 0 or step == 1:
                logger.info(f"Step {step}/{max_iters} | loss={avg_loss.item():.6f}")
                history.append({"step": step, "loss": avg_loss.item()})

        # 找到最优起点
        with torch.no_grad():
            final_losses = []
            for s in range(num_starts):
                p_s = params[s].unsqueeze(0)
                loss_s = torch.tensor(0.0, device=self.device)
                for c in range(num_corners):
                    v = pvt_corners[c].unsqueeze(0)
                    p_norm = self.norm_p.normalize(p_s)
                    v_norm = self.norm_v.normalize(v)
                    sigma_hat, y_hat_norm = self.model(p_norm, v_norm)
                    y_hat = self.norm_y.denormalize(y_hat_norm)
                    losses = self.design_loss(y_hat, sigma_hat)
                    loss_s = loss_s + losses["loss"]
                final_losses.append(loss_s.item() / num_corners)

            best_idx = int(np.argmin(final_losses))
            best_params = params[best_idx].detach().cpu()
            best_loss = final_losses[best_idx]

        return {
            "best_params": best_params,
            "best_loss": best_loss,
            "best_idx": best_idx,
            "all_params": params.detach().cpu(),
            "all_losses": final_losses,
            "history": history,
        }

    def evaluate(
        self,
        params: torch.Tensor,
        pvt_corners: torch.Tensor,
    ) -> dict[str, Any]:
        """评估一组参数在所有 PVT Corner 下的性能。

        Parameters
        ----------
        params : torch.Tensor
            设计参数，形状 ``(dim_p,)``。
        pvt_corners : torch.Tensor
            PVT Corner，形状 ``(C, K)``。

        Returns
        -------
        dict[str, Any]
            包含每个 Corner 的预测性能和可行性。
        """
        self.model.eval()
        params = params.to(self.device).unsqueeze(0)  # (1, dim_p)
        pvt_corners = pvt_corners.to(self.device)
        results: list[dict[str, Any]] = []

        with torch.no_grad():
            for c in range(pvt_corners.shape[0]):
                v = pvt_corners[c].unsqueeze(0)
                p_norm = self.norm_p.normalize(params)
                v_norm = self.norm_v.normalize(v)
                sigma_hat, y_hat_norm = self.model(p_norm, v_norm)
                y_hat = self.norm_y.denormalize(y_hat_norm)

                results.append({
                    "pvt": pvt_corners[c].cpu().numpy(),
                    "sigma": sigma_hat.item(),
                    "y": y_hat.squeeze(0).cpu().numpy(),
                })

        return {"corners": results}

