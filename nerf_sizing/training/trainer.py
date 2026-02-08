"""训练器: 标准 PyTorch 训练循环"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..model.circuit_field_net import CircuitFieldNet
from ..utils.normalization import Normalizer
from .loss import TrainingLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制。

    Parameters
    ----------
    patience : int
        允许的无改善 epoch 数。
    min_delta : float
        最小改善量。
    """

    def __init__(self, patience: int = 50, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """检查是否应该停止。返回 True 表示应停止。"""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """训练器。

    Parameters
    ----------
    model : CircuitFieldNet
        代理模型。
    config : dict
        训练配置 (training 部分)。
    device : str
        计算设备。
    """

    def __init__(
        self,
        model: CircuitFieldNet,
        config: dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.criterion = TrainingLoss(
            lambda_feasibility=config.get("lambda_feasibility", 1.0),
        )
        self.optimizer = Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        epochs = config.get("epochs", 500)
        scheduler_type = config.get("scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = None

        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 50),
        )

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 归一化器
        self.norm_p = Normalizer("minmax")
        self.norm_v = Normalizer("minmax")
        self.norm_y = Normalizer("minmax")

        self.history: list[dict[str, float]] = []

    def fit_normalizers(
        self,
        train_loader: DataLoader,
    ) -> None:
        """根据训练数据拟合归一化器。"""
        all_p, all_v, all_y = [], [], []
        for batch in train_loader:
            all_p.append(batch["params"])
            all_v.append(batch["pvt"])
            all_y.append(batch["metrics"])
        all_p = torch.cat(all_p, dim=0)
        all_v = torch.cat(all_v, dim=0)
        all_y = torch.cat(all_y, dim=0)

        self.norm_p.fit(all_p)
        self.norm_v.fit(all_v)
        self.norm_y.fit(all_y)

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """单个训练 epoch。"""
        self.model.train()
        total_loss = 0.0
        total_perf = 0.0
        total_feas = 0.0
        n_batches = 0

        for batch in loader:
            p = self.norm_p.normalize(batch["params"].to(self.device))
            v = self.norm_v.normalize(batch["pvt"].to(self.device))
            y_gt = self.norm_y.normalize(batch["metrics"].to(self.device))
            sigma_gt = batch["feasibility"].to(self.device)

            sigma_hat, y_hat = self.model(p, v)
            losses = self.criterion(y_hat, y_gt, sigma_hat, sigma_gt)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            total_loss += losses["loss"].item()
            total_perf += losses["loss_perf"].item()
            total_feas += losses["loss_feas"].item()
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "loss_perf": total_perf / n_batches,
            "loss_feas": total_feas / n_batches,
        }

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> dict[str, float]:
        """验证 epoch。"""
        self.model.eval()
        total_loss = 0.0
        total_perf = 0.0
        total_feas = 0.0
        n_batches = 0

        for batch in loader:
            p = self.norm_p.normalize(batch["params"].to(self.device))
            v = self.norm_v.normalize(batch["pvt"].to(self.device))
            y_gt = self.norm_y.normalize(batch["metrics"].to(self.device))
            sigma_gt = batch["feasibility"].to(self.device)

            sigma_hat, y_hat = self.model(p, v)
            losses = self.criterion(y_hat, y_gt, sigma_hat, sigma_gt)

            total_loss += losses["loss"].item()
            total_perf += losses["loss_perf"].item()
            total_feas += losses["loss_feas"].item()
            n_batches += 1

        return {
            "val_loss": total_loss / n_batches,
            "val_loss_perf": total_perf / n_batches,
            "val_loss_feas": total_feas / n_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> list[dict[str, float]]:
        """执行完整训练。

        Parameters
        ----------
        train_loader : DataLoader
            训练数据加载器。
        val_loader : DataLoader | None
            验证数据加载器 (可选)。

        Returns
        -------
        list[dict[str, float]]
            每个 epoch 的训练/验证指标历史。
        """
        epochs = self.config.get("epochs", 500)
        self.fit_normalizers(train_loader)

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            record = {"epoch": epoch, **train_metrics}

            if val_loader is not None:
                val_metrics = self._eval_epoch(val_loader)
                record.update(val_metrics)
                monitor_loss = val_metrics["val_loss"]
            else:
                monitor_loss = train_metrics["loss"]

            self.history.append(record)

            if epoch % 50 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"loss={train_metrics['loss']:.6f} | "
                    f"perf={train_metrics['loss_perf']:.6f} | "
                    f"feas={train_metrics['loss_feas']:.6f}"
                    + (f" | val_loss={val_metrics['val_loss']:.6f}" if val_loader else "")
                )

            # 早停检查
            if self.early_stopping.step(monitor_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # 保存最终模型
        self.save_checkpoint("final.pt")
        return self.history

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> Path:
        """保存检查点。"""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "norm_p": self.norm_p.state_dict(),
                "norm_v": self.norm_v.state_dict(),
                "norm_y": self.norm_y.state_dict(),
                "history": self.history,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """加载检查点。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.norm_p.load_state_dict(ckpt["norm_p"])
        self.norm_v.load_state_dict(ckpt["norm_v"])
        self.norm_y.load_state_dict(ckpt["norm_y"])
        self.history = ckpt.get("history", [])
        logger.info(f"Checkpoint loaded from {path}")

