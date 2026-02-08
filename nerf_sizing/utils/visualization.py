"""可视化工具: 训练曲线、优化轨迹、参数空间"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: list[dict[str, float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """绘制训练损失曲线。

    Parameters
    ----------
    history : list[dict[str, float]]
        训练历史记录列表。
    save_path : str | Path | None
        图片保存路径 (可选)。

    Returns
    -------
    plt.Figure
        Matplotlib 图对象。
    """
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    loss_perf = [h["loss_perf"] for h in history]
    loss_feas = [h["loss_feas"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, loss, "b-", linewidth=1.5)
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss_perf, "r-", linewidth=1.5)
    axes[1].set_title("Performance Loss (MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, loss_feas, "g-", linewidth=1.5)
    axes[2].set_title("Feasibility Loss (BCE)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    # 如有验证损失则叠加
    if "val_loss" in history[0]:
        val_loss = [h["val_loss"] for h in history]
        axes[0].plot(epochs, val_loss, "b--", linewidth=1.5, label="val")
        axes[0].legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_optimization_trajectory(
    history: list[dict[str, float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """绘制优化损失下降轨迹。

    Parameters
    ----------
    history : list[dict[str, float]]
        优化历史记录。
    save_path : str | Path | None
        图片保存路径。
    """
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, "b-o", markersize=3, linewidth=1.5)
    ax.set_title("Design Optimization Trajectory")
    ax.set_xlabel("Step")
    ax.set_ylabel("Design Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_param_comparison(
    param_names: list[str],
    param_values: np.ndarray,
    param_bounds: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """绘制优化后参数在范围中的位置 (条形图)。

    Parameters
    ----------
    param_names : list[str]
        参数名称列表。
    param_values : np.ndarray
        优化后的参数值。
    param_bounds : list[dict]
        参数边界。
    save_path : str | Path | None
        图片保存路径。
    """
    n = len(param_names)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.8)))

    lower = np.array([b["min"] for b in param_bounds])
    upper = np.array([b["max"] for b in param_bounds])

    # 归一化到 [0, 1] 用于显示
    normalized = (param_values - lower) / (upper - lower + 1e-30)

    y_pos = np.arange(n)
    ax.barh(y_pos, normalized, height=0.6, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized Position (0=min, 1=max)")
    ax.set_title("Optimized Parameters (Normalized)")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    # 添加数值标注
    for i, (val, name) in enumerate(zip(param_values, param_names)):
        ax.text(normalized[i] + 0.02, i, f"{val:.2e}", va="center", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

