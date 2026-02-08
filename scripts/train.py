"""训练入口脚本: 加载数据 -> 训练代理模型"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nerf_sizing.data.dataset import CircuitDataset
from nerf_sizing.model.circuit_field_net import CircuitFieldNet
from nerf_sizing.training.trainer import Trainer
from nerf_sizing.utils.config import load_config
from nerf_sizing.utils.visualization import plot_training_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 CircuitFieldNet 代理模型")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--data", type=str, default="data/train_data.npz", help="训练数据路径")
    parser.add_argument("--device", type=str, default="auto", help="计算设备 (cpu/cuda/auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info(f"加载数据: {args.data}")
    data = np.load(args.data)
    params = data["params"]
    pvt = data["pvt"]
    metrics = data["metrics"]
    feasibility = data["feasibility"]
    logger.info(f"数据形状: params={params.shape}, pvt={pvt.shape}, metrics={metrics.shape}")

    # 划分训练/验证集
    train_ratio = cfg["data"].get("train_ratio", 0.8)
    n_total = len(params)
    n_train = int(n_total * train_ratio)

    indices = np.random.RandomState(cfg["data"].get("seed", 42)).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = CircuitDataset(
        params[train_idx], pvt[train_idx], metrics[train_idx], feasibility[train_idx]
    )
    val_dataset = CircuitDataset(
        params[val_idx], pvt[val_idx], metrics[val_idx], feasibility[val_idx]
    )

    batch_size = cfg["training"].get("batch_size", 256)
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False)

    # 创建模型
    model = CircuitFieldNet.from_config(cfg["model"])
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    trainer = Trainer(model, cfg["training"], device=device)
    history = trainer.train(train_loader, val_loader)

    # 可视化
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, save_path=output_dir / "training_curves.png")
    logger.info(f"训练曲线已保存到 {output_dir / 'training_curves.png'}")

    logger.info("训练完成!")


if __name__ == "__main__":
    main()

