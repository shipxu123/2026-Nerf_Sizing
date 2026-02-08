"""推理/优化入口脚本: 加载训练好的模型 -> 优化设计参数"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nerf_sizing.data.sampler import PVTCornerGenerator
from nerf_sizing.model.circuit_field_net import CircuitFieldNet
from nerf_sizing.optimization.optimizer import DesignOptimizer
from nerf_sizing.utils.config import load_config
from nerf_sizing.utils.normalization import Normalizer
from nerf_sizing.utils.visualization import plot_optimization_trajectory, plot_param_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="优化电路设计参数")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt", help="模型检查点路径")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"使用设备: {device}")

    # 加载模型
    model = CircuitFieldNet.from_config(cfg["model"])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"模型已从 {args.checkpoint} 加载")

    # 加载归一化器
    norm_p = Normalizer()
    norm_p.load_state_dict(ckpt["norm_p"])
    norm_v = Normalizer()
    norm_v.load_state_dict(ckpt["norm_v"])
    norm_y = Normalizer()
    norm_y.load_state_dict(ckpt["norm_y"])

    # 生成 PVT Corners
    pvt_gen = PVTCornerGenerator(cfg["pvt"])
    pvt_corners = torch.tensor(pvt_gen.generate(), dtype=torch.float32)
    logger.info(f"PVT Corners 数量: {pvt_corners.shape[0]}")

    # 打印设计指标
    logger.info("设计指标 (Specs):")
    for spec in cfg["specs"]:
        logger.info(f"  {spec['name']} {spec['direction']} {spec['target']}")

    # 创建优化器
    design_optimizer = DesignOptimizer(
        model=model,
        config=cfg["optimization"],
        specs=cfg["specs"],
        param_bounds=cfg["params"],
        norm_p=norm_p,
        norm_v=norm_v,
        norm_y=norm_y,
        device=device,
    )

    # 执行优化
    logger.info("开始优化...")
    result = design_optimizer.optimize(pvt_corners)

    best_params = result["best_params"].numpy()
    best_loss = result["best_loss"]
    logger.info(f"最优损失: {best_loss:.6f}")
    logger.info("最优参数:")
    for name, val in zip([p["name"] for p in cfg["params"]], best_params):
        logger.info(f"  {name} = {val:.6e}")

    # 在所有 Corner 下评估最优参数
    eval_result = design_optimizer.evaluate(result["best_params"], pvt_corners)
    logger.info("所有 PVT Corner 下的性能:")
    spec_names = [s["name"] for s in cfg["specs"]]
    for corner in eval_result["corners"]:
        pvt_str = f"P={corner['pvt'][0]:.1f}, V={corner['pvt'][1]:.2f}, T={corner['pvt'][2]:.0f}"
        perf_str = ", ".join(f"{name}={val:.4f}" for name, val in zip(spec_names, corner["y"]))
        logger.info(f"  [{pvt_str}] sigma={corner['sigma']:.3f} | {perf_str}")

    # 可视化
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if result["history"]:
        plot_optimization_trajectory(result["history"], save_path=output_dir / "optimization_trajectory.png")

    param_names = [p["name"] for p in cfg["params"]]
    plot_param_comparison(param_names, best_params, cfg["params"], save_path=output_dir / "param_comparison.png")

    logger.info(f"可视化已保存到 {output_dir}")
    logger.info("优化完成!")


if __name__ == "__main__":
    main()

