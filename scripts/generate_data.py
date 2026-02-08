"""数据生成脚本: 使用 LHS 采样 + SPICE 仿真生成训练数据"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nerf_sizing.data.sampler import LatinHypercubeSampler, PVTCornerGenerator
from nerf_sizing.data.spice_interface import MockSpiceInterface
from nerf_sizing.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成电路仿真训练数据")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--output", type=str, default="data/train_data.npz", help="输出文件路径")
    parser.add_argument("--num-samples", type=int, default=None, help="样本数量 (覆盖配置)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_samples = args.num_samples or cfg["data"]["num_samples"]
    seed = cfg["data"].get("seed", 42)

    # 初始化采样器
    sampler = LatinHypercubeSampler(cfg["params"], seed=seed)
    pvt_gen = PVTCornerGenerator(cfg["pvt"])
    spice = MockSpiceInterface()

    # 生成参数样本
    logger.info(f"使用 LHS 采样 {num_samples} 个设计参数样本...")
    param_samples = sampler.sample(num_samples)

    # 生成 PVT Corners
    pvt_corners = pvt_gen.generate()
    num_corners = pvt_corners.shape[0]
    logger.info(f"PVT Corners 数量: {num_corners}")

    # 计算总样本数 (每个参数组合 x 每个 PVT Corner)
    total = num_samples * num_corners
    logger.info(f"总仿真次数: {total}")

    all_params = []
    all_pvt = []
    all_metrics = []
    all_feasibility = []

    for i in range(num_samples):
        for j in range(num_corners):
            metrics, feasibility = spice.simulate(param_samples[i], pvt_corners[j])
            all_params.append(param_samples[i])
            all_pvt.append(pvt_corners[j])
            all_metrics.append(metrics)
            all_feasibility.append(feasibility)

        if (i + 1) % 500 == 0:
            logger.info(f"  进度: {i + 1}/{num_samples} 参数组合")

    # 保存数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        params=np.array(all_params),
        pvt=np.array(all_pvt),
        metrics=np.array(all_metrics),
        feasibility=np.array(all_feasibility),
    )
    logger.info(f"数据已保存到 {output_path} (共 {total} 条记录)")


if __name__ == "__main__":
    main()

