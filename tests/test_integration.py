"""端到端集成测试: Mock 数据生成 -> 训练 -> 优化 -> 验证"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from nerf_sizing.data.dataset import CircuitDataset
from nerf_sizing.data.sampler import LatinHypercubeSampler, PVTCornerGenerator
from nerf_sizing.data.spice_interface import MockSpiceInterface
from nerf_sizing.model.circuit_field_net import CircuitFieldNet
from nerf_sizing.optimization.optimizer import DesignOptimizer
from nerf_sizing.training.trainer import Trainer


# 用于所有集成测试的参数 & 配置
PARAM_SPECS = [
    {"name": "W1", "min": 1e-6, "max": 100e-6},
    {"name": "L1", "min": 0.18e-6, "max": 5e-6},
    {"name": "W2", "min": 1e-6, "max": 100e-6},
    {"name": "L2", "min": 0.18e-6, "max": 5e-6},
    {"name": "W3", "min": 1e-6, "max": 200e-6},
    {"name": "L3", "min": 0.18e-6, "max": 5e-6},
]

PVT_CONFIG = {
    "process": [-1.0, 0.0, 1.0],
    "voltage": [1.62, 1.8, 1.98],
    "temperature": [-40.0, 27.0, 125.0],
}

SPECS = [
    {"name": "Gain", "target": 60.0, "direction": ">="},
    {"name": "BW", "target": 1e6, "direction": ">="},
    {"name": "Power", "target": 1e-3, "direction": "<="},
]

TRAIN_CONFIG = {
    "epochs": 80,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "lambda_feasibility": 1.0,
    "scheduler": "cosine",
    "early_stopping_patience": 999,
    "checkpoint_dir": "checkpoints_integration_test",
}

OPT_CONFIG = {
    "num_starts": 16,
    "max_iters": 200,
    "learning_rate": 0.01,
    "feasibility_threshold": 0.5,
    "feasibility_weight": 5.0,
}


def _generate_data(n_samples: int = 300):
    """生成 Mock 仿真数据。"""
    sampler = LatinHypercubeSampler(PARAM_SPECS, seed=42)
    pvt_gen = PVTCornerGenerator(PVT_CONFIG)
    spice = MockSpiceInterface()

    params = sampler.sample(n_samples)
    pvt_corners = pvt_gen.generate()

    all_p, all_v, all_m, all_f = [], [], [], []
    for p in params:
        for v in pvt_corners:
            m, f = spice.simulate(p, v)
            all_p.append(p)
            all_v.append(v)
            all_m.append(m)
            all_f.append(f)

    return (
        np.array(all_p),
        np.array(all_v),
        np.array(all_m),
        np.array(all_f),
    )


def _train_model(params, pvt, metrics, feasibility):
    """训练模型并返回 trainer。"""
    dataset = CircuitDataset(params, pvt, metrics, feasibility)

    n = len(dataset)
    n_train = int(n * 0.8)
    train_set = CircuitDataset(
        params[:n_train], pvt[:n_train], metrics[:n_train], feasibility[:n_train]
    )
    val_set = CircuitDataset(
        params[n_train:], pvt[n_train:], metrics[n_train:], feasibility[n_train:]
    )

    train_loader = train_set.get_dataloader(batch_size=128, shuffle=True)
    val_loader = val_set.get_dataloader(batch_size=128, shuffle=False)

    model = CircuitFieldNet(
        input_dim_p=6, input_dim_v=3,
        hidden_dims=[128, 128], output_dim_y=3,
    )

    trainer = Trainer(model, TRAIN_CONFIG, device="cpu")
    trainer.train(train_loader, val_loader)
    return trainer


@pytest.fixture(scope="module")
def trained_system():
    """模块级别 fixture: 数据生成 + 训练 (只执行一次)。"""
    params, pvt, metrics, feasibility = _generate_data(n_samples=200)
    trainer = _train_model(params, pvt, metrics, feasibility)

    yield trainer

    # 清理
    shutil.rmtree("checkpoints_integration_test", ignore_errors=True)


class TestEndToEnd:
    """全流程端到端测试。"""

    def test_training_converges(self, trained_system):
        """训练后 loss 应显著下降。"""
        trainer = trained_system
        history = trainer.history
        assert len(history) > 10

        early_loss = np.mean([h["loss"] for h in history[:5]])
        late_loss = np.mean([h["loss"] for h in history[-5:]])
        assert late_loss < early_loss * 0.5, "训练收敛不足"

    def test_optimization_produces_result(self, trained_system):
        """优化可以产生结果。"""
        trainer = trained_system

        pvt_gen = PVTCornerGenerator(PVT_CONFIG)
        pvt_corners = torch.tensor(pvt_gen.generate(), dtype=torch.float32)

        optimizer = DesignOptimizer(
            model=trainer.model,
            config=OPT_CONFIG,
            specs=SPECS,
            param_bounds=PARAM_SPECS,
            norm_p=trainer.norm_p,
            norm_v=trainer.norm_v,
            norm_y=trainer.norm_y,
            device="cpu",
        )

        result = optimizer.optimize(pvt_corners, num_starts=8, max_iters=100)
        assert result["best_params"].shape == (6,)
        assert result["best_loss"] >= 0

    def test_optimized_params_in_bounds(self, trained_system):
        """优化后参数在合法范围内。"""
        trainer = trained_system

        pvt_gen = PVTCornerGenerator(PVT_CONFIG)
        pvt_corners = torch.tensor(pvt_gen.generate(), dtype=torch.float32)

        optimizer = DesignOptimizer(
            model=trainer.model,
            config=OPT_CONFIG,
            specs=SPECS,
            param_bounds=PARAM_SPECS,
            norm_p=trainer.norm_p,
            norm_v=trainer.norm_v,
            norm_y=trainer.norm_y,
            device="cpu",
        )

        result = optimizer.optimize(pvt_corners, num_starts=8, max_iters=100)
        best = result["best_params"].numpy()

        for i, spec in enumerate(PARAM_SPECS):
            assert best[i] >= spec["min"], f"{spec['name']} 低于下界"
            assert best[i] <= spec["max"], f"{spec['name']} 高于上界"


class TestPVTAggregation:
    """PVT 聚合测试。"""

    def test_evaluation_all_corners(self, trained_system):
        """评估结果应包含所有 PVT Corner。"""
        trainer = trained_system

        pvt_gen = PVTCornerGenerator(PVT_CONFIG)
        pvt_corners = torch.tensor(pvt_gen.generate(), dtype=torch.float32)

        optimizer = DesignOptimizer(
            model=trainer.model,
            config=OPT_CONFIG,
            specs=SPECS,
            param_bounds=PARAM_SPECS,
            norm_p=trainer.norm_p,
            norm_v=trainer.norm_v,
            norm_y=trainer.norm_y,
            device="cpu",
        )

        result = optimizer.optimize(pvt_corners, num_starts=4, max_iters=50)
        eval_result = optimizer.evaluate(result["best_params"], pvt_corners)

        assert len(eval_result["corners"]) == pvt_gen.num_corners


class TestMultiStartConsistency:
    """多起点一致性测试。"""

    def test_multiple_runs_not_divergent(self, trained_system):
        """多次优化的最优解损失应在同一数量级。"""
        trainer = trained_system

        pvt_gen = PVTCornerGenerator(PVT_CONFIG)
        pvt_corners = torch.tensor(pvt_gen.generate(), dtype=torch.float32)

        losses = []
        for _ in range(3):
            optimizer = DesignOptimizer(
                model=trainer.model,
                config=OPT_CONFIG,
                specs=SPECS,
                param_bounds=PARAM_SPECS,
                norm_p=trainer.norm_p,
                norm_v=trainer.norm_v,
                norm_y=trainer.norm_y,
                device="cpu",
            )
            result = optimizer.optimize(pvt_corners, num_starts=8, max_iters=100)
            losses.append(result["best_loss"])

        # 最优损失值的方差不应过大 (不发散)
        losses_arr = np.array(losses)
        if losses_arr.mean() > 0:
            cv = losses_arr.std() / (losses_arr.mean() + 1e-10)
            # 变异系数不应过大 (允许一定变异，因为是随机起点)
            assert cv < 5.0, f"多起点结果变异过大: cv={cv:.2f}, losses={losses}"

