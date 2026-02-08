"""优化流程测试: DesignLoss、参数优化器、参数裁剪"""

import numpy as np
import pytest
import torch

from nerf_sizing.model.circuit_field_net import CircuitFieldNet
from nerf_sizing.optimization.design_loss import DesignLoss
from nerf_sizing.optimization.optimizer import DesignOptimizer
from nerf_sizing.utils.normalization import Normalizer


class TestDesignLoss:
    """设计损失函数测试。"""

    @pytest.fixture
    def specs(self):
        return [
            {"name": "Gain", "target": 60.0, "direction": ">="},
            {"name": "BW", "target": 1e6, "direction": ">="},
            {"name": "Power", "target": 1e-3, "direction": "<="},
        ]

    @pytest.fixture
    def design_loss(self, specs):
        return DesignLoss(specs, feasibility_threshold=0.8, feasibility_weight=10.0)

    def test_all_specs_met_loss_zero(self, design_loss):
        """满足所有 Spec 且可行时 loss=0。"""
        # Gain=80 >= 60, BW=2e6 >= 1e6, Power=0.5e-3 <= 1e-3
        y_hat = torch.tensor([[80.0, 2e6, 0.5e-3]])
        sigma_hat = torch.tensor([[0.95]])

        losses = design_loss(y_hat, sigma_hat)
        assert losses["loss_spec"].item() < 1e-6
        assert losses["loss_feas"].item() < 1e-6

    def test_spec_violated_loss_positive(self, design_loss):
        """不满足 Spec 时 loss > 0。"""
        # Gain=40 < 60 (不满足)
        y_hat = torch.tensor([[40.0, 2e6, 0.5e-3]])
        sigma_hat = torch.tensor([[0.95]])

        losses = design_loss(y_hat, sigma_hat)
        assert losses["loss_spec"].item() > 0

    def test_feasibility_violated_loss_positive(self, design_loss):
        """可行性低于阈值时 loss > 0。"""
        y_hat = torch.tensor([[80.0, 2e6, 0.5e-3]])
        sigma_hat = torch.tensor([[0.3]])  # < 0.8

        losses = design_loss(y_hat, sigma_hat)
        assert losses["loss_feas"].item() > 0

    def test_gradient_direction(self, design_loss):
        """梯度方向正确: Gain 不足时，梯度应驱动 Gain 增大。"""
        # Gain 不满足
        y_hat = torch.tensor([[40.0, 2e6, 0.5e-3]], requires_grad=True)
        sigma_hat = torch.tensor([[0.95]])

        losses = design_loss(y_hat, sigma_hat)
        losses["loss"].backward()

        # dL/d(Gain) 应为负 (希望 Gain 增大 -> loss 减小)
        assert y_hat.grad[0, 0] < 0

    def test_power_over_spec(self, design_loss):
        """Power 超标时 loss > 0 且梯度驱动 Power 减小。"""
        y_hat = torch.tensor([[80.0, 2e6, 5e-3]], requires_grad=True)
        sigma_hat = torch.tensor([[0.95]])

        losses = design_loss(y_hat, sigma_hat)
        losses["loss"].backward()

        # dL/d(Power) 应为正 (希望 Power 减小)
        assert y_hat.grad[0, 2] > 0


class TestDesignOptimizer:
    """设计参数优化器测试。"""

    @pytest.fixture
    def setup(self):
        """创建模型、归一化器、优化器等。"""
        model = CircuitFieldNet(
            input_dim_p=4, input_dim_v=2,
            hidden_dims=[32, 32], output_dim_y=2,
        )

        # 模拟已训练的归一化器
        norm_p = Normalizer("minmax")
        norm_p.fit(torch.rand(100, 4))
        norm_v = Normalizer("minmax")
        norm_v.fit(torch.rand(100, 2))
        norm_y = Normalizer("minmax")
        norm_y.fit(torch.randn(100, 2) * 10)

        specs = [
            {"name": "M1", "target": 0.5, "direction": ">="},
            {"name": "M2", "target": 0.3, "direction": "<="},
        ]

        param_bounds = [
            {"name": "P1", "min": 0.0, "max": 1.0},
            {"name": "P2", "min": 0.0, "max": 1.0},
            {"name": "P3", "min": 0.0, "max": 1.0},
            {"name": "P4", "min": 0.0, "max": 1.0},
        ]

        config = {
            "num_starts": 4,
            "max_iters": 50,
            "learning_rate": 0.01,
            "feasibility_threshold": 0.5,
            "feasibility_weight": 1.0,
        }

        optimizer = DesignOptimizer(
            model=model,
            config=config,
            specs=specs,
            param_bounds=param_bounds,
            norm_p=norm_p,
            norm_v=norm_v,
            norm_y=norm_y,
            device="cpu",
        )

        pvt_corners = torch.tensor([[0.0, 0.5], [1.0, 0.5]])

        return optimizer, pvt_corners

    def test_optimize_runs(self, setup):
        """优化流程可正常运行。"""
        optimizer, pvt_corners = setup
        result = optimizer.optimize(pvt_corners, num_starts=4, max_iters=20)

        assert "best_params" in result
        assert "best_loss" in result
        assert result["best_params"].shape == (4,)
        assert isinstance(result["best_loss"], float)

    def test_params_within_bounds(self, setup):
        """优化过程中参数始终在物理合法范围内。"""
        optimizer, pvt_corners = setup
        result = optimizer.optimize(pvt_corners, num_starts=8, max_iters=50)

        all_params = result["all_params"]
        assert (all_params >= 0.0).all(), "参数低于下界"
        assert (all_params <= 1.0).all(), "参数高于上界"

    def test_evaluate(self, setup):
        """evaluate 方法可正常运行。"""
        optimizer, pvt_corners = setup
        result = optimizer.optimize(pvt_corners, num_starts=2, max_iters=10)

        eval_result = optimizer.evaluate(result["best_params"], pvt_corners)
        assert "corners" in eval_result
        assert len(eval_result["corners"]) == pvt_corners.shape[0]

        for corner in eval_result["corners"]:
            assert "sigma" in corner
            assert "y" in corner
            assert 0 <= corner["sigma"] <= 1

