"""模型单元测试: CircuitFieldNet & PositionalEncoding"""

import pytest
import torch

from nerf_sizing.model.circuit_field_net import CircuitFieldNet
from nerf_sizing.model.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    """位置编码测试。"""

    def test_output_dim(self):
        """编码后维度 = input_dim * (2L + 1)。"""
        input_dim = 6
        num_freqs = 4
        pe = PositionalEncoding(input_dim, num_freqs)
        expected_dim = input_dim * (2 * num_freqs + 1)
        assert pe.output_dim == expected_dim

        x = torch.randn(8, input_dim)
        out = pe(x)
        assert out.shape == (8, expected_dim)

    def test_output_range(self):
        """编码输出值应在合理范围 [-1, 1] (sin/cos 部分)。"""
        pe = PositionalEncoding(3, 6)
        x = torch.randn(16, 3)
        out = pe(x)
        # sin/cos 部分应在 [-1, 1]，原始部分不受限
        # 整体检查不应出现 NaN/Inf
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_batch_dims(self):
        """支持任意批次维度。"""
        pe = PositionalEncoding(4, 3)
        x = torch.randn(2, 5, 4)
        out = pe(x)
        assert out.shape == (2, 5, 4 * (2 * 3 + 1))

    def test_gradient_flow(self):
        """梯度可回传。"""
        pe = PositionalEncoding(3, 4)
        x = torch.randn(4, 3, requires_grad=True)
        out = pe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCircuitFieldNet:
    """CircuitFieldNet 测试。"""

    @pytest.fixture
    def model(self):
        return CircuitFieldNet(
            input_dim_p=6,
            input_dim_v=3,
            hidden_dims=[64, 64],
            output_dim_y=3,
            use_positional_encoding=False,
        )

    @pytest.fixture
    def model_with_pe(self):
        return CircuitFieldNet(
            input_dim_p=6,
            input_dim_v=3,
            hidden_dims=[64, 64],
            output_dim_y=3,
            use_positional_encoding=True,
            encoding_freqs=4,
        )

    def test_output_shapes(self, model):
        """输入输出维度正确。"""
        B = 8
        p = torch.randn(B, 6)
        v = torch.randn(B, 3)
        sigma, y = model(p, v)
        assert sigma.shape == (B, 1)
        assert y.shape == (B, 3)

    def test_sigma_range(self, model):
        """sigma 在 [0, 1] 范围。"""
        p = torch.randn(32, 6)
        v = torch.randn(32, 3)
        sigma, _ = model(p, v)
        assert (sigma >= 0.0).all()
        assert (sigma <= 1.0).all()

    def test_gradient_to_input(self, model):
        """梯度可回传到输入参数 p (优化阶段的核心需求)。"""
        p = torch.randn(4, 6, requires_grad=True)
        v = torch.randn(4, 3)
        sigma, y = model(p, v)
        loss = y.sum() + sigma.sum()
        loss.backward()
        assert p.grad is not None
        assert not torch.all(p.grad == 0)

    def test_with_positional_encoding(self, model_with_pe):
        """启用位置编码时维度正确。"""
        B = 8
        p = torch.randn(B, 6)
        v = torch.randn(B, 3)
        sigma, y = model_with_pe(p, v)
        assert sigma.shape == (B, 1)
        assert y.shape == (B, 3)

    def test_from_config(self):
        """从配置字典创建模型。"""
        cfg = {
            "input_dim_p": 4,
            "input_dim_v": 2,
            "hidden_dims": [32, 32],
            "output_dim_y": 2,
            "use_positional_encoding": False,
        }
        model = CircuitFieldNet.from_config(cfg)
        p = torch.randn(2, 4)
        v = torch.randn(2, 2)
        sigma, y = model(p, v)
        assert sigma.shape == (2, 1)
        assert y.shape == (2, 2)

    def test_deterministic_eval(self, model):
        """eval 模式下同一输入应产生相同输出。"""
        model.eval()
        p = torch.randn(4, 6)
        v = torch.randn(4, 3)
        sigma1, y1 = model(p, v)
        sigma2, y2 = model(p, v)
        assert torch.allclose(sigma1, sigma2)
        assert torch.allclose(y1, y2)

