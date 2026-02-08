"""训练流程测试: 损失函数、归一化、训练收敛性"""

import numpy as np
import pytest
import torch

from nerf_sizing.data.dataset import CircuitDataset
from nerf_sizing.data.sampler import LatinHypercubeSampler, PVTCornerGenerator
from nerf_sizing.data.spice_interface import MockSpiceInterface
from nerf_sizing.model.circuit_field_net import CircuitFieldNet
from nerf_sizing.training.loss import TrainingLoss
from nerf_sizing.training.trainer import Trainer, EarlyStopping
from nerf_sizing.utils.normalization import Normalizer


class TestTrainingLoss:
    """训练损失函数测试。"""

    @pytest.fixture
    def criterion(self):
        return TrainingLoss(lambda_feasibility=1.0)

    def test_loss_nonnegative(self, criterion):
        """L_train 值非负。"""
        y_hat = torch.randn(8, 3)
        y_gt = torch.randn(8, 3)
        sigma_hat = torch.sigmoid(torch.randn(8, 1))
        sigma_gt = torch.rand(8)

        losses = criterion(y_hat, y_gt, sigma_hat, sigma_gt)
        assert losses["loss"].item() >= 0
        assert losses["loss_perf"].item() >= 0
        assert losses["loss_feas"].item() >= 0

    def test_gradient_nonzero(self, criterion):
        """梯度不为零。"""
        y_hat = torch.randn(8, 3, requires_grad=True)
        y_gt = torch.randn(8, 3)
        sigma_hat = torch.sigmoid(torch.randn(8, 1, requires_grad=True))
        sigma_gt = torch.rand(8)

        losses = criterion(y_hat, y_gt, sigma_hat, sigma_gt)
        losses["loss"].backward()
        assert y_hat.grad is not None
        assert not torch.all(y_hat.grad == 0)

    def test_perfect_prediction_near_zero(self, criterion):
        """完美预测时 loss 趋近 0。"""
        y_gt = torch.randn(16, 3)
        # 使用接近 0/1 的确定性标签，使 BCE(p,p) 也趋近 0
        sigma_gt = torch.cat([torch.ones(8), torch.zeros(8)])

        y_hat = y_gt.clone()
        sigma_hat = sigma_gt.clone().unsqueeze(-1)

        losses = criterion(y_hat, y_gt, sigma_hat, sigma_gt)
        assert losses["loss_perf"].item() < 1e-6
        # BCE(0,0)=0, BCE(1,1)=0，因此完美预测时 feas loss 也趋近 0
        assert losses["loss_feas"].item() < 1e-4


class TestNormalizer:
    """归一化工具测试。"""

    @pytest.mark.parametrize("method", ["minmax", "zscore"])
    def test_invertibility(self, method):
        """normalize -> denormalize 的可逆性。"""
        data = torch.randn(100, 5) * 10 + 3
        norm = Normalizer(method)
        norm.fit(data)

        normalized = norm.normalize(data)
        recovered = norm.denormalize(normalized)

        assert torch.allclose(data, recovered, atol=1e-5)

    def test_minmax_range(self):
        """MinMax 归一化后值在 [0, 1]。"""
        data = torch.randn(100, 3)
        norm = Normalizer("minmax")
        norm.fit(data)
        normalized = norm.normalize(data)
        assert normalized.min() >= -1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_zscore_stats(self):
        """Z-score 归一化后均值接近 0，标准差接近 1。"""
        data = torch.randn(1000, 3) * 5 + 10
        norm = Normalizer("zscore")
        norm.fit(data)
        normalized = norm.normalize(data)
        assert torch.abs(normalized.mean(dim=0)).max() < 0.1
        assert (torch.abs(normalized.std(dim=0) - 1.0)).max() < 0.1

    def test_state_dict_roundtrip(self):
        """state_dict 保存/加载一致性。"""
        data = torch.randn(50, 4)
        norm1 = Normalizer("minmax")
        norm1.fit(data)

        state = norm1.state_dict()
        norm2 = Normalizer()
        norm2.load_state_dict(state)

        x = torch.randn(10, 4)
        assert torch.allclose(norm1.normalize(x), norm2.normalize(x))

    def test_not_fitted_raises(self):
        """未拟合时调用 normalize/denormalize 应报错。"""
        norm = Normalizer("minmax")
        with pytest.raises(RuntimeError):
            norm.normalize(torch.randn(5, 3))
        with pytest.raises(RuntimeError):
            norm.denormalize(torch.randn(5, 3))


class TestEarlyStopping:
    """早停机制测试。"""

    def test_no_stop_with_improving(self):
        """持续改善时不应停止。"""
        es = EarlyStopping(patience=5)
        for i in range(10):
            assert not es.step(10.0 - i)

    def test_stop_after_patience(self):
        """超过 patience 后应停止。"""
        es = EarlyStopping(patience=3)
        es.step(1.0)  # 设置 best
        assert not es.step(2.0)  # counter=1
        assert not es.step(2.0)  # counter=2
        assert es.step(2.0)      # counter=3, 触发停止


class TestTrainingConvergence:
    """训练收敛性测试。"""

    def _generate_mock_data(self, n_samples=200, n_corners=3):
        """生成小规模 Mock 数据。"""
        param_specs = [
            {"name": "W1", "min": 1e-6, "max": 100e-6},
            {"name": "L1", "min": 0.18e-6, "max": 5e-6},
            {"name": "W2", "min": 1e-6, "max": 100e-6},
            {"name": "L2", "min": 0.18e-6, "max": 5e-6},
            {"name": "W3", "min": 1e-6, "max": 200e-6},
            {"name": "L3", "min": 0.18e-6, "max": 5e-6},
        ]
        pvt_config = {
            "process": [0.0],
            "voltage": [1.8],
            "temperature": [-40.0, 27.0, 125.0],
        }

        sampler = LatinHypercubeSampler(param_specs, seed=42)
        pvt_gen = PVTCornerGenerator(pvt_config)
        spice = MockSpiceInterface()

        params_arr = sampler.sample(n_samples)
        pvt_arr = pvt_gen.generate()

        all_p, all_v, all_m, all_f = [], [], [], []
        for p in params_arr:
            for v in pvt_arr:
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

    def test_loss_decreases(self):
        """使用 MockSpice 数据训练 50 epoch, loss 应下降。"""
        params, pvt, metrics, feasibility = self._generate_mock_data(n_samples=100)

        dataset = CircuitDataset(params, pvt, metrics, feasibility)
        loader = dataset.get_dataloader(batch_size=64, shuffle=True)

        model = CircuitFieldNet(
            input_dim_p=6, input_dim_v=3,
            hidden_dims=[64, 64], output_dim_y=3,
        )

        config = {
            "epochs": 50,
            "learning_rate": 1e-3,
            "lambda_feasibility": 1.0,
            "scheduler": "cosine",
            "early_stopping_patience": 999,  # 禁用早停
            "checkpoint_dir": "checkpoints_test",
        }

        trainer = Trainer(model, config, device="cpu")
        history = trainer.train(loader)

        # 前 5 epoch 的平均 loss 应 > 后 5 epoch 的平均 loss
        early_loss = np.mean([h["loss"] for h in history[:5]])
        late_loss = np.mean([h["loss"] for h in history[-5:]])
        assert late_loss < early_loss, (
            f"训练未收敛: early_loss={early_loss:.4f}, late_loss={late_loss:.4f}"
        )

        # 清理
        import shutil
        shutil.rmtree("checkpoints_test", ignore_errors=True)

