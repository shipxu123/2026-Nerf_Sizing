"""采样器测试: LHS 采样器 & PVT Corner 生成器"""

import numpy as np
import pytest

from nerf_sizing.data.sampler import LatinHypercubeSampler, PVTCornerGenerator


class TestLatinHypercubeSampler:
    """LHS 采样器测试。"""

    @pytest.fixture
    def param_specs(self):
        return [
            {"name": "W1", "min": 1e-6, "max": 100e-6},
            {"name": "L1", "min": 0.18e-6, "max": 5e-6},
            {"name": "W2", "min": 1e-6, "max": 100e-6},
        ]

    def test_sample_count(self, param_specs):
        """样本数量正确。"""
        sampler = LatinHypercubeSampler(param_specs, seed=42)
        samples = sampler.sample(100)
        assert samples.shape == (100, 3)

    def test_sample_within_bounds(self, param_specs):
        """样本在指定范围内。"""
        sampler = LatinHypercubeSampler(param_specs, seed=42)
        samples = sampler.sample(500)

        for i, spec in enumerate(param_specs):
            assert samples[:, i].min() >= spec["min"]
            assert samples[:, i].max() <= spec["max"]

    def test_distribution_uniformity(self, param_specs):
        """分布均匀性: 各区间样本比例不应偏差太大。"""
        sampler = LatinHypercubeSampler(param_specs, seed=42)
        n = 1000
        samples = sampler.sample(n)

        # 对每个参数维度检查分布
        for i, spec in enumerate(param_specs):
            col = samples[:, i]
            lower, upper = spec["min"], spec["max"]
            mid = (lower + upper) / 2

            # 上下半区各应接近 50%
            ratio_lower = np.sum(col < mid) / n
            assert 0.35 < ratio_lower < 0.65, (
                f"参数 {spec['name']} 分布不均匀: lower_ratio={ratio_lower:.2f}"
            )

    def test_reproducibility(self, param_specs):
        """相同种子产生相同结果。"""
        s1 = LatinHypercubeSampler(param_specs, seed=123)
        s2 = LatinHypercubeSampler(param_specs, seed=123)
        np.testing.assert_array_equal(s1.sample(50), s2.sample(50))

    def test_different_seeds(self, param_specs):
        """不同种子产生不同结果。"""
        s1 = LatinHypercubeSampler(param_specs, seed=1)
        s2 = LatinHypercubeSampler(param_specs, seed=2)
        assert not np.allclose(s1.sample(50), s2.sample(50))


class TestPVTCornerGenerator:
    """PVT Corner 生成器测试。"""

    @pytest.fixture
    def pvt_config(self):
        return {
            "process": [-1.0, 0.0, 1.0],
            "voltage": [1.62, 1.8, 1.98],
            "temperature": [-40.0, 27.0, 125.0],
        }

    def test_corner_count(self, pvt_config):
        """Corner 数量 = P * V * T。"""
        gen = PVTCornerGenerator(pvt_config)
        corners = gen.generate()
        expected = 3 * 3 * 3  # 27
        assert corners.shape == (expected, 3)
        assert gen.num_corners == expected

    def test_corner_values(self, pvt_config):
        """Corner 值应来自输入列表。"""
        gen = PVTCornerGenerator(pvt_config)
        corners = gen.generate()

        for row in corners:
            assert row[0] in pvt_config["process"]
            assert row[1] in pvt_config["voltage"]
            assert row[2] in pvt_config["temperature"]

    def test_all_combinations(self, pvt_config):
        """所有组合都应存在。"""
        gen = PVTCornerGenerator(pvt_config)
        corners = gen.generate()
        corner_set = set(map(tuple, corners))
        assert len(corner_set) == gen.num_corners

