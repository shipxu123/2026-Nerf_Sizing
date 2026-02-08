"""SPICE 仿真器接口 (抽象层 + Mock 实现)"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class SpiceInterface(abc.ABC):
    """SPICE 仿真器抽象基类。

    子类需实现 ``simulate`` 方法。
    """

    @abc.abstractmethod
    def simulate(
        self,
        params: np.ndarray,
        pvt: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """运行仿真并返回性能指标和可行性。

        Parameters
        ----------
        params : np.ndarray
            设计参数向量，形状 ``(N,)``。
        pvt : np.ndarray
            PVT 条件向量，形状 ``(K,)``。

        Returns
        -------
        metrics : np.ndarray
            性能指标向量，形状 ``(M,)``。
        feasibility : float
            可行性标量 (0~1)。
        """
        ...

    def simulate_batch(
        self,
        params_batch: np.ndarray,
        pvt_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量仿真。

        Parameters
        ----------
        params_batch : np.ndarray
            形状 ``(B, N)``。
        pvt_batch : np.ndarray
            形状 ``(B, K)``。

        Returns
        -------
        metrics_batch : np.ndarray
            形状 ``(B, M)``。
        feasibility_batch : np.ndarray
            形状 ``(B,)``。
        """
        metrics_list = []
        feas_list = []
        for p, v in zip(params_batch, pvt_batch):
            m, f = self.simulate(p, v)
            metrics_list.append(m)
            feas_list.append(f)
        return np.stack(metrics_list), np.array(feas_list)


class MockSpiceInterface(SpiceInterface):
    """Mock SPICE 仿真器，使用解析函数模拟简单运放。

    输出指标 (M=3):
        0 - Gain (dB)
        1 - BW (Hz)
        2 - Power (W)

    参数假设 (N=6): W1, L1, W2, L2, W3, L3
    PVT 假设 (K=3): process, voltage, temperature
    """

    def simulate(
        self,
        params: np.ndarray,
        pvt: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        W1, L1, W2, L2, W3, L3 = params
        process, voltage, temperature = pvt

        # ---- 解析近似 ----
        # gm ~ sqrt(2 * mu * Cox * W/L * Id)
        # 简化: Gain ~ 20*log10(k * sqrt(W1/L1) * sqrt(W2/L2))
        k_gain = 50.0
        gm1 = np.sqrt(max(W1 / L1, 1e-3))
        gm2 = np.sqrt(max(W2 / L2, 1e-3))
        gain_linear = k_gain * gm1 * gm2 * (1.0 + 0.1 * process)
        gain_db = 20.0 * np.log10(max(gain_linear, 1e-6))

        # BW ~ gm / (2*pi*C_load), C_load ~ W3*L3
        c_load = W3 * L3 * 1e6 + 1e-12  # 归一化 + 下限
        gm_total = np.sqrt(max(W1 / L1, 1e-3)) * 1e-3
        bw = gm_total / (2.0 * np.pi * c_load) * voltage * 1e6
        bw *= (1.0 - 0.002 * (temperature - 27.0))  # 温度效应

        # Power ~ V * I_bias, I_bias ~ k * (W3/L3)
        i_bias = 1e-4 * (W3 / L3) * 1e-6
        power = voltage * i_bias * (1.0 + 0.001 * (temperature - 27.0))

        metrics = np.array([gain_db, bw, power], dtype=np.float64)

        # ---- 可行性判定 ----
        # 简单判定：增益 > 0 且 功耗合理 => 认为电路收敛
        feasibility = 1.0
        if gain_db < 0:
            feasibility *= 0.1
        if power > 0.01:
            feasibility *= 0.5
        if power < 0:
            feasibility = 0.0

        feasibility = float(np.clip(feasibility, 0.0, 1.0))
        return metrics, feasibility

