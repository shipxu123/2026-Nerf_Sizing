"""Fourier 位置编码模块 (可选, 提升高频拟合能力)"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fourier 位置编码。

    将 D 维输入映射到 ``D * (2L + 1)`` 维，其中 L 为频率数。

    编码公式::

        gamma(x) = [x, sin(2^0 * pi * x), cos(2^0 * pi * x),
                       sin(2^1 * pi * x), cos(2^1 * pi * x),
                       ...,
                       sin(2^{L-1} * pi * x), cos(2^{L-1} * pi * x)]

    Parameters
    ----------
    input_dim : int
        输入维度。
    num_freqs : int
        频率数量 L。
    """

    def __init__(self, input_dim: int, num_freqs: int = 6) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.output_dim = input_dim * (2 * num_freqs + 1)

        # 预计算频率 2^0, 2^1, ..., 2^{L-1}
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs)  # (L,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入，形状 ``(..., D)``。

        Returns
        -------
        torch.Tensor
            编码后输出，形状 ``(..., D * (2L + 1))``。
        """
        # x: (..., D)
        encoded = [x]
        for freq in self.freqs:
            encoded.append(torch.sin(freq * torch.pi * x))
            encoded.append(torch.cos(freq * torch.pi * x))
        return torch.cat(encoded, dim=-1)

