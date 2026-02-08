"""CircuitFieldNet: 核心可微分代理模型 (MLP + 双输出头)"""

from __future__ import annotations

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class CircuitFieldNet(nn.Module):
    """电路场网络 —— 可微分代理模型。

    共享主干 (Shared Backbone) + 双输出头 (Dual Head) 结构:
      - Feasibility Head: sigma in [0, 1]
      - Performance Head: y in R^M

    Parameters
    ----------
    input_dim_p : int
        设计参数维度 N。
    input_dim_v : int
        PVT 条件维度 K。
    hidden_dims : list[int]
        隐藏层维度列表。
    output_dim_y : int
        性能指标维度 M。
    use_positional_encoding : bool
        是否启用 Fourier 位置编码。
    encoding_freqs : int
        位置编码频率数 (仅在启用时有效)。
    """

    def __init__(
        self,
        input_dim_p: int = 6,
        input_dim_v: int = 3,
        hidden_dims: list[int] | None = None,
        output_dim_y: int = 3,
        use_positional_encoding: bool = False,
        encoding_freqs: int = 6,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.use_positional_encoding = use_positional_encoding

        # 位置编码
        if use_positional_encoding:
            self.pe_p = PositionalEncoding(input_dim_p, encoding_freqs)
            self.pe_v = PositionalEncoding(input_dim_v, encoding_freqs)
            actual_input_dim = self.pe_p.output_dim + self.pe_v.output_dim
        else:
            self.pe_p = None
            self.pe_v = None
            actual_input_dim = input_dim_p + input_dim_v

        # 共享主干 MLP
        layers: list[nn.Module] = []
        prev_dim = actual_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Feasibility Head (sigma): Sigmoid 输出 [0, 1]
        self.sigma_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid(),
        )

        # Performance Head (y): 线性输出
        self.perf_head = nn.Linear(prev_dim, output_dim_y)

        # 权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 均匀初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Parameters
        ----------
        p : torch.Tensor
            设计参数，形状 ``(B, N)``。
        v : torch.Tensor
            PVT 条件，形状 ``(B, K)``。

        Returns
        -------
        sigma_hat : torch.Tensor
            可行性预测，形状 ``(B, 1)``。
        y_hat : torch.Tensor
            性能预测，形状 ``(B, M)``。
        """
        if self.use_positional_encoding:
            p = self.pe_p(p)
            v = self.pe_v(v)

        x = torch.cat([p, v], dim=-1)  # (B, input_dim)
        features = self.backbone(x)     # (B, hidden_dims[-1])

        sigma_hat = self.sigma_head(features)  # (B, 1)
        y_hat = self.perf_head(features)       # (B, M)

        return sigma_hat, y_hat

    @classmethod
    def from_config(cls, cfg: dict) -> "CircuitFieldNet":
        """从配置字典创建模型。

        Parameters
        ----------
        cfg : dict
            ``model`` 部分的配置字典。
        """
        return cls(
            input_dim_p=cfg["input_dim_p"],
            input_dim_v=cfg["input_dim_v"],
            hidden_dims=cfg.get("hidden_dims", [256, 256]),
            output_dim_y=cfg["output_dim_y"],
            use_positional_encoding=cfg.get("use_positional_encoding", False),
            encoding_freqs=cfg.get("encoding_freqs", 6),
        )

