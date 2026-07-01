import torch
import torch.nn as nn

from mamba_ssm import Mamba2


class Mamba2Encoder(nn.Module):
    """Mamba-2 编码器，用于提取观测序列的时序特征"""

    def __init__(self,
                 d_input: int,
                 d_model: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 n_layers: int = 2,
                 **kwargs):
        """
        Args:
            d_input: 输入维度（观测维度）
            d_model: Mamba 隐藏维度
            d_state: SSM 状态维度
            d_conv: 卷积宽度
            expand: 扩展因子
            headdim: 头维度
            n_layers: Mamba 层数
        """
        super(Mamba2Encoder, self).__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers

        # 输入投影层：将观测维度映射到 Mamba 隐藏维度
        self.input_proj = nn.Linear(d_input, d_model)

        # Mamba-2 层
        self.mamba_layers = nn.ModuleList([
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
            for _ in range(n_layers)
        ])

        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_input)

        Returns:
            编码后的特征，形状为 (batch_size, d_model)
        """
        # 投影到隐藏维度
        h = self.input_proj(x)

        # 通过 Mamba 层
        for i in range(self.n_layers):
            residual = h
            h = self.mamba_layers[i](h)
            h = self.norms[i](h + residual)

        # 取最后一个时间步的输出作为序列特征
        h = h[:, -1, :]

        return h
