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

        # --- 防呆校验（T0.3-d，A1 修订：按真实约束判断，不再用 nheads 代理）---
        # 约束来源是 causal_conv1d 的 channel-last 布局，不是 Triton：
        # mamba_ssm 融合路径（mamba_split_conv1d_scan_combined）里，conv 的输入是
        # zxbcdt 的切片视图（ssd_combined.py:964-965），转置后的通道 stride 恰为
        # in_proj 的输出维
        #     D = 2 * d_model * expand + 2 * ngroups * d_state + nheads   （本类 ngroups=1）
        # causal_conv1d 要求 x.stride(0) 与 x.stride(2) 均为 8 的倍数，落到 D 上即 D % 8 == 0
        # （xBC.stride(2)=D；stride(0)=seq_len*D 随之满足）。
        # 实测 headdim=64 / d_model=128（nheads=4, D=548）报：
        #   RuntimeError: causal_conv1d with channel last layout requires strides
        #   (x.stride(0) and x.stride(2)) to be multiples of 8
        # 在默认 expand=2、d_state=16 下 D%8==0 与 nheads%8==0 等价（16 组 d_model×headdim
        # 组合实测与预测 16/16 吻合）；直接判 D 可在 d_state 非 4 倍数时不误杀合法配置。
        # 保持 nheads=8 时对应 headdim = d_model * expand // 8。
        d_ssm = d_model * expand
        if d_ssm % headdim != 0:
            raise ValueError(
                f"d_model*expand ({d_ssm}) must be divisible by headdim ({headdim})."
            )
        nheads = d_ssm // headdim
        d_proj = 2 * d_ssm + 2 * d_state + nheads
        if d_proj % 8 != 0:
            raise ValueError(
                f"causal_conv1d channel-last stride constraint violated: in_proj output dim "
                f"D = 2*d_model*expand + 2*d_state + nheads = {d_proj} is not a multiple of 8 "
                f"(d_model={d_model}, expand={expand}, d_state={d_state}, headdim={headdim}, "
                f"nheads={nheads}). With the defaults (expand=2, d_state=16) this is "
                f"equivalent to nheads % 8 == 0. Keep nheads=8 by setting "
                f"mamba_headdim = d_model*expand//8 = {d_ssm // 8} "
                f"(e.g. d_model=128 -> 32, 96 -> 24, 192 -> 48), or change d_state."
            )

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
