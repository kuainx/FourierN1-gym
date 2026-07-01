import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .actor_critic_mlp import ActorCriticMLP
from .mamba_encoder import Mamba2Encoder
from .mlp import MLP


class Mamba2LayerJit(nn.Module):
    """纯 PyTorch 实现的 Mamba2 前向传播，用于 JIT 导出（不依赖 Triton 内核）。

    从已训练的 Mamba2 层复制参数，使用逐时间步顺序扫描替代 Triton 块扫描。
    """

    def __init__(self, mamba2_layer):
        super().__init__()
        # 复制子模块和参数
        self.in_proj = mamba2_layer.in_proj
        self.conv1d = mamba2_layer.conv1d
        self.dt_bias = mamba2_layer.dt_bias
        self.A_log = mamba2_layer.A_log
        self.D = mamba2_layer.D
        self.out_proj = mamba2_layer.out_proj

        # 复制配置
        self.d_state = mamba2_layer.d_state
        self.d_conv = mamba2_layer.d_conv
        self.d_inner = mamba2_layer.d_inner
        self.d_ssm = mamba2_layer.d_ssm
        self.nheads = mamba2_layer.nheads
        self.headdim = mamba2_layer.headdim
        self.ngroups = mamba2_layer.ngroups
        self.D_has_hdim = mamba2_layer.D_has_hdim
        self.rmsnorm = mamba2_layer.rmsnorm
        self.norm_before_gate = mamba2_layer.norm_before_gate

        # 复制 RMSNorm 参数（纯 PyTorch 实现）
        if mamba2_layer.rmsnorm:
            self.norm_weight = mamba2_layer.norm.weight
            self.norm_eps = mamba2_layer.norm.eps
            self.norm_group_size = mamba2_layer.norm.group_size

    def _rms_norm_gated(self, x, z):
        """纯 PyTorch 实现的带门控 RMSNorm"""
        dtype = x.dtype
        x = x.float()
        z = z.float()

        if not self.norm_before_gate:
            x = x * F.silu(z)

        gs = self.norm_group_size
        if gs is None or gs == x.shape[-1]:
            rstd = 1.0 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.norm_eps)
            out = x * rstd * self.norm_weight.float()
        else:
            # 分组 RMSNorm
            x_group = x.view(*x.shape[:-1], -1, gs)
            rstd = 1.0 / torch.sqrt(x_group.square().mean(dim=-1, keepdim=True) + self.norm_eps)
            out = (x_group * rstd).reshape(*x.shape) * self.norm_weight.float()

        if self.norm_before_gate:
            out = out * F.silu(z)

        return out.to(dtype)

    def forward(self, u):
        """纯 PyTorch 前向传播，使用逐时间步顺序 SSM 扫描"""
        batch, seqlen, _ = u.shape

        A = -torch.exp(self.A_log.float())  # (nheads,)

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        # 因果 1D 卷积
        xBC = self.conv1d(xBC.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
        xBC = F.silu(xBC)

        x, B, C = torch.split(
            xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        # 重塑
        x = x.view(batch, seqlen, self.nheads, self.headdim)   # (B, L, H, P)
        B = B.view(batch, seqlen, self.ngroups, self.d_state)   # (B, L, G, N)
        C = C.view(batch, seqlen, self.ngroups, self.d_state)   # (B, L, G, N)

        D = self.D.float()
        if self.D_has_hdim:
            D = D.view(self.nheads, self.headdim)
        else:
            D = D.view(self.nheads, 1)

        # 顺序 SSM 扫描（替代 Triton 块扫描）
        state = torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                            device=u.device, dtype=torch.float32)

        outputs = []
        for t in range(seqlen):
            dt_t = F.softplus(dt[:, t].float() + self.dt_bias.float())  # (B, H)
            dA_t = torch.exp(dt_t * A)  # (B, H)
            x_t = x[:, t]  # (B, H, P)

            if self.ngroups == 1:
                B_t = B[:, t, 0]  # (B, N)
                C_t = C[:, t, 0]  # (B, N)
            else:
                B_t = B[:, t].repeat_interleave(self.nheads // self.ngroups, dim=1)
                C_t = C[:, t].repeat_interleave(self.nheads // self.ngroups, dim=1)

            dBx = torch.einsum("bh,bn,bhp->bhpn", dt_t, B_t, x_t)
            state = state * dA_t[:, :, None, None] + dBx
            y_t = torch.einsum("bhpn,bn->bhp", state, C_t)
            y_t = y_t + D * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, H, P)
        y = y.reshape(batch, seqlen, self.d_ssm).to(u.dtype)

        if self.rmsnorm:
            y = self._rms_norm_gated(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)
        return out


class Mamba2EncoderJit(nn.Module):
    """JIT 兼容的 Mamba2Encoder，使用纯 PyTorch Mamba2 层替代 Triton 版本"""

    def __init__(self, encoder):
        super().__init__()
        self.d_input = encoder.d_input
        self.d_model = encoder.d_model
        self.n_layers = encoder.n_layers

        self.input_proj = encoder.input_proj
        self.norms = encoder.norms

        # 将 Mamba2 层替换为 JIT 兼容版本
        self.mamba_layers = nn.ModuleList([
            Mamba2LayerJit(layer) for layer in encoder.mamba_layers
        ])

    def forward(self, x):
        h = self.input_proj(x)
        for i in range(self.n_layers):
            residual = h
            h = self.mamba_layers[i](h)
            h = self.norms[i](h + residual)
        h = h[:, -1, :]
        return h


class ActorCriticMamba(ActorCriticMLP):
    """带 Mamba-2 编码器的 ActorCritic 网络

    接口与 ActorCriticMLP 完全兼容。
    输入为扁平的历史序列 (batch, seq_len * obs_dim)，内部 reshape 为序列形式，
    通过 Mamba 建模时序，取最后时刻输出送入 Actor/Critic 头。
    """

    def __init__(self,
                 actor_num_input,
                 critic_num_input,
                 actor_num_output,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_weights=False,
                 fixed_std=False,
                 init_noise_std=0.2,
                 # Mamba 参数
                 mamba_d_model=128,
                 mamba_d_state=16,
                 mamba_d_conv=4,
                 mamba_expand=2,
                 mamba_headdim=64,
                 mamba_n_layers=2,
                 seq_len=10,
                 **kwargs):
        """
        Args:
            actor_num_input: Actor 输入维度 (seq_len * obs_dim)
            critic_num_input: Critic 输入维度 (seq_len * obs_dim 或 seq_len * critic_obs_dim)
            actor_num_output: Actor 输出维度（动作维度）
            actor_hidden_dims: Actor 隐藏层维度
            critic_hidden_dims: Critic 隐藏层维度
            activation: 激活函数名称
            init_weights: 是否初始化权重
            fixed_std: 是否固定标准差
            init_noise_std: 初始噪声标准差
            mamba_d_model: Mamba 隐藏维度
            mamba_d_state: SSM 状态维度
            mamba_d_conv: 卷积宽度
            mamba_expand: 扩展因子
            mamba_headdim: 头维度
            mamba_n_layers: Mamba 层数
            seq_len: 序列长度
        """
        print("----------------------------------")
        print("ActorCriticMamba")
        print(f"actor_num_input: {actor_num_input}")
        print(f"critic_num_input: {critic_num_input}")
        print(f"actor_num_output: {actor_num_output}")
        print("----------------------------------")


        # 调用父类构造函数（仅用于初始化基本参数，不使用其 MLP）
        super(ActorCriticMamba, self).__init__(
            actor_num_input=actor_num_input,
            critic_num_input=critic_num_input,
            actor_num_output=actor_num_output,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_weights=init_weights,
            fixed_std=fixed_std,
            init_noise_std=init_noise_std,
        )

        # 保存序列长度
        self.seq_len = seq_len

        # 计算单步观测维度
        self.actor_obs_dim = actor_num_input // seq_len

        print(f"Actor obs_dim per step: {self.actor_obs_dim}")
        print(f"Critic input dim: {critic_num_input}")
        print(f"Seq length: {self.seq_len}")

        # 创建 Mamba 编码器（仅 Actor 使用）
        self.actor_mamba = Mamba2Encoder(
            d_input=self.actor_obs_dim,
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            n_layers=mamba_n_layers,
        )

        # Critic 使用简单的线性投影层（只处理单步观测）
        self.critic_proj = nn.Linear(critic_num_input, mamba_d_model)

        # 重新创建 Actor 和 Critic 头，使用 mamba_d_model 作为输入维度
        self.actor = MLP(input_size=mamba_d_model,
                         output_size=actor_num_output,
                         hidden_dims=actor_hidden_dims,
                         activation=activation,
                         init_weights=init_weights)

        self.critic = MLP(input_size=mamba_d_model,
                          output_size=1,
                          hidden_dims=critic_hidden_dims,
                          activation=activation,
                          init_weights=init_weights)

        print(f"\033[94mActor Mamba Encoder: {self.actor_mamba}\033[0m")
        print(f"\033[94mCritic Projection: {self.critic_proj}\033[0m")
        print(f"\033[94mActor MLP Head: {self.actor}\033[0m")
        print(f"\033[94mCritic MLP Head: {self.critic}\033[0m")

    def reset(self, dones=None):
        """重置（Mamba 无状态需要重置，保持接口兼容）

        Args:
            dones (bool): 环境重置标志
        """
        pass

    def _encode_actor_obs(self, observations: torch.Tensor) -> torch.Tensor:
        """编码 Actor 观测序列

        Args:
            observations: 扁平的历史观测，形状 (batch_size, seq_len * actor_obs_dim)

        Returns:
            编码后的特征，形状 (batch_size, mamba_d_model)
        """
        batch_size = observations.shape[0]

        # reshape 为序列形式 (batch_size, seq_len, actor_obs_dim)
        obs_seq = observations.view(batch_size, self.seq_len, self.actor_obs_dim)

        # 通过 Mamba 编码器，取最后时刻输出
        encoded = self.actor_mamba(obs_seq)

        return encoded

    def _encode_critic_obs(self, observations: torch.Tensor) -> torch.Tensor:
        """编码 Critic 观测（单步）

        Args:
            observations: 单步观测，形状 (batch_size, critic_num_input)

        Returns:
            编码后的特征，形状 (batch_size, mamba_d_model)
        """
        # 直接通过线性投影层
        encoded = self.critic_proj(observations)

        return encoded

    def update_distribution(self, observations):
        """更新动作分布

        Args:
            observations: Actor 观测，形状 (batch_size, seq_len * actor_obs_dim)
        """
        # 通过 Mamba 编码
        encoded_obs = self._encode_actor_obs(observations)

        # 使用编码后的特征更新分布
        mean = self.actor(encoded_obs)
        std = self.std.to(mean.device)

        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, **kwargs):
        """生成动作

        Args:
            observations: Actor 观测，形状 (batch_size, seq_len * actor_obs_dim)

        Returns:
            采样的动作
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        """推理模式生成动作

        Args:
            observations: Actor 观测，形状 (batch_size, seq_len * actor_obs_dim)

        Returns:
            动作均值
        """
        # 通过 Mamba 编码
        encoded_obs = self._encode_actor_obs(observations)

        # 使用编码后的特征
        actions_mean = self.actor(encoded_obs)
        return actions_mean

    def evaluate(self, critic_observations=None, **kwargs):
        """评估状态价值

        Args:
            critic_observations: Critic 观测，形状 (batch_size, critic_num_input)

        Returns:
            状态价值
        """
        # 通过 Mamba 编码
        encoded_obs = self._encode_critic_obs(critic_observations)

        # 使用编码后的特征
        value = self.critic(encoded_obs)
        return value

    def save_jit(self, path: str):
        """导出可部署的 TorchScript 模型

        使用纯 PyTorch 实现的 Mamba2 层替代 Triton 版本，
        避免 torch.jit.trace 无法跟踪 Triton 内核的问题。

        Args:
            path: 保存路径
        """
        # 将 Mamba 编码器转换为纯 PyTorch 版本（不使用 Triton 内核）
        jit_encoder = Mamba2EncoderJit(self.actor_mamba)
        jit_encoder.eval()

        # 创建 Actor 推理 wrapper
        actor_wrapper = ActorJitWrapper(
            actor_mamba=jit_encoder,
            actor_mlp=self.actor,
            seq_len=self.seq_len,
            actor_obs_dim=self.actor_obs_dim,
        )
        actor_wrapper.eval()

        # 深拷贝后移到 CPU trace，避免影响原始模型的 CUDA 参数
        actor_wrapper = copy.deepcopy(actor_wrapper).cpu()

        dummy_input = torch.zeros(1, self.seq_len * self.actor_obs_dim)
        traced_actor = torch.jit.trace(actor_wrapper, dummy_input)

        # 保存
        traced_actor.save(path)
        print(f"Saved JIT model to {path}")


class ActorJitWrapper(nn.Module):
    """用于 JIT 导出的 Actor 包装器"""

    def __init__(self,
                 actor_mamba,
                 actor_mlp: MLP,
                 seq_len: int,
                 actor_obs_dim: int):
        super(ActorJitWrapper, self).__init__()
        self.actor_mamba = actor_mamba
        self.actor_mlp = actor_mlp
        self.seq_len = seq_len
        self.actor_obs_dim = actor_obs_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: 扁平的历史观测，形状 (batch_size, seq_len * actor_obs_dim)

        Returns:
            动作均值，形状 (batch_size, actor_num_output)
        """
        batch_size = observations.shape[0]

        # reshape 为序列形式
        obs_seq = observations.view(batch_size, self.seq_len, self.actor_obs_dim)

        # 通过 Mamba 编码器
        encoded = self.actor_mamba(obs_seq)

        # 通过 MLP 头
        actions_mean = self.actor_mlp(encoded)

        return actions_mean
