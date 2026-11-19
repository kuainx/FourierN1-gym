"""P0-3 复核脚本：`nheads % 8` 防呆是否有依据（headdim=64 / nheads=4 是否真的不可用）。

用途：在有 GPU 的机器上用 n1 环境跑一遍，给 P0-3 一个确定结论。
    /home/mao/.conda/envs/n1/bin/python outputs/probe_nheads_headdim.py

三部分：
  1. mamba_ssm 原生 Mamba2 对 headdim=32/64/128 的接受度（模块级约束）
  2. 项目 Mamba2Encoder 防呆对 d_model × headdim 的拦截矩阵
  3. 【需要 GPU】绕过防呆构造 nheads=4 的编码器，跑 Triton 并行扫描前向，
     并与纯 PyTorch Jit 版逐元素对比。2026-11-19 实测：headdim=64 前向直接报
     causal_conv1d channel-last stride 错误（详见文末结论），故 T0.3-d 的约束成立、
     但其归因（「Triton 对齐」）与判据形式（nheads%8）需按 D%8 修订。
"""
import itertools
import sys

import torch

sys.path.insert(0, "/home/mao/N1/rsl_rl")
sys.path.insert(0, "/home/mao/N1/FastTD3")

from mamba_ssm import Mamba2

print("== 0) 运行环境 ==")
print("  python %s / torch %s / cuda_built=%s"
      % (sys.version.split()[0], torch.__version__, torch.version.cuda))
print("  torch.cuda.is_available() = %s, device_count = %d"
      % (torch.cuda.is_available(), torch.cuda.device_count()))
if torch.cuda.is_available():
    print("  device0 = %s" % torch.cuda.get_device_name(0))

print("== 1) mamba_ssm 原生 Mamba2（d_model=128, expand=2）==")
for hd in (32, 64, 128):
    try:
        m = Mamba2(d_model=128, d_state=16, d_conv=4, expand=2, headdim=hd, ngroups=1)
        print("  headdim=%3d -> OK   nheads=%d d_ssm=%d ngroups=%d"
              % (hd, m.nheads, m.d_ssm, m.ngroups))
    except Exception as e:
        print("  headdim=%3d -> FAIL %s: %s" % (hd, type(e).__name__, str(e)[:120]))

print("== 2) 项目防呆拦截矩阵（Mamba2Encoder.__init__）==")
from rsl_rl.modules.mamba_encoder import Mamba2Encoder

ok = []
for dm, hd in itertools.product((32, 48, 64, 96, 128, 160, 192, 256), (32, 64)):
    nheads = dm * 2 // hd
    try:
        Mamba2Encoder(d_input=50, d_model=dm, d_state=16, d_conv=4,
                      expand=2, headdim=hd, n_layers=1)
        state = "PASS"
        ok.append((dm, hd))
    except ValueError:
        state = "BLOCKED"
    print("  d_model=%3d headdim=%2d -> nheads=%2d  %s" % (dm, hd, nheads, state))
print("  ==> 可用组合 %d/%d: %s" % (len(ok), 16, ok))

print("== 3) GPU 前向复核（nheads=4, headdim=64）==")
if not torch.cuda.is_available():
    print("  SKIP: torch.cuda.is_available()=False（当前环境无可用 GPU 设备）")
    sys.exit(0)

from rsl_rl.modules.actor_critic_mamba import Mamba2EncoderJit


def _make_unguarded_encoder(d_input, d_model, d_state, d_conv, expand, headdim, n_layers):
    """按 Mamba2Encoder.__init__ 的结构搭一个实例，但跳过防呆校验。

    用于回答“若没有这条防呆，Triton 到底能不能跑”——直接改校验逻辑会污染被测对象，
    这里用 __new__ + 逐字段赋值复制结构，语义与正式实现一致。
    """
    enc = Mamba2Encoder.__new__(Mamba2Encoder)
    torch.nn.Module.__init__(enc)
    enc.d_input = d_input
    enc.d_model = d_model
    enc.n_layers = n_layers
    enc.input_proj = torch.nn.Linear(d_input, d_model)
    enc.mamba_layers = torch.nn.ModuleList([
        Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv,
               expand=expand, headdim=headdim)
        for _ in range(n_layers)
    ])
    enc.norms = torch.nn.ModuleList([
        torch.nn.LayerNorm(d_model) for _ in range(n_layers)
    ])
    return enc


dev = "cuda"
for hd in (64, 32):
    enc = _make_unguarded_encoder(d_input=50, d_model=128, d_state=16, d_conv=4,
                                  expand=2, headdim=hd, n_layers=2).to(dev).eval()
    jit = Mamba2EncoderJit(enc).to(dev).eval()
    x = torch.randn(8, 20, 50, device=dev)
    try:
        with torch.no_grad():
            y_tri = enc(x)     # Triton 并行扫描（训练链）
            y_jit = jit(x)     # 纯 PyTorch 顺序扫描（导出链）
        print("  headdim=%2d nheads=%d  forward OK  Triton-vs-Jit max|diff|=%.3e"
              % (hd, enc.mamba_layers[0].nheads, (y_tri - y_jit).abs().max().item()))
    except Exception as e:
        print("  headdim=%2d nheads=%d  forward FAIL  %s: %s"
              % (hd, enc.mamba_layers[0].nheads, type(e).__name__, str(e)[:300]))

print("  结论（2026-11-19 已实测）：headdim=64/d_model=128 报 causal_conv1d channel-last"
      " stride 错误——约束真实存在，但原因是 conv 输入视图的通道 stride = in_proj 输出维 "
      "D=2*d_model*expand+2*d_state+nheads 须为 8 的倍数，与 Triton/头数本身无关。"
      "默认 expand=2/d_state=16 下 D%8==0 等价于 nheads%8==0（16 组组合预测/实测 16/16 吻合）。")
