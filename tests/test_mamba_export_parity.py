"""T1.2 自动化数值对拍单测：训练前向 vs 导出复现（Mamba2 链路）。

实施方案 outputs/Mamba2实施方案.md 阶段 1/T1.2 三用例（含 case c batch 回归）：

  a1. Mamba2EncoderJit（导出复现） vs 官方 Mamba2.step 逐帧递推
      —— 导出物正确性主判据，严格 1e-5（实测 ~7e-7，复现审查报告 §4 实验 1）
  a2. Mamba2Encoder（Triton 训练前向） vs Mamba2EncoderJit
      —— 并行扫描 vs 顺序扫描的 fp32 累序漂移，预算 5e-4
      （实测层 1.3e-4 / 编码器 5e-5；计划文档判据 1e-5 系参照
        Jit-vs-step 基线 4.77e-07 所定，对本对比过严，见附录 C 修订记录）
  b1. 端到端：eager 导出包装（含归一化烘焙） vs traced —— trace/烘焙保真，严格 1e-5
  b2. 端到端：actor(norm_obs)（Triton 训练链） vs traced —— 训练→部署总漂移，预算 5e-4
      （两条链路均走真实导出代码：PPO save_jit / FastTD3 _save_jit_mamba）
  c.  全部用例均以 batch=1 与 batch=8 各跑一遍（batch 维不固化）

用法（n1 conda 环境）：
    /home/mao/.conda/envs/n1/bin/python tests/test_mamba_export_parity.py
后续阶段 4 的任何前向改动，必须先过本单测（合入门槛，方案 §7 前置）。
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(ROOT, "FastTD3"))

from rsl_rl.modules.actor_critic_mamba import (
    ActorCriticMamba,
    Mamba2EncoderJit,
    Mamba2LayerJit,
)
from rsl_rl.modules.mamba_encoder import Mamba2Encoder

TOL_STRICT = 1e-5    # 导出复现/trace 烘焙保真
TOL_DRIFT = 5e-4     # Triton 并行扫描 vs 顺序扫描的 fp32 漂移预算
SEQ_LEN = 20
OBS_DIM = 50
N_ACT = 13
FLAT = SEQ_LEN * OBS_DIM

device = "cuda" if torch.cuda.is_available() else "cpu"
results = []


def record(name, diff, tol):
    ok = diff < tol
    results.append((name, diff, ok))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: max|diff|={diff:.3e} (tol {tol:.0e})")
    return ok


def build_encoder():
    enc = Mamba2Encoder(
        d_input=OBS_DIM, d_model=128, d_state=16, d_conv=4,
        expand=2, headdim=32, n_layers=2,
    ).to(device)
    enc.eval()
    return enc


def step_reference(layer, u):
    """官方 Mamba2.step 逐帧递推（审查报告 §4 实验 1 同口径）。"""
    batch, seqlen, _ = u.shape
    conv_state, ssm_state = layer.allocate_inference_cache(
        batch, seqlen, dtype=u.dtype)
    outs = []
    for t in range(seqlen):
        y, conv_state, ssm_state = layer.step(u[:, t:t + 1], conv_state, ssm_state)
        outs.append(y.squeeze(1))
    return torch.stack(outs, dim=1)


def case_a():
    """编码器级：a1 Jit vs 官方 step（严格）；a2 Triton vs Jit（漂移预算）。"""
    torch.manual_seed(0)
    enc = build_encoder()
    enc_jit = Mamba2EncoderJit(enc).to(device).eval()
    layer, layer_jit = enc.mamba_layers[0], None
    layer_jit = Mamba2LayerJit(layer)

    for B in (1, 8):
        x = torch.randn(B, SEQ_LEN, OBS_DIM, device=device)
        h = torch.randn(B, SEQ_LEN, 128, device=device)
        with torch.no_grad():
            y_ref = enc(x)
            y_jit = enc_jit(x)
            z_step = step_reference(layer, h)
            z_jit = layer_jit(h)
        record(f"a1 Jit vs official step (B={B})",
               (z_jit - z_step).abs().max().item(), TOL_STRICT)
        record(f"a2 Triton vs Jit encoder (B={B})",
               (y_ref - y_jit).abs().max().item(), TOL_DRIFT)


def case_b_ppo():
    """PPO 链路端到端：act_inference vs save_jit 的 traced。"""
    torch.manual_seed(1)
    ac = ActorCriticMamba(
        actor_num_input=FLAT, critic_num_input=183, actor_num_output=N_ACT,
        actor_hidden_dims=[256, 256, 128, 64],  # T0.2 统一头容量
        seq_len=SEQ_LEN, num_stack=SEQ_LEN,
        mamba_d_model=128, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_headdim=32, mamba_n_layers=2,
    ).to(device)
    ac.eval()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "policy_jit.pt")
        ac.save_jit(path)                      # 真实导出代码（deepcopy→CPU→trace）
        traced = torch.jit.load(path).to(device).eval()

        # eager 导出包装（Jit 编码器 + 同一 MLP 头），隔离 trace 烘焙误差
        eager_jit = ActorCriticMamba(
            actor_num_input=FLAT, critic_num_input=183, actor_num_output=N_ACT,
            actor_hidden_dims=[256, 256, 128, 64],
            seq_len=SEQ_LEN, num_stack=SEQ_LEN,
            mamba_d_model=128, mamba_d_state=16, mamba_d_conv=4,
            mamba_expand=2, mamba_headdim=32, mamba_n_layers=2,
        ).to(device)
        eager_jit.load_state_dict(ac.state_dict())
        eager_jit.actor_mamba = Mamba2EncoderJit(ac.actor_mamba).to(device)
        eager_jit.eval()

        def eager_ref(obs):
            return ac.act_inference(obs)

        def eager_export(obs):
            return eager_jit.act_inference(obs)

        # 手工展开 _compare 的 b2/b1（此处直接循环，避免字符串拼接混乱）
        for B in (1, 8):
            obs = torch.randn(B, FLAT, device=device)
            with torch.no_grad():
                a_ref = eager_ref(obs)
                a_eager = eager_export(obs)
                a_jit = traced(obs)
            record(f"b2 PPO actor vs traced (B={B})",
                   (a_ref - a_jit).abs().max().item(), TOL_DRIFT)
            record(f"b1 PPO eager-export vs traced (B={B})",
                   (a_eager - a_jit).abs().max().item(), TOL_STRICT)


def case_b_fasttd3():
    """FastTD3 链路端到端（含归一化烘焙）：actor(norm_obs) vs traced。"""
    from fast_td3.fast_td3 import MambaActor
    from fast_td3.fast_td3_utils import EmpiricalNormalization
    from rsl_rl.runners.off_policy_runner import OffPolicyRunner

    torch.manual_seed(2)
    actor = MambaActor(
        n_obs=FLAT, n_act=N_ACT, num_envs=8, init_scale=0.01,
        hidden_dim=1024, num_stack=SEQ_LEN, obs_dim=OBS_DIM,
        mamba_d_model=128, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_headdim=32, mamba_n_layers=2,
        device=device,
    ).to(device)
    actor.eval()

    # 归一化器：先以 train 模式积累统计，再冻结（与导出烘焙语义一致）
    norm = EmpiricalNormalization(shape=FLAT, device=device)
    norm.train()
    with torch.no_grad():
        for _ in range(20):
            norm(torch.randn(64, FLAT, device=device))
    norm.eval()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "policy_jit.pt")
        # 真实导出代码：OffPolicyRunner._save_jit_mamba 仅依赖 actor/obs_normalizer
        stub = SimpleNamespace(actor=actor, obs_normalizer=norm)
        OffPolicyRunner._save_jit_mamba(stub, path)
        traced = torch.jit.load(path).to(device).eval()

        for B in (1, 8):
            obs = torch.randn(B, FLAT, device=device)
            with torch.no_grad():
                a_ref = actor(norm(obs))       # 训练链（Triton）
                a_jit = traced(obs)            # 导出链（Jit，归一化已烘焙）
            record(f"b2 FastTD3 actor(norm_obs) vs traced (B={B})",
                   (a_ref - a_jit).abs().max().item(), TOL_DRIFT)


if __name__ == "__main__":
    print(f"device={device}, tol_strict={TOL_STRICT}, tol_drift={TOL_DRIFT}")
    case_a()          # 含 case c（B=1/8）
    case_b_ppo()      # 含 case c（B=1/8）
    case_b_fasttd3()  # 含 case c（B=1/8）

    n_fail = sum(1 for _, _, ok in results if not ok)
    print(f"\n=== {len(results) - n_fail} passed, {n_fail} failed ===")
    sys.exit(1 if n_fail else 0)
