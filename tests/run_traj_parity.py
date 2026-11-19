"""T1.3 录制轨迹端到端对拍：部署端滑窗逻辑 + 导出模型 vs Python 训练链。

实施方案 outputs/Mamba2实施方案.md 阶段 1/T1.3：
  逐帧输入 50 维原始观测序列，按部署契约（docs/deploy_contract_mamba.md §1-2）
  维护 20 帧滑窗（新帧在末尾、启动/复位补零），每步分别过：
    - 部署链：导出的 policy_jit.pt（归一化已烘焙） → action_deploy
    - Python 链：actor(norm(flat_window))（训练侧 actor） → action_python
  逐位/近逐位比对并报告 max|diff| 量级。

用法（n1 conda 环境）：
  # 自检（随机初始化权重 + 合成观测流，验证对拍管线本身）：
  python tests/run_traj_parity.py
  # 真实录制轨迹（npz 键：obs_raw (T,50) float32；actions (T,13) 可选）：
  python tests/run_traj_parity.py --traj traj.npz --jit policy_jit.pt --ckpt model.pt

真实轨迹录制方法（训练侧）：play.py 加载 checkpoint 跑仿真，每控制步把
compute_observation_profile 的 50 维单帧 obs 与下发动作写入 npz；
重启/复位时在轨迹里插入 reset 标记（或分段存多条），对拍脚本按段补零重建窗。
判据：同一初始状态、同一观测序列下动作差应 ≤ 漂移预算（fp32 自检 ~1e-6；
AMP bf16 训练 checkpoint → fp32 导出的漂移属预期，量级记录即可）。

严肃性约定（T1.3 修订）：
  - 给 --ckpt 时：actor 权重必须 0 missing / 0 unexpected，且必须能读到
    `obs_normalizer_state_dict`，否则直接 SystemExit —— 不做「静默降级」；
  - 不给 --ckpt 时：权重与归一化统计都是随机的，两条链路共用同一份，
    diff 天然接近 0，输出会带 [SELF-TEST] 标记，**不计入验收**。
"""
import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(ROOT, "FastTD3"))

SEQ_LEN = 20
OBS_DIM = 50
N_ACT = 13
FLAT = SEQ_LEN * OBS_DIM
TOL_DRIFT = 5e-4


def rebuild_windows(obs_raw: torch.Tensor, resets=None):
    """部署侧滑窗维护（契约 §1-2）：新帧在末尾、展平 [最老…最新]、复位补零。

    Args:
        obs_raw: (T, 50) 逐帧原始观测
        resets: (T,) bool，True 表示该步为 episode 首步（此前窗清零）；None 视为单段
    Returns:
        (T, 1000) 每步的展平滑窗
    """
    T = obs_raw.shape[0]
    win = torch.zeros(SEQ_LEN, OBS_DIM, dtype=obs_raw.dtype)
    out = torch.zeros(T, FLAT, dtype=obs_raw.dtype)
    for t in range(T):
        if resets is not None and bool(resets[t]):
            win.zero_()                      # 复位清窗（补零，不留旧值）
        win = torch.roll(win, -1, dims=0)    # 丢最老帧
        win[-1] = obs_raw[t]                 # 新帧追加末尾
        out[t] = win.reshape(-1)
    return out


def read_ckpt(path):
    """读一次 checkpoint，返回 (actor_state_dict, obs_normalizer_state_dict|None)。

    字段名与 OffPolicyRunner.save 一致：`actor_state_dict` / `obs_normalizer_state_dict`。
    """
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise SystemExit(f"{path}: 期望 dict，实得 {type(state).__name__}")
    actor_sd = state.get("actor_state_dict", state.get("model_state_dict"))
    if actor_sd is None:
        actor_sd = state            # 直接存了 actor 的 state_dict
    # 兼容 qnet/actor 共用命名前缀的情况
    actor_sd = {k.split("actor.", 1)[-1] if k.startswith("actor.") else k: v
                for k, v in actor_sd.items()}
    return actor_sd, state.get("obs_normalizer_state_dict")


def load_actor(ckpt_path, device, actor_sd=None):
    """从 FastTD3 checkpoint 恢复 MambaActor；无 checkpoint 用随机初始化。

    有权重时按「错配即失败」处理：权重名对不上必须报错，否则会退化成
    「两条链路共用同一份权重、diff 天然为 0」的假通过。
    """
    from fast_td3.fast_td3 import MambaActor
    actor = MambaActor(
        n_obs=FLAT, n_act=N_ACT, num_envs=4, init_scale=0.01,
        hidden_dim=1024, num_stack=SEQ_LEN, obs_dim=OBS_DIM,
        mamba_d_model=128, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_headdim=32, mamba_n_layers=2,
        device=device,
    ).to(device)
    if ckpt_path:
        if actor_sd is None:
            actor_sd, _ = read_ckpt(ckpt_path)
        missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
        if missing or unexpected:
            raise SystemExit(
                "checkpoint 权重与 MambaActor 结构不匹配，拒绝继续对拍：\n"
                f"  missing({len(missing)})       = {list(missing)[:5]}\n"
                f"  unexpected({len(unexpected)}) = {list(unexpected)[:5]}"
            )
        print(f"loaded actor from {ckpt_path} (missing=0, unexpected=0)")
    else:
        print("no --ckpt given: using random-init actor（仅 pipeline 自检，不计入验收）")
    actor.eval()
    return actor


def build_norm(ckpt_path, device, norm_sd=None):
    """归一化：优先用 checkpoint 的训练侧统计；有 ckpt 但缺统计即报错。"""
    from fast_td3.fast_td3_utils import EmpiricalNormalization
    norm = EmpiricalNormalization(shape=FLAT, device=device)
    if ckpt_path:
        if norm_sd is None:
            _, norm_sd = read_ckpt(ckpt_path)
        if norm_sd is None:
            raise SystemExit(
                f"{ckpt_path} 内没有 obs_normalizer_state_dict，无法复现训练侧归一化；"
                "请使用 OffPolicyRunner.save 产出的 checkpoint。"
            )
        norm.load_state_dict(norm_sd)
        print("loaded obs_normalizer from checkpoint")
    else:
        norm.train()
        with torch.no_grad():
            norm(torch.randn(64, FLAT, device=device))
        print("WARNING: 无 ckpt，归一化用随机统计（仅 pipeline 自检，不计入验收）")
    norm.eval()
    return norm


def build_export(actor, norm, device):
    """走真实导出代码生成 policy_jit.pt，返回 traced 模型（部署端等价物）。"""
    from fast_td3.fast_td3_utils import EmpiricalNormalization
    from rsl_rl.runners.off_policy_runner import OffPolicyRunner

    stub = SimpleNamespace(actor=actor, obs_normalizer=norm)
    fd, path = tempfile.mkstemp(suffix="_policy_jit.pt")
    os.close(fd)
    OffPolicyRunner._save_jit_mamba(stub, path)
    traced = torch.jit.load(path).to(device).eval()
    os.remove(path)
    return traced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=None, help="录制轨迹 npz（键 obs_raw: (T,50)）")
    ap.add_argument("--jit", default=None, help="已导出 policy_jit.pt（缺省则现场导出）")
    ap.add_argument("--ckpt", default=None, help="FastTD3 checkpoint（缺省随机权重）")
    ap.add_argument("--steps", type=int, default=200, help="合成轨迹长度（自检模式）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit(
            "本脚本需要 CUDA：Python 参考链 actor() 走 mamba_ssm 的 Triton 融合路径"
            "（causal_conv1d_cuda），在 CPU 上会报 Expected x.is_cuda()。"
        )
    torch.manual_seed(3)

    # --- 观测序列 ---
    resets = None
    if args.traj:
        data = torch.load(args.traj) if args.traj.endswith(".pt") else None
        if data is None:
            import numpy as np
            z = np.load(args.traj)
            obs_raw = torch.from_numpy(z["obs_raw"]).float()
            resets = (torch.from_numpy(z["resets"]).bool()
                      if "resets" in z.files else None)
        else:
            obs_raw = data["obs_raw"].float()
            resets = data.get("resets")
        print(f"traj: {args.traj}, T={obs_raw.shape[0]}")
    else:
        obs_raw = torch.randn(args.steps, OBS_DIM) * 0.5
        print(f"synthetic traj: T={args.steps} (self-test)")

    windows = rebuild_windows(obs_raw, resets)     # (T, 1000) 部署侧窗重建

    # --- 两条链路 ---
    actor_sd, norm_sd = read_ckpt(args.ckpt) if args.ckpt else (None, None)
    actor = load_actor(args.ckpt, device, actor_sd)
    norm = build_norm(args.ckpt, device, norm_sd)

    if args.jit:
        traced = torch.jit.load(args.jit).to(device).eval()
        print(f"loaded traced: {args.jit}")
    else:
        traced = build_export(actor, norm, device)
        print("exported policy_jit.pt from current actor (self-test)")

    diffs = []
    with torch.no_grad():
        for t in range(windows.shape[0]):
            w = windows[t:t + 1].to(device)
            a_py = actor(norm(w))            # Python 训练链（含归一化）
            a_dp = traced(w)                 # 部署链（归一化已烘焙）
            diffs.append((a_py - a_dp).abs().max().item())
    diffs = torch.tensor(diffs)
    print(f"steps={diffs.numel()}  max|diff|={diffs.max().item():.3e}  "
          f"mean|diff|={diffs.mean().item():.3e}  (drift budget {TOL_DRIFT:.0e})")
    ok = diffs.max().item() < TOL_DRIFT
    tag = "" if args.ckpt else "  [SELF-TEST: 无 checkpoint，只验证导出管线，不计入验收]"
    print("RESULT:", "PASS" if ok else "FAIL", tag)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
