"""T1.4 全窗推理单步时延评估（部署算力预算对照）。

实施方案 outputs/Mamba2实施方案.md 阶段 1/T1.4：
  测导出模型（全窗 20 步顺序 SSM）单步推理时延，对照 50 Hz 控制周期预算 20 ms。
  在目标部署硬件（如 Jetson）上运行本脚本得实测值；本机（RTX 4090 / CPU）数据
  仅为参考上限。

用法（n1 conda 环境）：
  python tests/bench_infer_latency.py [--jit policy_jit.pt] [--iters 500] [--device cuda|cpu]
决策口径（方案 T1.4）：单步时延 < 20 ms → T1.5 增量导出转可选项；
  超预算 → T1.5 升为必做。
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
BUDGET_MS = 20.0  # 50 Hz 控制周期


def build_traced(device):
    from fast_td3.fast_td3 import MambaActor
    from fast_td3.fast_td3_utils import EmpiricalNormalization
    from rsl_rl.runners.off_policy_runner import OffPolicyRunner

    actor = MambaActor(
        n_obs=FLAT, n_act=N_ACT, num_envs=4, init_scale=0.01,
        hidden_dim=1024, num_stack=SEQ_LEN, obs_dim=OBS_DIM,
        mamba_d_model=128, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_headdim=32, mamba_n_layers=2,
        device=device,
    ).to(device).eval()
    norm = EmpiricalNormalization(shape=FLAT, device=device).eval()
    stub = SimpleNamespace(actor=actor, obs_normalizer=norm)
    fd, path = tempfile.mkstemp(suffix="_policy_jit.pt")
    os.close(fd)
    OffPolicyRunner._save_jit_mamba(stub, path)   # 真实导出（CPU trace）
    traced = torch.jit.load(path)
    os.remove(path)
    return traced


def bench(fn, iters, device, warmup=20):
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
        times = []
        for _ in range(iters):
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    else:
        import time
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor(times)
    return t.mean().item(), t.median().item(), t.quantile(0.99).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jit", default=None, help="已导出 policy_jit.pt（缺省现场导出）")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", default=None, help="cuda / cpu（默认 cuda）")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.jit:
        traced = torch.jit.load(args.jit)
    else:
        traced = build_traced("cpu")   # 导出物恒为 CPU trace（部署等价物）

    traced = traced.to(device).eval()
    obs = torch.zeros(1, FLAT, device=device)
    with torch.no_grad():
        fn = lambda: traced(obs)
        mean_ms, med_ms, p99_ms = bench(fn, args.iters, device)

    print(f"device={device}  iters={args.iters}  model={'loaded ' + args.jit if args.jit else 'random-init'}")
    print(f"per-step latency: mean={mean_ms:.3f} ms  median={med_ms:.3f} ms  p99={p99_ms:.3f} ms")
    print(f"budget (50 Hz):   {BUDGET_MS:.1f} ms")
    if mean_ms < BUDGET_MS:
        print(f"RESULT: PASS —— 时延低于预算，T1.5 增量导出转可选项")
    else:
        print(f"RESULT: OVER BUDGET —— T1.5 增量 step 导出升为必做")
    sys.exit(0)


if __name__ == "__main__":
    main()
