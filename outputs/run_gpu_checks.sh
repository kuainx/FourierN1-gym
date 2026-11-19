#!/usr/bin/env bash
# 在“有 GPU 的真实终端”里一次性跑完所有需要 GPU 的验收项。
# 用法： bash outputs/run_gpu_checks.sh
# 说明： WorkBuddy 的执行沙箱把 /dev/nvidia* 屏蔽了，torch 看不到 GPU；
#        该脚本必须在普通终端（非工具沙箱）里执行。
set -u

PY=/home/mao/.conda/envs/n1/bin/python
ROOT=/home/mao/N1
cd "$ROOT" || exit 1

echo "################ 0. GPU 环境 ################"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
           --format=csv,noheader || { echo "!! nvidia-smi 失败，先修驱动再继续"; exit 1; }
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())" || exit 1
if ! $PY -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "!! torch.cuda.is_available()=False，终止"; exit 1
fi

overall=0

echo
echo "################ 1. P0-3：nheads / headdim 复核 ################"
$PY outputs/probe_nheads_headdim.py || overall=1

echo
echo "################ 2. 导出保真（a1 严格 1e-5 / a2,b2 漂移 5e-4） ################"
$PY tests/test_mamba_export_parity.py || overall=1

echo
echo "################ 3. 推理时延（20ms 预算） ################"
$PY tests/bench_infer_latency.py || overall=1
# bench 现在按预算返回退出码：超预算 = 1（不再恒 0）

echo
if [ "$overall" -eq 0 ]; then
  echo "############ 完成：全部通过，把上面输出完整贴回来即可 ############"
else
  echo "############ 完成：存在失败项（见上，overall=$overall），请一并贴回 ############"
fi
exit "$overall"
