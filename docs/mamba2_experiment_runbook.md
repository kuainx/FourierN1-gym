# Mamba2 对照实验运行手册（阶段 2 / T2.1–T2.2）

- 版本：v1（2026-11-19，随 outputs/Mamba2实施方案.md 阶段 2 实施）
- 目的：在公平对照下回答「Mamba2 对爬楼任务有没有收益」，形成 M2 go/no-go 结论。
- 前置：M0（工程基线加固）与 M1（部署契约 + 对拍单测）已验收；
  **每次开训前先跑** `tests/test_mamba_export_parity.py`（10/10 通过）。

---

## 1. 实验矩阵（T2.1）

| 组 | 配置 | 开关操作 | 训练命令（在 `legged_gym/legged_gym/` 下执行） |
|---|---|---|---|
| 基线 A | FastTD3 + MLP | 无需改动（现状默认 `use_mamba=False`） | `python scripts/train.py --task=N1_FastTD3 --headless --num_envs=2048 --run_name=mamba2_A_s{SEED} --seed={SEED}` |
| 实验 B | FastTD3 + Mamba | `n1_config_fasttd3.py` → `fast_td3.use_mamba = True`（`mamba_*` 已显式就位，`mamba_headdim=32` 保留） | 同上，`--run_name=mamba2_B_s{SEED}` |
| 基线 C | PPO + MLP | 无需改动（M0 后默认 `class_name="ActorCriticMLP"`） | `python scripts/train.py --task=N1 --headless --num_envs=6144 --run_name=mamba2_C_s{SEED} --seed={SEED}` |
| 实验 D | PPO + Mamba | 取消 `n1_config_main_body.py` 中 `N1MainBodyCfgPPO.policy` 内的**整块注释**（该处模板含 `class_name="ActorCriticMamba"`、`seq_len`、`mamba_*`、统一头容量），跑完必须还原；**不要**改 `n1_config.py::N1CfgPPO.policy`（那是 C/D 共用父类，改了会把 mamba 键注入基线 C） | 同 C，`--run_name=mamba2_D_s{SEED}` |

要点：

- **执行目录**：`cd ~/N1/legged_gym/legged_gym && python scripts/train.py ...`。
  注意是内层 `legged_gym/legged_gym/`（`train.py` 在那里），不是仓库根下的 `legged_gym/`。
  等价写法：任意目录 `python -m legged_gym.scripts.train ...`（包已 egg-install）。
- **wandb**：`train.py:49` 无条件 `wandb.init`，没有关闭开关；离线/未登录时先
  `export WANDB_MODE=offline`（或 `wandb login`），否则会卡在初始化。
- 每组 **≥3 个随机种子**（`--seed=0/1/2`），run_name 带组号与种子（如 `mamba2_B_s0`）。
- 防呆已生效：实验 D 漏写 `class_name` 会直接报错（mamba 键误入 MLP 不再静默回落）；
  `seq_len` 与 `num_stack` 不一致构造即抛错；实验 B 漏配 `mamba_headdim` 会直接报错。
- 实验 B 的 `torch.compile` 会被自动强制关闭（T0.3-e，mamba_ssm 算子不兼容），属预期。
- 等实验 D/D' 之外不改任何其他键；Mamba 头容量已统一 [256, 256, 128, 64]（T0.2）。

## 2. 公平性控制清单（T2.2）

| 控制项 | 约定 |
|---|---|
| 模型结构变量 | Mamba vs MLP 仅变时序编码器（+ 头容量按 T0.2 统一口径）；A/B 与 C/D 各自内部其余超参逐键一致 |
| `num_stack` | 20（各组一致；阶段 3 消融时才改） |
| 奖励 / 地形 / 自模仿 | 各组一致（`n1_config_fasttd3.py` / `n1_config_main_body.py` 现配置不动） |
| `num_envs` | A/B 共用 2048；C/D 共用 6144（算法内一致） |
| 训练步数 | 同算法两组一致；记录实际停止点（env step 数为横轴） |
| 头容量 | 已统一（T0.2）；历史 run 与本轮不可比处以本表注明 |
| 参数量（实测） | PPO：C actor 1.716 M / critic 0.878 M → D actor 0.358 M / critic 0.090 M（4.8× / 9.8×）；FastTD3：A 1.684 M → B 0.358 M（4.7×）。**这是容量混淆项**：B/D 落后无法归因给时序建模，领先则说服力更强但需在结论里注明 |

参数量混淆的处置（二选一，**开跑前定**）：

1. 加一组容量对齐对照（推荐）：MLP 基线头降到 `[256, 256, 128, 64]`，与 Mamba 组等参，
   记作 C′/A′；原 C/A 保留为"历史口径"参照。
2. 反向对齐：把 Mamba 头加宽到与 MLP 等参（约 +1.3 M），CLI 侧改 `actor_hidden_dim`
   与 `mamba_d_model`，代价是偏离 T0.2 已统一的配方。

无论选哪条，M2 结论里必须写明"本对照不是等参对比"。

## 3. 评估指标与记录

指标（每组每种子）：

1. 爬楼成功率（评估口径统一：同一地形/指令脚本下 ≥N episode 统计）；
2. 平均回报 / episode 长度（训练日志 TensorBoard）；
3. 样本效率曲线（横轴 env step）；
4. 训练吞吐（s/iter）；
5. 导出后跑 `tests/test_mamba_export_parity.py` + `tests/run_traj_parity.py --traj <录制轨迹> --ckpt <ckpt>` 对拍通过。

结果记录表（复制填写）：

| 组 | 种子 | 成功率 | 最终回报 | 达到基线回报的 env step | s/iter | 对拍 | 备注 |
|---|---|---|---|---|---|---|---|
| A | 0/1/2 | | | | | | |
| B | 0/1/2 | | | | | | |
| C | 0/1/2 | | | | | | |
| D | 0/1/2 | | | | | | |

## 4. go/no-go 决策规则（M2）

- **go（长历史假设成立）**：Mamba 组（B 或 D）在样本效率或最终性能上**稳定**优于同算法
  MLP 组（多种子一致）→ 进入阶段 3（路径 A：`num_stack` 20→40→80 消融），
  验证收益是否来自时序容量。
- **no-go**：无差异或更差 → 结论记录进附录 C；阶段 3/4 降级为低优先级；
  保留工程成果（契约文档、对拍单测）供后续使用。
- 跨算法（B vs D）只做参考（算法差异非本实验变量），Mamba 收益结论以
  **同算法内 B vs A、D vs C** 为准。

## 5. 开训前自检（每轮必做）

- [ ] `tests/test_m0_guards.py` 10/10（防呆回归，含 D%8 头维度约束）
- [ ] `tests/test_mamba_export_parity.py` 10/10（前向/导出等价）
- [ ] 执行目录为 `legged_gym/legged_gym/`，且 `WANDB_MODE` 已按环境设好
- [ ] 开关状态与矩阵一致（A/C 默认即基线；B 只动 `use_mamba`；D 只动 policy 显式键）
- [ ] run_name / seed 已按约定命名（便于后续汇总）
