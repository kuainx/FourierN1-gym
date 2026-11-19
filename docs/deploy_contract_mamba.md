# Mamba2 策略部署契约（deploy contract）

- 版本：v1（2026-11-19，随实施方案 outputs/Mamba2实施方案.md 阶段 1/T1.1 固化）
- 适用导出物：`policy_jit.pt`（`save_jit` / `_save_jit_mamba` 产出，TorchScript trace 模型）
- 适用模型：FastTD3 + MambaActor 与 PPO + ActorCriticMamba 两条链路的 Mamba2 策略
  （MLP 基线的窗/观测契约与本文件一致，仅网络结构不同）
- **发布约定**：每次导出 `policy_jit.pt` 时，必须与本契约的版本号一并交付；
  契约与导出物不同版号不得混用。任何观测顺序/缩放/窗长/归一化变更都视为
  **破坏性变更**：必须重训 + 重导出 + 本契约升版。
- 依据：`outputs/Mamba2使用审查报告.md` §3.4（部署契约草案）、§3.0–§3.5（部署侧审查）。

---

## 0. 接口摘要

| 项 | 约定 |
|---|---|
| 输入 | `obs_flat`，float32，形状 `(batch, 1000)` = 20 帧 × 50 维滑窗展平 |
| 输出 | `action`，float32，形状 `(batch, 13)`，tanh 归一化关节动作（[-1, 1]） |
| 控制频率 | 50 Hz（`sim.dt=0.005 × decimation=4` → 20 ms/步） |
| 推理模式 | **全窗推理**：每控制步重算整窗（2 层 × 20 步顺序 SSM）；无增量状态缓存 |
| 归一化 | `EmpiricalNormalization` 统计（1000 维整窗）已烘焙进导出图（eval 模式）；部署端**不得**再做归一化或反归一化 |
| batch 维 | 不固化（torch 2.4.1 实测 batch=1/8 均可）；但 **窗长 20 被 trace 固化**，不可改 |

---

## 1. 滑窗维护（frame stacking）

1. 维护一个 **20 帧移位寄存器**：每控制步将**最新单帧观测**追加到**末尾**，同时丢弃最老帧。
2. 展平顺序固定为 **[最老帧 … 最新帧]**，即 `obs_flat = window.reshape(20 * 50)`，
   其中 `window[t]` 为从旧到新的第 t 帧。喂给模型前不再做任何重排。
3. 与训练侧 `obs_stack`（`use_stack=True, num_stack=20`）语义逐位一致；
   重建规则可对照 `FastTD3/fast_td3/fast_td3_utils.py` 的 `_gather_stack`。

## 2. 零填充语义（启动 / 复位）

1. 模型启动与**每次复位**（摔倒、急停恢复、换 episode）后，窗内尚未累积到的帧位置
   **补零（全 50 维为 0）**。
2. **禁止**用首帧重复填充、**禁止**保持复位前旧值。这与训练 env 的 `obs_stack`
   清零重建完全一致，任何偏离都会引入训练时不存在的输入分布。
3. 复位后的首个控制步：窗 = 19 个零帧 + 1 个新帧（最新帧在末尾）；
   其后逐步移位，20 步后窗完全由真实帧构成。

## 3. 观测构成（每步 50 维，逐项复刻 `n1_code.compute_observation_profile`）

按下列顺序拼接（维度合计 3+3+3+13+13+13+2 = 50）：

| # | 分量 | 维数 | 缩放 | 说明 |
|---|---|---|---|---|
| 1 | `commands[:, 0:3] × commands_scale` | 3 | `commands_scale = [lin_vel, lin_vel, ang_vel] = [2.0, 2.0, 0.25]` | 指令：vx, vy, yaw_rate（base_velocity 档） |
| 2 | `base_ang_vel × obs_scales.ang_vel` | 3 | `0.25` | 机体系角速度 |
| 3 | `base_projected_gravity × obs_scales.gravity` | 3 | `1.0` | 重力在机体系投影 |
| 4 | `dof_pos_offset × obs_scales.dof_pos` | 13 | `1.0` | 关节相对默认姿态的偏移（左腿 6 + 右腿 6 + 腰 1） |
| 5 | `dof_vel × obs_scales.dof_vel` | 13 | `0.05` | 关节角速度 |
| 6 | `actions × obs_scales.action` | 13 | `1.0` | **上一步下发的动作**（tanh 输出 × action_scale 后），首步为 0 |
| 7 | `swing_clock` | 2 | 无 | 步态时钟，须按训练同款口径生成（见下） |

要点：

- **`actions` 是上一步动作**（×action_scale 之后的关节指令值，不是网络原始 tanh 输出）；
  复位后第一步该分量为 0。
- **`swing_clock`** 为步态时钟（2 维），部署端必须按训练侧同款公式/相位约定生成；
  stand/walk 任务差异由 `commands` 与 `swing_clock` 承载，不得自造时钟口径。
- `dof_pos` 用的是 **offset**（相对默认姿态），不是绝对关节角。
- 各分量的物理量单位均为国际单位制，缩放系数见上表（来源
  `legged_robot_config.py` `normalization.obs_scales`；如配置变更以训练配置为准并升版本）。

## 4. 动作还原

1. 模型输出 `action ∈ [-1, 1]^{13}`（tanh 归一化动作，关节顺序与观测中 dof 一致）。
2. 部署端按训练配置还原关节指令：
   `joint_target = default_dof_pos + action × action_scale`（position 控制），
   再由 PD 控制器（训练同款 **PD 增益**）跟踪。
3. 控制约定必须与训练一致：
   - **decimation**：策略 50 Hz → 物理 200 Hz（`sim.dt=0.005`，decimation=4），
     部署端 PD 环频率与之一致；
   - **控制延迟**：按训练侧动作延迟约定（如有启用）同款补偿；
   - **action_scale / PD 增益 / 默认姿态**：随导出物从训练配置读取，不得目测修改。
4. 动作侧 clip：训练用 `clip_actions=100`（几乎不触发）；部署端不必额外 clip。

## 5. 观测 clip 禁令

**不要对观测做任何额外 clip/限幅。** 训练中 `obs_stack` 取的是 clip 之前的值
（`legged_robot_code.py` 中 `clip_observations=100` 发生在入栈之后，实际极少触发）。
部署端若自行加 clip（哪怕看起来“更安全”），都会引入训练时不存在的非线性。

## 6. 窗长与配置冻结

1. **窗长固定 20 帧**（≈ 0.4 s ≈ 一个步态周期）。导出 trace 已将 20 步循环展开为
   常量，部署输入必须恰好 20 帧（1000 维），不能改窗长、不能改观测顺序/缩放。
2. 归一化统计（1000 维整窗 `EmpiricalNormalization`）烘焙在导出图内，随 checkpoint
   版本绑定；**换配置（窗长/观测构成/归一化）必须重训 + 重导出 + 本契约升版**。
3. 部署端不得在模型前后再接自定义归一化/标准化层。

## 7. 数值对拍（部署验收流程）

部署方接入后、实机运行前，必须完成两级对拍（判据与用例见
`tests/test_mamba_export_parity.py`）：

1. **单测级**（开发侧，每次导出后）：
   - (a) `Mamba2Encoder`（Triton 训练前向）vs `Mamba2EncoderJit`（导出复现）逐输出比对；
   - (b) 端到端 `actor(norm_obs)` vs `traced(obs)` 动作比对（含归一化烘焙）；
   - (c) batch=1 与 batch=8 回归（batch 维不固化）。
   - 判据：fp32 下 `max|diff| < 1e-5`（参考基线 4.77e-07）。
2. **轨迹级**（部署方）：录制一段仿真/实机轨迹（逐帧 50 维原始观测 + 动作），
   在部署端运行导出模型，与 Python 策略输出逐位/近逐位比对（含归一化）。
   AMP bf16 训练 → fp32 导出的漂移属预期，量级记录在案即可
   （可参考 `tests/run_traj_parity.py`）。

## 8. 已知限制（本版契约）

1. **全窗推理时延**：每控制步重算 20 步顺序 SSM。目标硬件（如 Jetson）单步时延
   需实测（T1.4）；若超 20 ms 预算，须实施增量 `step()` 导出改造（T1.5，
   以官方 `Mamba2.step` + `allocate_inference_cache` 为参照，注意复位清状态
   与窗外衰减尾巴两个语义点）。
2. **终止步观测语义**：训练侧终止转移存入的是清零窗（已知取舍，两条链路一致）；
   部署时输入窗恒含真实最新帧，无影响。
3. torch 版本迁移（非 2.4.1）时，trace 的 batch 维行为须重测（用例 c）。

---

## 附：验收清单（部署方签字用）

- [ ] 滑窗顺序/展平与 §1 一致（对照 `_gather_stack` 自查）
- [ ] 复位补零语义与 §2 一致（无首帧重复/旧值保持）
- [ ] 50 维观测逐项构成/缩放与 §3 一致（含 swing_clock 口径、actions=上一步）
- [ ] 动作还原（action_scale / PD / decimation / 控制延迟）与 §4 一致
- [ ] 无额外观测 clip（§5）
- [ ] 归一化未重复施加（§0/§6）
- [ ] 轨迹级对拍通过（§7，diff 量级已记录）
- [ ] 单步时延 < 20 ms 预算（或已立项 T1.5 增量导出）
