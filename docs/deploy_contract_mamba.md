# Mamba2 策略部署契约（deploy contract）

- 版本：v1（2026-11-19，M1 部署接口固化）
- 适用导出物：`policy_jit.pt`（`save_jit` / `_save_jit_mamba` 产出，TorchScript trace 模型）
- 适用模型：FastTD3 + MambaActor 与 PPO + ActorCriticMamba 两条链路的 Mamba2 策略
  （MLP 基线的窗/观测契约与本文件一致，仅网络结构不同）
- **发布约定**：每次导出 `policy_jit.pt` 时，必须与本契约的版本号一并交付；
  契约与导出物不同版号不得混用。任何观测顺序/缩放/窗长/归一化变更都视为
  **破坏性变更**：必须重训 + 重导出 + 本契约升版。
- 依据：Mamba2 训练/导出链路代码审查与数值对拍；结论已固化于本文与
  `tests/test_mamba_export_parity.py`。

---

## 0. 接口摘要

| 项 | 约定 |
|---|---|
| 输入 | `obs_flat`，float32，形状 `(batch, 1000)` = 20 帧 × 50 维滑窗展平 |
| 输出 | `action`，float32，形状 `(batch, 13)`。量程**随算法不同**：FastTD3+MambaActor 末端有 `nn.Tanh()` → `[-1,1]`；PPO+ActorCriticMamba 为线性头（正态均值），**无 [-1,1] 保证**，部署端不得再套 tanh（见 §4） |
| 控制频率 | 50 Hz（`sim.dt=0.005 × decimation=4` → 20 ms/步） |
| 推理模式 | **全窗推理**：每控制步重算整窗（2 层 × 20 步顺序 SSM）；无增量状态缓存 |
| 归一化 | 两条链路**都不要求**部署端做归一化/反归一化，但原因不同：<br>• **FastTD3**：训练用 `EmpiricalNormalization`（1000 维整窗统计），已**烘焙进导出图**（`_save_jit_mamba`，eval 模式）；<br>• **PPO**：全程**没有**归一化——`on_policy_runner` 无 normalizer、env 只做 `obs_scales` 手工缩放，导出图里不存在任何 norm 节点。 |
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
| 1 | `commands[:, 0:3] × commands_scale` | 3 | `commands_scale = [lin_vel, lin_vel, ang_vel] = [1.0, 1.0, 1.0]` | 指令：vx, vy, yaw_rate（base_velocity 档） |
| 2 | `base_ang_vel × obs_scales.ang_vel` | 3 | `1.00` | 机体系角速度 |
| 3 | `base_projected_gravity × obs_scales.gravity` | 3 | `1.0` | 重力在机体系投影 |
| 4 | `dof_pos_offset × obs_scales.dof_pos` | 13 | `1.0` | 关节相对默认姿态的偏移（左腿 6 + 右腿 6 + 腰 1） |
| 5 | `dof_vel × obs_scales.dof_vel` | 13 | `0.10` | 关节角速度 |
| 6 | `actions × obs_scales.action` | 13 | `1.0` | **上一步网络原始输出**（仅经 `clip_actions=±100`，实际不触发；**未乘** `action_scale`），首步为 0 |
| 7 | `swing_clock` | 2 | 无 | 步态时钟，须按训练同款口径生成（见下） |

要点：

- **`actions` 是上一步网络原始输出**，只经过 `clip_actions`（`normalization.clip_actions=100`，
  `legged_robot_code.py:260-261` 取的是其对称值，**不是**配置里的 `clip_actions_max/min` 数组），
  **没有乘 `action_scale`**（`_compute_torques` 里的 `actions_scaled` 是局部变量，不回写
  `self.actions`，`legged_robot_code.py:1647`）。N1 全关节 `action_scale=1`（`n1_config.py:284-303`）
  所以两者数值相同，但**语义必须按“未缩放”理解**：一旦修改 `action_scale`，obs 第 6 段仍是未缩放值。
  复位后第一步该分量为 0。
- **`swing_clock`** 为步态时钟（2 维），stand/walk 任务差异由 `commands` 与 `swing_clock`
  承载，不得自造时钟口径。训练侧唯一实现：`legged_robot_fftai_code.py:139-150`
  （stand 判定见 `n1_code.py:135-146`），逐行复刻如下（`dt = sim.dt × decimation = 0.02 s`）：

  ```
  phase = n * dt / 0.9                  # 周期 0.9 s = 45 控制步；n = episode 内已过步数
  s     = sin(2 * pi * phase)
  c_l   = -s  if s <= -0.10 else 0.0    # 分量 0：左腿
  c_r   =  s  if s >=  0.10 else 0.0    # 分量 1：右腿
  swing_clock = [c_l, c_r]              # 两分量恒 ≥ 0，交替脉冲

  if ||commands[0:2]|| <= 0.10 and abs(commands[2]) <= 0.10:   # stand 指令
      swing_clock = [0.0, 0.0]
  ```

  **两处时序细节必须一致**（否则整窗输入偏一步）：
  - `n` 在本步 `post_physics_step` 开头先自增、随后才算时钟（`legged_robot_code.py:779→782`），
    故复位后的第 1 个控制步用 `n = 1`，`phase = 0.0222` 而非 0；
  - `reset_idx` 在时钟算完之后才把 `n` 清零（`:790`），且 `compute_observations` 更晚（`:797`），
    所以**复位后首帧观测里的 `swing_clock` 是上一步（终止时刻）的旧值**，不是 `[0,0]`。
    部署端若在复位时直接置零，首帧会有 1 步相位偏差（20 ms）。
  - stand 判定用的是**未缩放的原始指令**（`commands_scale` 不参与该判定）。
- `dof_pos` 用的是 **offset**（相对默认姿态），不是绝对关节角。
- 各分量的物理量单位均为国际单位制。**缩放系数必须取自本次训练使用的配置**：
  N1 在 `n1_config.py:495-503` 覆盖了 `normalization.obs_scales`
  （`lin_vel=ang_vel=gravity=dof_pos=action=1.0`、`dof_vel=0.10`），
  而 `commands_scale` 不是独立参数，由 `legged_robot_code.py:338-352` 现算为
  `[lin_vel, lin_vel, ang_vel]`，故为 `[1, 1, 1]`。
  **禁止从 legged_gym 上游默认的 `legged_robot_config.py` 推**（那份是
  `ang_vel=0.25 / dof_vel=0.05 / commands_scale=[2,2,0.25]`，与 N1 不符）；
  若配置变更须以 checkpoint 的 `train_cfg` 为准并升版本号。

## 4. 动作还原

1. 模型输出 `action`（13 维，关节顺序与观测中 dof 一致）。量程随算法而定：
   - **FastTD3**：末端 `nn.Tanh()`，`action ∈ [-1,1]^13`，直接乘 `action_scale`；
   - **PPO**：线性输出层（正态分布的均值），**不在 [-1,1] 内**，训练与部署都不做 tanh、
     也不做额外 clip（`clip_actions=100` 实际不触发）。部署 PPO 导出物时若补一层 tanh，
     等于给策略加训练时不存在压缩，属破坏性偏差。可否越界取决于训练收敛结果，
     部署端**不得**假设动作有界。
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

1. **单测级**（开发侧，每次导出后）：`tests/test_mamba_export_parity.py`。
   判据**按参照系分两档**，不可合并成单一条阈值——两者的 fp32 累序本就不同：

   | 档 | 用例 | 参照系 | 判据 |
   |---|---|---|---|
   | 严格 | a1：`Mamba2EncoderJit` vs 官方 `Mamba2.step` 递推 | 同一计算顺序 | `max abs diff` < 1e-5 |
   | 严格 | b1：PPO eager-export vs traced（导出链路自洽） | 同一计算顺序 | `max abs diff` < 1e-5（实测 0.0） |
   | 漂移 | a2：`Mamba2Encoder`（Triton 并行扫描）vs `Mamba2EncoderJit`（顺序扫描） | **不同扫描顺序** | `max abs diff` < 5e-4 |
   | 漂移 | b2：`actor(norm_obs)` vs `traced(obs)`（含归一化烘焙） | 不同扫描顺序 + 归一化融合 | `max abs diff` < 5e-4 |

   5e-4 是 fp32 并行累加 vs 顺序累加的舍入差异预算，**不是模型误差**；
   参考实测（RTX 4090 / fp32 / 2026-11-19）：a1 `8.3e-07`、a2 `2.2e-05`（B=1）
   与 `9.4e-05`（B=8）、b2 `1.0e-06`~`4.5e-06`。
   同时须回归 batch=1 与 batch=8（batch 维不固化）。
2. **轨迹级**（部署方）：录制一段仿真/实机轨迹（逐帧 50 维原始观测 + 动作），
   在部署端运行导出模型，与 Python 策略输出逐位/近逐位比对（含归一化）。
   本档走**漂移档 5e-4**（部署端递推顺序与训练并行扫描不同）；
   若部署端也按顺序扫描复现，可要求严格档 1e-5。
   AMP bf16 训练 → fp32 导出的漂移属预期，量级记录在案即可。
   对拍时须加载 checkpoint 内的 `obs_normalizer_state_dict`，不要用随机统计。

## 8. 已知限制（本版契约）

1. **全窗推理时延**：每控制步重算 20 步顺序 SSM。本机参考实测（RTX 4090 / fp32 /
   随机初始化策略）：单次前向 **4.5 ms**（低于 50 Hz 的 20 ms 预算）；
   x86 CPU **49.6 ms**（超预算）。目标硬件（如 Jetson）仍须实测；
   若超 20 ms 预算，须实施增量 `step()` 导出改造，以官方 `Mamba2.step` +
   `allocate_inference_cache` 为参照，注意复位清状态与窗外衰减尾巴两个语义点。
2. **终止步观测语义**：训练侧终止转移存入的是清零窗（已知取舍，两条链路一致）；
   部署时输入窗恒含真实最新帧，无影响。
3. torch 版本迁移（非 2.4.1）时，trace 的 batch 维行为须重测（用例 c）。
4. **Mamba 头维度约束**：约束来自 `causal_conv1d` 的 channel-last
   布局，不是 Triton。融合路径里 conv 的输入是 `zxbcdt` 的切片视图
   （`mamba_ssm/ops/triton/ssd_combined.py:964-965`），其通道 stride 等于 in_proj 输出维
   `D = 2*d_model*expand + 2*ngroups*d_state + nheads`，而 `causal_conv1d` 要求
   `x.stride(0)`、`x.stride(2)` 均为 8 的倍数，即 **`D % 8 == 0`**。
   默认 `expand=2, d_state=16` 下该条件等价于 `nheads % 8 == 0`（16 组 d_model×headdim
   组合实测与预测 16/16 吻合）。不满足时的原始报错（headdim=64 / d_model=128）：

   ```
   RuntimeError: causal_conv1d with channel last layout requires strides
   (x.stride(0) and x.stride(2)) to be multiples of 8
   ```

   因此导出侧不受影响（Jit 走纯 PyTorch 卷积），但**训练前向会直接崩**；
   改架构时保持 `nheads=8`，即 `headdim = d_model*expand//8`
   （d_model=96→24、128→32、192→48，均在实测可用集内）。

---

## 附：验收清单（部署方签字用）

- [ ] 滑窗顺序/展平与 §1 一致（对照 `_gather_stack` 自查）
- [ ] 复位补零语义与 §2 一致（无首帧重复/旧值保持）
- [ ] 50 维观测逐项构成/缩放与 §3 一致（含 swing_clock 口径、actions=上一步）
- [ ] 动作还原（action_scale / PD / decimation / 控制延迟）与 §4 一致
- [ ] 无额外观测 clip（§5）
- [ ] 归一化未重复施加（§0/§6）
- [ ] 轨迹级对拍通过（§7，档位 5e-4 与 diff 量级已记录）
- [ ] 单步时延 < 20 ms 预算（否则需实施增量 `step()` 导出）
