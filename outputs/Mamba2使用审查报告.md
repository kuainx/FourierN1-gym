# N1 双足机器人上楼梯项目 —— Mamba2 使用审查报告（训练 + 部署）

- 审查日期：2026-11-19
- 审查范围：Mamba2 在 **训练**（FastTD3 / PPO 两条链路）与 **部署**（JIT 导出与推理契约）两方面的使用
- 审查方式：**只读代码审查 + 独立数值验证实验**（未修改任何项目代码）
- 训练入口：`python scripts/train.py --task=N1_FastTD3 --headless --num_envs=4096 --run_name=s3`

---

## 0. 结论摘要

**总体判断：Mamba2 的核心使用方式正确、自洽，工程上有若干亮点（帧堆叠对拍验证、导出剥离 Triton 依赖）。未发现 P0 级缺陷。**

风险一览：
- **P1** —— 部署契约（滑窗/零填充/obs 构成/动作还原）未随导出物文档化（§3.5）
- **P1** —— 全窗推理、无增量状态缓存，实机时延隐患（§3.5）
- **P2** —— 缺训练/导出数值对拍自动化（§3.5）；配置开关存在静默失效陷阱（§2.4）

另注意**当前检出配置并未启用 Mamba2**：
- `n1_config_fasttd3.py`：`use_mamba = False`（`sim_type = "sim_actor"` 开启的是 SEM 实验 2）
- PPO 配置：`class_name = "ActorCriticMLP"`（`ActorCriticMamba` 模板整段注释在 `n1_config.py` 609–638 行）

> 后续演进（长历史扩展、序列化训练改造、MLP 兼容性）见 §7 讨论稿。

---

## 1. 架构总览（Mamba2 相关链路）

> 任务注册：`task_registry` 将 `N1_FastTD3` 映射为 env `N1` + `N1MainBodyCfgFastTD3`（`n1_config_fasttd3.py`，runner=`OffPolicyRunner`）；PPO 基线走 `N1` + `N1MainBodyCfgPPO`（`n1_config_main_body.py`，runner=`OnPolicyRunnerMirror`）。

### 1.1 数据流（FastTD3 + Mamba2，即 `use_mamba=True`）

```
env (N1, num_obs=50/步, num_stack=20, num_pri_obs=183)
  └─ obs_stack: [最老帧 … 最新帧] 展平 (1000 维)          ← legged_robot_code.compute_observation_stack
        │  (reset 时 obs_stack[env_ids] 置 0，再重新累积)
        ▼
OffPolicyRunner.learn
  ├─ obs_normalizer (EmpiricalNormalization, 1000 维)     ← off_policy_runner.py:432
  ├─ MambaActor.explore(norm_obs)                          ← fast_td3.py:371
  │     reshape (B,20,50) → Mamba2Encoder(2 层 Mamba2) → 取 h[:,-1] → MLP 头 → tanh
  └─ SimpleReplayBuffer（单帧存储 50 维，sample 时 _gather_stack 重建 20 帧窗，
        跨 reset 边界按 env 语义清零）→ 训练时 actor 消费重建窗，critic 消费 183 维特权单步
```

### 1.2 数据流（PPO + Mamba2，`policy.class_name="ActorCriticMamba"`）

```
obs_stack (1000 维) ──► ActorCriticMamba._encode_actor_obs
                          reshape (B,20,50) → Mamba2Encoder → MLP actor 头（高斯策略）
pri_obs (183 维)    ──► critic_proj 线性投影 → MLP critic 头（无时序、非对称 AC）
```

### 1.3 部署链路

```
play.py (EXPORT_POLICY=True)
  ├─ PPO:   ActorCriticMamba.save_jit  → ActorJitWrapper + Mamba2EncoderJit
  └─ FastTD3: OffPolicyRunner.save_jit → _save_jit_mamba → MambaPolicyJit
        （两者都：纯 PyTorch Mamba2 复现替代 Triton 内核 → torch.jit.trace → policy_jit.pt）
部署端输入：展平 20 帧滑窗（[最老..最新]，1000 维）→ 输出 tanh 归一化动作
```

### 1.4 配置开关位置

| 链路 | 配置文件 | 开关 | 当前值 |
|---|---|---|---|
| FastTD3 | `legged_gym/envs/n1/n1_config_fasttd3.py` → `fast_td3` 段 | `use_mamba`（+ `mamba_d_model/d_state/d_conv/expand/headdim/n_layers`） | **False**（headdim=32，其余默认 128/16/4/2/2） |
| PPO | `legged_gym/envs/n1/n1_config_main_body.py` → `policy`（继承 `n1_config.py`） | `class_name = "ActorCriticMamba"` + `seq_len`（=堆叠帧数） | **ActorCriticMLP**（Mamba 模板注释于 `n1_config.py` 609–638） |

共用：`n1_config_main_body.py` 中 `use_stack=True, num_stack=20`（`n1test` 为 5，不同）。

---

## 2. 训练侧审查

### 2.1 Mamba2 编码器使用方式（两条链路共用 `Mamba2Encoder`）

- 输入 `(B, L=20, 50)`，`input_proj → [Mamba2 + 残差 + LayerNorm] × 2 → h[:, -1, :]`（`mamba_encoder.py`）。
- **窗口顺序正确**：`obs_stack` 拼接方式为「丢最老、新帧追加在末尾」（`legged_robot_code.py:1049`），即展平序为 [最老…最新]；Mamba 因果扫描取 `h[:, -1]` 恰为最新帧 —— 时序方向匹配。✔
- **无状态窗口设计自洽**：不携带 conv_state/ssm_state，`reset()` 为空实现合理；训练（整窗并行）与推理（同一整窗前向）**前向完全一致**，无隐藏状态错位风险。✔
- 结构/参数流与 mamba_ssm 官方 `Mamba2` 语义一致（`headdim=32 → nheads=8`；真实约束是 in_proj 输出维 `D=2*d_model*expand+2*d_state+nheads` 为 8 的倍数，见 `docs/deploy_contract_mamba.md` §8-4）。

### 2.2 FastTD3 链路（`OffPolicyRunner` + `MambaActor`）

- `MambaActor`（`fast_td3.py:371`）：`self.seq_len = num_stack`（时间维）与 SEM 的 `seq_len`（单纯形组数）参数命名重载但在构造中已隔离，不会串味。✔
- Actor 时序（20×50）+ Critic 非对称（183 维特权单步）设计合理（critic 观测含 base_lin_vel 等特权量，`n1_code.py:37`）。✔
- 观测归一化 `EmpiricalNormalization(1000)` 在 rollout 与 sample 两侧口径一致，且随 checkpoint 保存、烘焙进导出模型。✔
- `explore(dones=prev_dones)` 在 episode 边界重采样探索噪声尺度，语义正确。✔

### 2.3 帧堆叠 / 重置 / 回放窗重建（本次审查重点，结论：正确）

关键机制（存在隐蔽的张量别名语义，但当前实现是正确的）：

1. `compute_observation_stack` 用 `torch.cat` **重新绑定** `obs_stack`（新张量），因此 runner 持有的 `obs` 引用保留 step 前内容 → **(s_t, a_t) 配对正确**（entry t 存的 obs 帧 = 动作时刻的最新帧，next 帧 = step 后新帧）。
2. done 当步 `reset_idx` 会**原地清零** runner 仍引用的 obs 张量（`legged_robot_code.py:1875`）→ 终止转移存入 buffer 的 s_t 为全零窗（代码注释已声明“当步帧被清零丢失”，属已知取舍）。
3. `SimpleReplayBuffer` 只存单帧，`_gather_stack` 重建 20 帧窗并按 done 清零规则与 env 语义对齐（obs 模式：最后一个 done 及其之前清零；next 模式：仅清 done 行之前，保留 done 行的新 episode 首帧）。✔
4. **帧堆叠对拍**：`verify_stack_frames=True`（配置开启）训练初期逐步比对 buffer 重建与 env 当步返回，遇 reset 自动延长验证窗口，失配直接报错并输出逐帧诊断。✔ 这是很好的工程保障。

PPO 链路（`PPO.act` → `transition.observations` 存引用 → `process_env_step` 中 `copy_` 进 storage）：由于上述原地清零发生在 `copy_` 之前，**PPO 同样丢弃终止步真实 obs（存全零窗）**，与 FastTD3 行为一致（利于两算法公平对比），但 PPO 侧代码无注释说明。

### 2.4 训练侧问题清单

| 级别 | 问题 | 位置 | 说明 / 建议 |
|---|---|---|---|
| P2 | 终止步真实 obs 被清零后入库（s_t=全零窗） | `off_policy_runner.py:452-473`、`ppo.py:179/228`、`legged_robot_code.py:1875` | 两条链路一致、影响有界（全零窗仅出现在终止转移，部署时输入窗恒含真实最新帧），FastTD3 侧已注释为已知语义。若要修正：step 前 `obs.clone()` 入库，并同步修改 `_gather_stack` 清零规则 + 对拍口径 |
| P2 | `ActorCriticMamba` 不校验 `seq_len` 与 env `num_stack` 一致性，`actor_num_input // seq_len` 为静默整除 | `actor_critic_mamba.py:263` | 若 policy 配置 `seq_len=20` 与 env `num_stack` 不一致，维度整除时**静默错位**（把两帧拼作一帧）、不整除时才 crash。建议加 `assert actor_num_input % seq_len == 0 and seq_len == env.num_stack`；改造时建议将 `policy.seq_len` 改名 `bptt_len` 消歧（两参数命名分离约定见 §7.4） |
| P2 | PPO 的 Mamba 开关模板有“双注释”陷阱：`#     # class_name = "ActorCriticMamba"` | `n1_config.py:610` | 整段反注释后 `class_name` 仍不生效，静默回落 ActorCriticMLP（仅 kwargs warning）。启用 PPO+Mamba 时务必在 `N1MainBodyCfgPPO.policy` 显式写 `class_name = "ActorCriticMamba"` |
| P2 | 跨算法实验头容量不对等 + 注释失实 | `fast_td3.py:425` vs `n1_config.py:621` | MambaActor 头为 128→256→256→128→64，注释称 “aligned with PPO-Mamba [256,256,128,64]”，但 PPO 模板实为 `actor_hidden_dims=[256,128]`。做 PPO vs FastTD3 的 Mamba 对比实验前需统一 |
| P3 | 双重导入路径：`from mamba_encoder import Mamba2Encoder`（sys.path hack）与 `from rsl_rl.modules...` 并存 | `fast_td3.py:8-13,404`、`off_policy_runner.py:1221` | 同一文件两个模块身份，`isinstance` 跨路径会失效、打包易碎。建议统一为包内导入 |
| P3 | 代码默认 `headdim=64`（nheads=4）与约束注释冲突 | `fast_td3.py:394`、`actor_critic_mamba.py:191` | 当前配置显式 `mamba_headdim=32`（nheads=8）安全；若漏配该键走默认值，会触发 `causal_conv1d with channel last layout requires strides ... multiples of 8`（RD 4090 实测，非 Triton 约束） |
| P3 | `torch.compile` 默认开启 `_update_main/_update_pol`，图内含 mamba_ssm 自定义 Triton 算子 | `off_policy_runner.py:361-369` | 通常以 graph break 方式工作，但版本兼容性需实测；建议 `use_mamba=True` 时验证或对 Mamba 路径关闭 compile |

---

## 3. 部署侧审查

### 3.0 训练与部署的两种计算模式（概念澄清）

Mamba2 权重在训练与部署中共用，但存在两条**数学等价**的计算路径——「训练整窗并行、部署单步递推」是 SSM 类模型的标准分工：

| | 训练：整窗并行扫描 | 部署：增量递推 |
|---|---|---|
| 输入 / 计算 | 一次输入整个历史窗（本项目 20 帧），`Mamba2.forward` parallel scan | 每控制步只输入 1 帧，`Mamba2.step` + 跨步缓存 conv_state / ssm_state，O(1)/步 |
| 动机 | GPU 并行 + 梯度沿整窗反传，训练吞吐优先 | 控制周期预算有限，历史不重算，时延优先 |
| 本项目现状 | ✔ 已如此实现（§2.1） | ✘ 未实现：导出物为全窗模型，每步重算（§3.5 P1） |

两种路径在**同一初始状态 + 同一输入序列**下输出一致（数值验证见 §4 实验 1），因此把部署端改造成 `step()` 式增量推理在数学上是安全的；改造时的两个注意点（episode 复位显式清零状态、20 帧外衰减尾巴）集中见 §7.3。

### 3.1 导出链路

- PPO：`export_policy_as_jit` → `ActorCriticMamba.save_jit`（`actor_critic_mamba.py:404`）
- FastTD3：`play.py:82` 对无 `algorithm` 属性的 runner 调 `OffPolicyRunner.save_jit` → `_save_jit_mamba`（`off_policy_runner.py:1219`）
- 两者思路一致：**用纯 PyTorch 逐帧扫描的 `Mamba2LayerJit` 替换 Triton 内核版 Mamba2**，deepcopy 到 CPU 后 `torch.jit.trace`。导出模型不依赖 mamba_ssm/Triton —— 对实机部署是正确且必要的设计。✔
- 观测归一化统计随导出模型一起烘焙（`MambaPolicyJit` 内嵌 `EmpiricalNormalization`，trace 时 eval 模式，统计不再更新）。✔

### 3.2 Mamba2LayerJit 复现正确性（已数值验证）

`Mamba2LayerJit`（`actor_critic_mamba.py:13`）与 mamba_ssm 官方 `Mamba2.step` 逐帧递推做过数值对拍，结论**等价**（fp32、B=3、L=20、随机权重，数值详见 §4 实验 1）。逐项比对结论（与 mamba_ssm 源码一致）：

- in_proj 分解 `z0,x0,z,xBC,dt` 切分 ✔
- 因果 conv1d 切片（等价 `[:, :-(d_conv-1)]`）+ SiLU ✔
- `softplus(dt + dt_bias)`、`A=-exp(A_log)`、`dA=exp(ΔA)`、`dBx=Δ·B·x`、`y=C·h+D·x` ✔
- RMSNormGated（`norm_before_gate=False`：先 `x·silu(z)` 再 RMS 归一）✔
- `d_mlp>0` 时 `cat([silu(z0)*x0, y])` ✔

一个**潜在缺口**（当前配置不触发）：`Mamba2LayerJit` 只实现了 `rmsnorm=True` 的门控路径；若未来把 Mamba2 建成 `rmsnorm=False`，mamba_ssm 会走 `y·silu(z)` 门控而 JIT 版会漏掉 z 门控。当前 `Mamba2Encoder` 恒用默认 `rmsnorm=True`，无实际影响。

另：AMP(bf16) 训练与 fp32 导出存在数值漂移，量级可接受；如需部署端逐位对齐，可在导出后做端到端对拍。

### 3.3 trace 行为实测

- **batch 维未被固化**：同一 trace 模型 batch=1 与 batch=8 均正常（torch 2.4.1）。✔（跨 torch 版本迁移时建议重测；如需彻底规避可改 `view(-1, L, D)`）
- **seq_len=20 被展开为常量**（20 步循环 trace 时 unroll）：部署输入必须恰好 20 帧，不能改窗长。

### 3.4 部署契约（部署端必须遵守，否则训练/实机不一致）

1. **滑窗维护**：20 帧移位寄存器，新帧追加在**末尾**、丢最老帧；展平顺序 [最老…最新]。
2. **零填充语义**：启动与每次复位（摔倒/急停恢复）后，窗内未累积的帧位置**补零**（不是重复首帧、不是保持旧值），与训练 env 的 `obs_stack` 清零重建完全一致。
3. **观测构成逐项复刻** `n1_code.compute_observation_profile`（每步 50 维）：`commands×commands_scale(3) + base_ang_vel×scale(3) + projected_gravity×scale(3) + dof_pos_offset×scale(13) + dof_vel×scale(13) + actions×scale(13) + swing_clock(2)`；
   其中 `swing_clock` 为步态时钟，部署端需按训练同款口径生成；`actions` 是上一步下发的动作（×action_scale 后）。
4. **动作输出**：tanh 归一化动作（[-1,1]），部署端需按训练配置的 action_scale / PD 增益 / decimation / 控制延迟约定还原为关节指令。
5. **不要额外做观测 clip**：训练中 `obs_stack` 取的是 clip 之前的值（`legged_robot_code.py:674` 的 clip 发生在入栈之后；`clip_observations=100` 实际极少触发，影响小）。
6. **窗长固定 20**，观测顺序/缩放与归一化统计不得更换（换配置必须重训/重导出）。

### 3.5 部署侧问题清单

| 级别 | 问题 | 位置 | 说明 / 建议 |
|---|---|---|---|
| P1 | 部署契约靠口口相传：滑窗、零填充、obs 构成、动作还原均未随导出物固化/文档化 | `save_jit` 产物 + `n1_code.py` | 实机部署前写一份 deploy 接口说明（§3.4 即草案），并在部署端做“同一段录制观测 → Python 策略 vs 实机策略动作逐位比对” |
| P1 | 全窗推理、无增量状态缓存，实机时延风险（计算模式辨析见 §3.0） | `Mamba2EncoderJit` / `_save_jit_mamba` | 每控制步重算 2 层×20 步顺序 SSM；Mamba2 原生支持 conv_state/ssm_state 的 O(1)/步递推（`Mamba2.step`）却未用于导出。算力受限（如 Jetson）时可能超控制周期。docstring “recursively during inference” 即此意图但失实（`fast_td3.py:374-376`）。建议实现带 cache 的 step 版导出（等价性已验证 §4 实验 1；改造注意点见 §7.3） |
| P2 | 无自动化数值对拍：训练 Triton 前向 vs 导出 JIT 前向 | 仓库缺测试 | 本次人工验证递推等价（§4），建议固化为单测：(a) `Mamba2Encoder` vs `Mamba2EncoderJit` 逐输出比对；(b) `actor(obs)` vs `traced(obs)` 端到端动作比对（含归一化） |
| P2 | trace 固化 seq_len=20；batch 维在不同 torch 版本 trace 行为可能不同 | `actor_critic_mamba.py:429`、`off_policy_runner.py:1281` | 跨版本迁移时重测；可改 `view(-1, L, D)` 规避 batch 固化风险 |
| P3 | `testplay.py` 用 `N1test`（num_stack=5）+ 硬编码动作，不能加载/验证 N1 主配置（num_stack=20）的 checkpoint | `testplay.py:43`、`n1test/n1_config_main_body.py:19` | 仅调试用；实机前验证请用 `play.py`（EXPORT_POLICY=True 导出 + 内置 rollout 回放） |

---

## 4. 验证记录（本次审查执行的实验）

| # | 实验 | 结果 |
|---|---|---|
| 1 | `Mamba2LayerJit` vs mamba_ssm `Mamba2.step` 逐帧递推（fp32，B=3，L=20，随机输入/权重） | max\|diff\|=4.77e-07，**等价** ✔ |
| 2 | `torch.jit.trace` 后 batch=8 推理（项目同款 `view(batch_size, L, D)` 写法，torch 2.4.1） | 正常，**batch 未固化** ✔ |
| 3 | 代码逐行比对（split/conv 因果切片/softplus/dA/dBx/y/RMSNormGated/d_mlp） | 与 mamba_ssm 一致 ✔ |
| 4 | 帧堆叠 buffer 重建语义推演（obs/next 两种模式 × done 清零规则） | 与 env 语义一致，且有 `verify_stack_frames` 运行时对拍兜底 ✔ |

---

## 5. 亮点（建议保留的做法）

1. `verify_stack_frames` 帧堆叠对拍（含 reset 后延长验证 + 逐帧诊断输出）—— 防住了帧栈类 bug 中最难查的一类。
2. 导出端纯 PyTorch 复现替代 Triton 内核 —— 部署正确方向，且复现精度经验证达 fp32 水平。
3. replay buffer 单帧存储 + 采样时重建 —— 内存降为 1/20，且重建语义与 env 对齐。
4. Actor 时序 + Critic 非对称特权的设计与论文常用做法一致；窗口式（而非 RNN 隐状态）设计使训练/推理天然一致、reset 处理无坑。

---

## 6. 建议行动（分三档，均不在本次修改）

**A. 部署前必做**

1. 按 §3.4 写出部署接口文档；录制一段仿真轨迹做「Python 策略 vs 部署策略」动作逐位对拍（含归一化）。
2. 固化对拍单测：(a) `Mamba2Encoder` vs `Mamba2EncoderJit` 逐输出比对；(b) 端到端 actor vs traced 动作比对。
3. 实机算力评估：先测全窗推理单步时延；若超预算，实现带 conv_state/ssm_state 的增量导出（以官方 `Mamba2.step` 为参照，见 §7.5；等价性已验证 §4 实验 1；改造注意点见 §7.3）。

**B. 开 Mamba 实验前必做**

4. 显式设置 `class_name="ActorCriticMamba"`（PPO）/ `use_mamba=True`（FastTD3），并校验 `seq_len == num_stack`（§2.4）。
5. 统一 PPO/FastTD3 的 MLP 头容量后再做跨算法对比（§2.4）。
6. 小项：修正 `MambaActor` docstring、统一 `mamba_encoder` 导入路径、`Mamba2Encoder` 构造时校验 `actor_num_input % seq_len == 0`、`Mamba2LayerJit` 补 `rmsnorm=False` 分支或加 assert。

**C. 演进路线（对应 §7）**

7. 路径 A 消融：`num_stack` 20→40/80 重训，验证长历史对爬楼收益存不存在。
8. PPO 序列化（rollout 是现成连续序列，改动面小）→ FastTD3 序列 replay + burn-in（工程量最大、回报也最大）→ step() 递推部署（与序列训练同构）。

---

## 7. 演进方案讨论稿：长历史扩展与序列化训练改造

> 本节为前瞻性设计讨论（非本次审查结论），回答两个问题：Mamba 能否学超过 20 步的历史、改造方案能否兼容 MLP 基线。

### 7.1 长历史可行性：Mamba 能，但要区分两条路

时间尺度：`sim.dt=0.005 × decimation=4` → 控制 50 Hz，**20 帧 ≈ 0.4 秒 ≈ 一个步态周期**。若想利用"前 1~2 秒的地形上下文"（如连续台阶高度模式），需要更长历史。

模型容量对比（本项目维度，hidden 256；训练激活显存两者都随 batch×L 线性温和增长，不单列）：

| 窗长 L | MLP 仅首层参数（L×50×256） | Mamba 编码器参数 |
|---|---|---|
| 20（当前） | 256K | 与 L 无关（恒定） |
| 100 | 1.28M | 同上 |
| 500 | 6.4M（fp32 ≈ 26MB；+Adam 状态 ×3 ≈ 77MB） | 同上 |

- **MLP 的短板**：参数随 L 线性涨，且无时间权重共享（展平窗当普通向量，每个历史位置学一套独立权重），样本效率低；窗长定死历史上限。
- **Mamba 的优势**：参数与 L 无关（路径 A 加窗只付计算）；`ssm_state` 是固定大小的压缩记忆，递推视角可携带任意长历史（路径 B）。
- **两点诚实的限制**：① 当前是窗口式训练，梯度只穿过 20 步——**换 Mamba 不等于自动获得长历史**，20 步以外的信息模型没动力学；② SSM 的记忆是学习到的指数衰减（A 矩阵决定时间常数），"能带无限历史"≠"能记住无限久远"，有效记忆长度取决于训练中学到的衰减。

### 7.2 两条扩展路径

| | 路径 A：加长窗口 | 路径 B：序列化训练 + 状态递推 |
|---|---|---|
| 做法 | `num_stack` 20→40/80，重训 | 训练改序列式（§7.3），部署改 `step()` 递推 |
| 历史上限 | 窗长封顶 | 理论无限（受衰减时间常数约束） |
| 改动量 | 只改配置（buffer 单帧存储不变；部署契约同步改窗长/归一化/trace） | 训练循环 + buffer + 导出接口 |
| 风险 | 低 | 中（off-policy 陈旧状态等，见 §7.3） |
| 定位 | **先做**：验证长历史对爬楼收益存不存在 | 收益确认后再上 |

### 7.3 路径 B 具体改造："改训练方式"的完整含义

改的**不是算法、不是网络结构、不是导出格式**，而是三件事：喂给策略的数据形态、梯度回传的时间跨度、隐藏状态的生命周期。

**现状（窗口式 feed-forward）**：PPO 64 步 rollout / FastTD3 单步入 buffer，采样时切成互不相干的 20 帧窗样本；Mamba 看 `(B,20,50)` 取 `h[:,-1]`，梯度只穿 20 步，`ssm_state` 用完即弃——**记忆由输入窗充当**，64 步 rollout 里躺着的连续历史被浪费。

**目标（序列式 + 截断 BPTT）**：

1. **采样单元**：连续片段 `o_{t-K+1}…o_t`（K=64~256，同 env 同 episode，按 done 切段）；观测回到逐帧 50 维，时序聚合交给 Mamba；
2. **前向**：`(B,K,50)` 整段过 Mamba2，**每步都输出动作** `(B,K,13)`；
3. **loss**：TD/policy loss 在序列每一步都算（样本利用率 ×K），done 处 mask + 不做 bootstrapped target；
4. **梯度**：BPTT 穿过 K 步（TD 误差能训到 t−K 步的表征——**梯度能到达的地方，模型才有动力把信息存进 ssm_state**）；
5. **截断**：chunk 交界 `detach`——前向状态照常传递（记忆连续），梯度到此为止（显存/算力有界）；
6. **burn-in**（FastTD3 特有难点）：off-policy 下序列初始状态会随网络更新"过期"，标准解法是序列前多取 M 步（如 16）只热身状态不算 loss；PPO 无此问题（on-policy 现算现用）。

代码改动点：

| 位置 | 现状 | 改成 |
|---|---|---|
| `SimpleReplayBuffer`（`fast_td3_utils.py`） | 存单帧，`_gather_stack` 重建窗 | 存/采连续片段（按 done 切段）；存储层可保持单帧 + 序列化取数 |
| `Mamba2Encoder`/`MambaActor`（`fast_td3.py`） | 输入 `(B,20,50)`，只输出末步 | 输入 `(B,K,50)`，输出 `(B,K,13)`；`reset()` 真正清状态 |
| `OffPolicyRunner._update_main/_update_pol` | 对独立转移算 loss | 对序列逐步算 loss + done mask |
| `EmpiricalNormalization` | 1000 维整窗统计 | 50 维逐帧统计（部署契约同步简化；废旧 checkpoint 不兼容，需重训） |
| runner 采集 | 不管状态 | 每 env 携带 `ssm_state`，done 清零（正是部署要做的事，训练顺手完成） |
| PPO 侧 | `rollout_storage` 存整窗 | 存 50 维单帧 + 序列取样——**64 步 rollout 天然就是 64 步序列，改造最便宜** |

**增量递推改造的两个注意点**（若在序列化之前先做 step() 部署改造，必须处理）：

1. **reset 语义**：训练靠「窗口零填充」隐式清历史；增量递推版必须在 episode 开始/复位时**显式清零 conv_state / ssm_state**，否则上一局的记忆会漏进新局。
2. **严格等价 vs 近似等价**：全窗重算与训练逐位一致；持久状态递推的 ssm_state 会携带 20 帧以外的衰减记忆（训练分布假设窗外为零）。实践中 Mamba 的衰减使这部分尾巴通常可忽略，但若要求与训练完全对拍，可用环形缓冲区保存最近 20 帧定期重算校准，或明确接受该漂移。

**训练/部署同构红利**：序列式训练后 `ssm_state` 本身是模型约定的记忆载体（训练跨步传递、部署跨步传递、reset 语义一致），上述两个注意点自然消解——done 清状态与训练一致、"严格 vs 近似等价"之争消失；导出可顺手从"整窗"改为"step+状态"。

### 7.4 MLP 兼容性：一条数据主干，两种消费方式

**结论：完全兼容，且共用序列管线后 Mamba vs MLP 对比更公平。** 一段 K 步片段内含 K 个 20 帧滑窗——MLP 只是"只用末端 20 帧"的消费者。

| 组件 | 共用？ | 说明 |
|---|---|---|
| 回放缓冲（单帧 50 维 + done） | ✅ | 现有 buffer 本就单帧存储，存储层零改动 |
| 序列采样器（片段 + 前缀） | ✅ | 同一片段喂 Mamba 整段，也喂 MLP 切窗 |
| done mask / 逐步 TD loss | ✅ | 两边逐步算 loss、done 处 mask |
| 归一化（建议统一 50 维逐帧） | ✅ | MLP 组装窗前逐帧 normalize 即可 |
| 前缀机制 | ✅ 概念共用 | MLP 前缀 = 19 帧窗填充；Mamba = burn-in 热身（更长）；都"前缀不算 loss" |
| Critic（183 维特权单步） | ✅ 不动 | 非对称 critic 无时序需求 |
| actor 前向 / 梯度跨度 | ❌ 分支 | Mamba 整段递推 + BPTT 跨 K；MLP 每步 20 帧窗 + 窗内梯度 |
| 部署 | ❌ 互不干扰 | MLP 契约不变（20 帧展平窗）；Mamba 可走 step() 递推 |

**K 是"对比公平性旋钮"**：K=20 时两边上下文/样本/loss 完全相同，只剩模型结构一个变量（理想消融设置）；K>20 时长历史只对 Mamba 生效，MLP 自动只看末端 20 帧（验证长历史收益的实验组）。切换只改一个配置数字。

**备注：MLP 与 Mamba 的长度必须分 2 个参数存。** MLP 的窗长（如 20 帧，决定输入 1000 维）与 Mamba 的序列长 K（64~256，决定 BPTT 跨度）语义不同、演化节奏不同，不得共用 `num_stack` 一个键——例如 `mlp_window_len`（= 现 `num_stack`，也决定部署端 MLP 契约）与 `seq_len`（序列采样/BPTT 长度，仅 Mamba 消费）。两者关系约定：MLP 始终取序列末端 `mlp_window_len` 帧切窗；校验 `seq_len >= mlp_window_len`。

两个细节：① MLP 同片段相邻窗重叠 19/20 帧，相关性升高，可用 `loss_stride`（如每 4 步算一次 loss）缓解或直接接受；② `use_mamba` 现有开关就是分支点：`use_mamba=True → actor(B,K,50) 整段 BPTT`，`False → 逐步组装 20 帧窗前馈`。

### 7.5 Mamba2 尚未用上的能力（仅列与 Mamba 直接相关项）

> 只盘点 Mamba2 特有的能力/参数/接口；通用 RL 训练技巧（辅助预测任务、特权蒸馏、时序 critic 等）与 Mamba 无直接关系，不计入本节。

| 项 | 现状 | 说明 / 建议 |
|---|---|---|
| 官方增量推理 API | 未用（导出为整窗 `Mamba2LayerJit`） | mamba_ssm 的 `Mamba2.step()` + `allocate_inference_cache()` 是官方递推接口（含 fused `selective_state_update`）。做 §6-A3 增量导出时优先以官方 step 为参照，避免自制 cache 的语义偏差 |
| 记忆容量与时间常数旋钮 | `d_state=16`；`dt_min/max`、`A_init_range` 走默认 | `d_state` 即 SSM"记忆槽"数量，扫 32/64 是最便宜的容量杠杆；时间常数机制见 §7.1 限制②。须随 §7 路径 B 一起扫，单独调无收益 |
| 结构效率旋钮 | `ngroups=1`（默认）；`d_ssm`/`d_mlp` 混合结构未用 | `ngroups` 为分组 SSM 的参数/算力效率旋钮；`d_ssm`（部分维度走 SSM、其余走 gated MLP）为 Mamba2 论文常见混合配置，且 `Mamba2LayerJit` 的 d_mlp 分支已预留 |
| `cu_seqlens` / `seq_idx` 变长打包 | 未用 | mamba_ssm `forward` 原生支持变长序列打包；§7.3 序列训练时不同 episode 段打包进同一 batch 可直接利用 |

（双向扫描：actor 必须因果、不适用；仅时序 critic 场景可考虑非因果扫描，不单独展开。）

注意：上述旋钮/接口大多要与 §7 的序列化训练改造配套才有实效；孤立启用（如只加大 `d_state` 而仍用 20 帧窗口训练）收益有限。
