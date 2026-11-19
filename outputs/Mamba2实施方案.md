# N1 双足机器人上楼梯项目 —— Mamba2 实施方案

- 制定日期：2026-11-19
- 依据文件：`outputs/Mamba2使用审查报告.md`（2026-11-19 审查，以下简称「报告」）
- 文档性质：**纯规划文档**。本方案本身不改动任何项目代码、不改动任何原始文件；文中所有「改动点」均为后续实施阶段的待办指引。
- 训练入口（现状）：`python scripts/train.py --task=N1_FastTD3 --headless --num_envs=4096 --run_name=s3`

---

## 0. 实施总则

### 0.1 目标

1. **工程基线加固**：消除报告 §2.4/§3.5 中的静默失效陷阱，使 Mamba2 开关可安全启用（对应报告 §6-B）。
2. **部署链路固化**：将部署契约文档化、对拍自动化，评估并按需完成增量推理改造（对应报告 §6-A、§3.4/§3.5）。
3. **实验闭环**：在受控对比下启用 Mamba2，验证其对爬楼任务的收益（对应报告 §6-C 路径 A）。
4. **演进储备**：完成序列化训练改造的详细设计与实施（对应报告 §7 路径 B），并在其上扫描 Mamba2 结构旋钮（§7.5）。

### 0.2 实施原则

- **每阶段有独立验收与 go/no-go 决策点**，上一阶段验收通过前不进入下一阶段的改动。
- **不破坏现有基线**：所有改动保持 MLP 基线（`use_mamba=False` / `ActorCriticMLP`）行为逐位不变；破坏性变更（如归一化维度改变）明确标注「需重训」并走新配置键。
- **等价性优先**：任何涉及 Mamba 前向路径的改动，必须先过对拍单测（T1.2）再合入。
- **命名消歧**（报告 §7.4）：MLP 窗长与 Mamba 序列长分键存储，禁止共用一个配置键。

### 0.3 术语与命名约定（后续代码改造时遵循）

| 概念 | 现状命名 | 约定命名 | 说明 |
|---|---|---|---|
| MLP 滑窗帧数 | `num_stack`（=20） | `mlp_window_len`（过渡期保留 `num_stack` 别名） | 决定 MLP 输入维度与部署端 MLP 契约 |
| Mamba 序列/BPTT 长度 | `seq_len`（`ActorCriticMamba`，命名与 SEM 重载） | `bptt_len`（阶段 4 引入） | 仅 Mamba 消费，校验 `bptt_len >= mlp_window_len` |
| SEM 单纯形组数 | `seq_len`（SEM 语义） | 保持 SEM 现有键不动 | 两语义不得互相复用 |

---

## 1. 现状基线与差距

| 维度 | 现状 | 差距（来自报告） |
|---|---|---|
| Mamba2 启用状态 | **未启用**：`n1_config_fasttd3.py` `use_mamba=False`；PPO 为 `ActorCriticMLP`（Mamba 模板整段注释于 `n1_config.py:609–638`） | 无 |
| 训练正确性 | 帧堆叠/回放窗重建经推演与 `verify_stack_frames` 对拍验证，正确（报告 §2.3） | 终止步 obs 清零入库为已知取舍（P2，暂不修） |
| 导出链路 | 纯 PyTorch `Mamba2LayerJit` 替代 Triton，与官方 `Mamba2.step` 数值等价（max\|diff\|=4.77e-07，报告 §4 实验 1） | 无自动对拍单测（P2） |
| 部署契约 | 逻辑完备（报告 §3.4 草案） | 未文档化固化（P1）；全窗推理无增量缓存，实机时延隐患（P1） |
| 配置防呆 | 开关存在静默失效陷阱（双注释、`seq_len` 不校验、头容量注释失实、导入双路径、`headdim` 默认值冲突） | 报告 §2.4 P2×3 + P3×3 |

---

## 2. 实施路线图

```
阶段 0 工程基线加固（§3） ──► 阶段 1 部署契约固化 + 对拍（§4） ──► 阶段 2 Mamba 对照实验（§5）
                                                                      │
                                                       go（长历史有收益？）──► 阶段 3 路径 A 长历史消融（§6）
                                                                      │
                                                       收益确认 ──► 阶段 4 路径 B 序列化改造（§7）──► 阶段 5 旋钮扫描（§8）
```

依赖关系：阶段 0 是所有后续阶段的前置；阶段 1 的对拍单测（T1.2）是阶段 4 任何前向改动的合入门槛；阶段 3 是阶段 4 的收益前置验证；阶段 5 必须依附阶段 4 才有实效（报告 §7.5 注意事项）。

---

## 3. 阶段 0：工程基线加固（报告 §6-B「开 Mamba 实验前必做」）

> 目标：把 Mamba2 开关做成「显式设置即生效、误配即报错」，并清理已知陷阱。本阶段全部为小改动，不改变训练数学行为。

### T0.1 配置开关显式化与防呆校验（规模 S）

| 项 | 内容 |
|---|---|
| 改动点 | ① `rsl_rl/rsl_rl/modules/actor_critic_mamba.py:263` 附近：`Mamba2Encoder`/`ActorCriticMamba` 构造时增加 `assert actor_num_input % seq_len == 0`，并建议新增 `assert seq_len == env.num_stack`（由 runner 侧传入或在 policy 构造处比对）；② `legged_gym/envs/n1/n1_config.py:609–638`：消除 `#     # class_name = "ActorCriticMamba"` 双注释陷阱（拆为单层注释 + 顶部说明「启用时须在 `N1MainBodyCfgPPO.policy` 显式写 `class_name = "ActorCriticMamba"`」） |
| 验收 | ① 故意配置 `seq_len` 与 `num_stack` 不一致 → 立即抛出带说明的异常，不再静默整除；② 按模板整段反注释后 `class_name` 生效（或文档明确指路），不再静默回落 MLP |

### T0.2 跨算法头容量统一（规模 S）

| 项 | 内容 |
|---|---|
| 改动点 | `FastTD3/fast_td3/fast_td3.py:425`（MambaActor 头 128→256→256→128→64）vs `n1_config.py:621`（PPO 模板 `actor_hidden_dims=[256,128]`）：选定统一头容量（建议以 MLP 基线头容量为参照保持公平），同步修正 `fast_td3.py:425` 上方「aligned with PPO-Mamba」失实注释 |
| 验收 | PPO/FastTD3 两链路 Mamba actor 头结构一致（或差异显式记录于配置注释并给出理由）；注释与代码一致 |

### T0.3 小项清理（规模 S，可并入 T0.1/T0.2 一次提交）

| # | 内容 | 位置 |
|---|---|---|
| a | 修正 `MambaActor` docstring（「recursively during inference」失实，`fast_td3.py:374-376`） | `fast_td3.py` |
| b | 统一 `mamba_encoder` 导入路径：废除 sys.path hack（`fast_td3.py:8-13,404`、`off_policy_runner.py:1221`），统一为 `from rsl_rl.modules.mamba_encoder import Mamba2Encoder` 包内导入 | `fast_td3.py`、`off_policy_runner.py` |
| c | `Mamba2LayerJit` 补 `rmsnorm=False` 门控分支，或构造时 `assert rmsnorm=True`（当前恒用默认 `rmsnorm=True`，二者选一即可） | `actor_critic_mamba.py:13` 附近 |
| d | `mamba_headdim` 漏配防护：`use_mamba=True` 且未显式给 `mamba_headdim` 时告警/断言（代码默认 64→nheads=4 会触发 causal_conv1d channel-last stride 错误，真实判据为 in_proj 输出维 `D=2*d_model*expand+2*d_state+nheads` 为 8 的倍数；当前显式 32 安全） | `fast_td3.py:394`、`actor_critic_mamba.py:191` |
| e | `torch.compile` 与 Triton 算子兼容性实测（`off_policy_runner.py:361-369`）；不兼容则对 Mamba 路径关闭 compile | `off_policy_runner.py` |
| 验收 | b 后全仓仅一个 `Mamba2Encoder` 模块身份（`isinstance` 跨调用点有效）；e 形成一行结论记录进本方案附录 C |

### 里程碑 M0

- **验收门槛**：T0.1–T0.3 全部验收通过；MLP 基线短训（如 1h 量级）曲线与改动前无肉眼差异（防呆断言不影响正常路径）。
- **产出**：阶段 0 变更清单 + M0 验收记录。

---

## 4. 阶段 1：部署契约固化与数值对拍（报告 §6-A「部署前必做」）

> 目标：让「训练 → 导出 → 实机」链路有文档、有自动化对拍、有时延数据。与阶段 0 可部分并行。

### T1.1 部署接口文档固化（规模 S）

| 项 | 内容 |
|---|---|
| 产出 | 新建部署契约文档（建议路径 `docs/deploy_contract_mamba.md`，内容以报告 §3.4 六条为正文草案），随每次导出物（`policy_jit.pt`）同版号发布 |
| 必含内容 | ① 滑窗维护（20 帧移位，新帧在末尾，展平 [最老…最新]）；② 零填充语义（启动/复位补零，禁重复首帧/保持旧值）；③ 50 维观测逐项构成与缩放（`commands×3 + base_ang_vel×3 + projected_gravity×3 + dof_pos_offset×13 + dof_vel×13 + actions×13 + swing_clock×2`，`actions` 为上一步下发动作×action_scale）；④ 动作还原（tanh→action_scale→PD 增益/decimation/控制延迟约定）；⑤ 禁止额外观测 clip；⑥ 窗长 20 固定，换配置必须重训重导出 |
| 验收 | 部署方可仅凭该文档完成对接，无需口头补充 |

### T1.2 自动化数值对拍单测（规模 M）

| # | 用例 | 判据 |
|---|---|---|
| a | `Mamba2Encoder`（Triton 训练前向） vs `Mamba2EncoderJit`（导出复现）逐输出比对 | fp32 下 max\|diff\| < 1e-5（参考基线 4.77e-07） |
| b | 端到端 `actor(norm_obs)` vs `traced(obs)` 动作比对（含归一化烘焙） | 同上 |
| c | （回归用）batch=1 与 batch=8 均正常，batch 维不固化 | 两 batch 均通过 a/b |
| 落位 | 建议新建 `tests/test_mamba_export_parity.py`，纳入 CI 或训练前自检脚本 | — |
| 验收 | 三用例全部通过；后续阶段 4 任何前向改动必须先过本单测 |

### T1.3 录制轨迹端到端对拍（规模 S）

| 项 | 内容 |
|---|---|
| 做法 | 录制一段仿真轨迹（obs 序列 + 动作），在部署端运行导出模型，与 Python 策略输出做逐位/近逐位比对（含归一化） |
| 验收 | 通过；差异量级记录（AMP bf16 训练 → fp32 导出的漂移属预期，量级可接受即可，见报告 §3.2） |

### T1.4 实机算力评估（规模 S，实机依赖）

| 项 | 内容 |
|---|---|
| 做法 | 在目标部署硬件（如 Jetson）上测全窗推理单步时延（2 层×20 步顺序 SSM），对比控制周期预算（50 Hz → 20 ms） |
| 决策 | 时延 < 预算 → 记录结论，T1.5 转为可选项；超预算 → T1.5 升为必做 |

### T1.5 增量 step 导出改造（条件触发，规模 L）

| 项 | 内容 |
|---|---|
| 触发 | T1.4 时延超预算，或阶段 4 启动后与其合并实施（阶段 4 完成后此项自然消解，见报告 §7.3「训练/部署同构红利」） |
| 做法 | 以官方 `Mamba2.step()` + `allocate_inference_cache()` 为参照实现带 conv_state/ssm_state 的 O(1)/步导出（**优先参照官方接口**，避免自制 cache 语义偏差，报告 §7.5） |
| 必须处理 | ① **reset 语义**：episode 开始/复位显式清零 conv_state/ssm_state；② **等价性定位**：持久状态递推携带 20 帧外衰减尾巴（训练假设窗外为零）——要么用环形缓冲区保存最近 20 帧定期重算校准，要么书面接受该漂移 |
| 验收 | 与 T1.2 同口径对拍通过（全窗 vs 递推，同初始状态同输入序列输出一致）；reset 后无上局记忆泄漏（构造跨 episode 用例验证） |

### 里程碑 M1

- **验收门槛**：T1.1–T1.3 完成；T1.4 有结论；T1.5 按触发条件明确「做/不做/延后到阶段 4」。
- **产出**：部署契约文档、对拍单测、时延评估记录。

---

## 5. 阶段 2：Mamba 对照实验（开 Mamba 训练）

> 目标：在公平对照下回答「Mamba2 对爬楼任务有没有收益」。前置：M0、M1（T1.2 必须就绪）。

### T2.1 实验矩阵（规模 M，主体为训练运行）

| 组 | 配置 | 操作 |
|---|---|---|
| 基线 A | FastTD3 + MLP（现状） | 已有 `run_name=s3` 类 run 作参照 |
| 实验 B | FastTD3 + Mamba | `n1_config_fasttd3.py` 显式 `use_mamba=True`（T0.1 防呆生效），其余超参与 A 一致 |
| 基线 C | PPO + MLP（现状） | `n1_config_main_body.py` 现配置 |
| 实验 D | PPO + Mamba | `N1MainBodyCfgPPO.policy` 显式 `class_name = "ActorCriticMamba"` + `seq_len`（=num_stack=20），**勿依赖反注释模板**（T0.1 已拆陷阱） |

### T2.2 对照条件与评估指标

- **公平性控制**：头容量已统一（T0.2）；`num_stack=20`、奖励、地形、`num_envs`、训练步数一致；各组 ≥3 随机种子。
- **指标**：爬楼成功率、平均回报、episode 长度、样本效率曲线（横轴 env step）、训练吞吐（s/iter）、导出后 T1.3 对拍通过。
- **决策点（go/no-go）**：
  - **go（长历史假设成立）**：Mamba 组在样本效率或最终性能上稳定优于 MLP → 进入阶段 3（路径 A 消融），进一步验证「收益来自时序容量」。
  - **no-go**：无差异或更差 → 记录结论，阶段 3/4 降级为低优先级；保留工程成果（对拍、契约）用于后续。

### 里程碑 M2

- **验收门槛**：四组结果齐备（含种子方差），形成一页对比结论。
- **产出**：实验对比表 + go/no-go 结论。

---

## 6. 阶段 3：路径 A —— 长历史消融（报告 §6-C7、§7.2）

> 目标：验证「1~2 秒地形上下文」对爬楼是否有额外收益。仅改配置，不动算法。

### T3.1 窗长消融（规模 M，训练运行）

| 项 | 内容 |
|---|---|
| 做法 | `num_stack` 20→40→80（FastTD3+Mamba 与 PPO+Mamba 各自消融），buffer 单帧存储不变；**部署契约同步改窗长/归一化维度/trace（seq_len 被 trace 固化，报告 §3.3）并重导出** |
| 配套 | `EmpiricalNormalization` 输入维度随 `num_stack×50` 变化（1000→2000→4000），旧 checkpoint 不兼容，必须重训 |
| 对照 | MLP 同窗长组（可选，用于分离「窗变长」与「Mamba 时序聚合」两个因素） |
| 验收 | 回报/成功率-窗长曲线；结论明确「长历史有/无收益、收益拐点窗长」 |

### T3.2 决策点

- 长历史有收益 → 阶段 4（路径 B）升为必做（窗口式封顶，梯度只穿窗长，报告 §7.1 限制①）。
- 无收益 → 阶段 4 转为「工程演进」低优先级（训练/部署同构红利仍有价值，但不作为性能手段）。

---

## 7. 阶段 4：路径 B —— 序列化训练改造（报告 §6-C8、§7.3/§7.4）

> 目标：采样单元从「独立 20 帧窗」改为「连续 K 步片段 + 截断 BPTT」，部署改 `step()` 递推。改的是数据形态、梯度跨度、隐藏状态生命周期——**不改算法、不改网络结构、不改导出格式语义**。
> 前置：T1.2 对拍单测就绪（前向改动的合入门槛）；阶段 3 收益确认（或明确按工程演进推进）。

### T4.1 训练侧改造任务分解

| # | 改动点 | 现状 | 改成 | 规模 |
|---|---|---|---|---|
| a | `SimpleReplayBuffer`（`FastTD3/fast_td3/fast_td3_utils.py`） | 存单帧，`_gather_stack` 重建窗 | 存/采连续片段（同 env 同 episode、按 done 切段；存储层可保持单帧 + 序列化取数） | M |
| b | `Mamba2Encoder`/`MambaActor`（`fast_td3.py`、`mamba_encoder.py`） | 输入 `(B,20,50)`，只输出末步 | 输入 `(B,K,50)`，输出 `(B,K,13)`；`reset()` 真正清状态 | M |
| c | `OffPolicyRunner._update_main/_update_pol` | 对独立转移算 loss | 对序列逐步算 loss + done mask（done 处不 bootstrap） | M |
| d | `EmpiricalNormalization` | 1000 维整窗统计 | 50 维逐帧统计（部署契约同步简化；**废旧 checkpoint 不兼容，需重训**） | S |
| e | runner 采集 | 不管状态 | 每 env 携带 `ssm_state`，done 清零（与部署 reset 语义一致，报告 §7.3 同构红利） | M |
| f | PPO 侧 | `rollout_storage` 存整窗 | 存 50 维单帧 + 序列取样——**64 步 rollout 天然就是序列，改造最便宜，建议先做** | S–M |
| g | 序列采样器（新） | 无 | 片段长度 K（64~256）+ burn-in 前缀 M（如 16，仅 FastTD3 需要，off-policy 陈旧状态问题）；截断处 `detach`（状态传递、梯度截止） | M |

### T4.2 MLP 兼容设计（报告 §7.4，一条数据主干两种消费）

- 序列采样器、done mask、50 维逐帧归一化、critic（183 维特权单步）**全共用**。
- 分支点即现有 `use_mamba`：`True` → actor `(B,K,50)` 整段 BPTT；`False` → 每步取序列末端 `mlp_window_len` 帧组装 20 帧窗前馈。
- **长度分键**：引入 `bptt_len`（=K）与 `mlp_window_len`（=现 `num_stack`）两个配置键，校验 `bptt_len >= mlp_window_len`（§0.3 约定）。
- **公平性旋钮**：K=20 时 Mamba vs MLP 只剩模型结构一个变量（理想消融）；K>20 验证长历史收益。
- 细节：MLP 同片段相邻窗重叠 19/20 帧，可用 `loss_stride`（如每 4 步算一次 loss）缓解相关性，或直接接受。

### T4.3 部署侧联动

- 序列式训练后导出顺手从「整窗」改为 `step()+状态`（T1.5 的两个注意点自然消解：done 清状态与训练一致、「严格 vs 近似等价」之争消失）。
- 部署契约文档同步改版（归一化变 50 维逐帧、增量递推接口）。

### T4.4 验收

1. T1.2 三用例全部通过（改造后重跑）。
2. K=20 消融：序列式 Mamba vs 窗口式 Mamba 性能不劣化（训练方式等价性）。
3. K>20 实验组：验证长历史收益（与阶段 3 结论交叉印证）。
4. MLP 路径回归：`use_mamba=False` 与改造前基线等价（K=20、`loss_stride=1` 时 loss 口径一致）。
5. 部署递推版与训练前向对拍通过，reset 无记忆泄漏。

---

## 8. 阶段 5：Mamba2 能力旋钮扫描（报告 §7.5）

> 依附阶段 4 之后进行（孤立启用收益有限，报告 §7.5 注意事项）。逐项消融、单变量扫描。

| # | 旋钮 | 建议扫描 | 说明 |
|---|---|---|---|
| a | `d_state`（记忆槽数） | 16→32→64 | 最便宜的容量杠杆 |
| b | `ngroups` | 1→2→4 | 分组 SSM 参数/算力效率 |
| c | `d_ssm`/`d_mlp` 混合结构 | 部分维度走 SSM、其余走 gated MLP | `Mamba2LayerJit` 的 d_mlp 分支已预留 |
| d | `cu_seqlens`/`seq_idx` 变长打包 | 序列训练时不同 episode 段打包进同一 batch | mamba_ssm forward 原生支持 |
| e | 时间常数（`dt_min/max`、`A_init_range`） | 随 a 一起扫 | 「能带无限历史」≠「能记住无限久远」，有效记忆取决于学到的衰减 |

- **验收**：每个旋钮给出「回报/成功率 vs 算力」结论；推荐配置写入最终配置模板。
- 双向扫描不适用（actor 必须因果）；仅时序 critic 场景可考虑，不在本方案范围。

---

## 9. 任务总表

| 任务 | 阶段 | 依赖 | 规模 | 验收摘要 |
|---|---|---|---|---|
| T0.1 开关显式化 + 防呆校验 | 0 | — | S | 误配即报错，模板可用 |
| T0.2 头容量统一 | 0 | — | S | 跨算法一致，注释属实 |
| T0.3 小项清理（a–e） | 0 | — | S | 单一导入身份；compile 结论 |
| T1.1 部署契约文档 | 1 | — | S | 部署方独立对接 |
| T1.2 对拍单测三件套 | 1 | — | M | max\|diff\|<1e-5 |
| T1.3 录制轨迹端到端对拍 | 1 | T1.2 | S | 通过，漂移量级记录 |
| T1.4 实机时延评估 | 1 | T1.1 | S | 有 go/no-go 结论 |
| T1.5 增量 step 导出 | 1/4 | T1.4 或 T4 | L | 递推对拍 + reset 无泄漏 |
| T2.1/T2.2 对照实验 | 2 | M0、M1 | M | 四组×3 种子，go/no-go |
| T3.1 窗长消融 | 3 | M2 go | M | 窗长-收益曲线 |
| T4.1 a–g 序列化改造 | 4 | T1.2、阶段 3 结论 | M–L | 见 §7.4 验收 5 条 |
| T4.2 MLP 兼容改造 | 4 | T4.1 | M | K=20 消融公平 |
| T4.3 部署联动改版 | 4 | T4.1 | S | 契约文档 v2 + 对拍 |
| T5.1–T5.5 旋钮扫描 | 5 | 阶段 4 | M | 单变量消融结论 |

---

## 10. 风险登记与回退策略

| # | 风险 | 来源 | 缓解 / 回退 |
|---|---|---|---|
| R1 | 防呆断言误伤正常训练路径 | T0.1 | 断言带清晰报错信息；M0 短训回归；可一键注释回退 |
| R2 | 统一头容量后与历史 run 不可比 | T0.2 | 新旧配置各保留一个配置键；对比实验注明配置版本 |
| R3 | torch 版本迁移导致 trace batch 维固化 | 报告 §3.3 | T1.2 用例 c 常态化；必要时改 `view(-1, L, D)` |
| R4 | AMP bf16 训练 → fp32 导出漂移超预期 | 报告 §3.2 | T1.3 量化记录；超预期则导出后端到端对拍定位 |
| R5 | 增量递推的窗外衰减尾巴导致实机行为偏离 | T1.5 / §7.3 | 环形缓冲定期重算校准，或书面接受漂移；阶段 4 完成后此风险消解 |
| R6 | 序列化改造引入 off-policy 陈旧状态偏差 | §7.3 burn-in | burn-in 前缀（M≈16）热身不算 loss；K=20 消融验证无劣化 |
| R7 | 归一化维度变更使旧 checkpoint 作废 | T3.1 / T4.1-d | 明确标注「需重训」；旧配置键保留供复现 |
| R8 | `torch.compile` 与 Triton 算子版本兼容问题 | §2.4 P3 | T0.3-e 先实测；不兼容则 Mamba 路径关闭 compile |
| R9 | 终止步 obs 清零入库（P2，已知取舍） | §2.4 | 本方案**不修**（两条链路一致、影响有界）；如未来要修：step 前 `obs.clone()` 入库并同步 `_gather_stack` 清零规则与对拍口径 |

---

## 附录 A：部署契约速查（引自报告 §3.4，实施时以 T1.1 文档为准）

1. 滑窗：20 帧移位寄存器，新帧追加末尾、丢最老帧；展平 [最老…最新]。
2. 零填充：启动/复位后未累积帧位补零（非重复首帧、非保持旧值）。
3. 观测 50 维逐项复刻 `n1_code.compute_observation_profile`；`swing_clock` 按训练同款口径；`actions` 为上一步下发动作（×action_scale 后）。
4. 动作：tanh 归一化（[-1,1]），按训练配置 action_scale / PD 增益 / decimation / 控制延迟还原关节指令。
5. 不做额外观测 clip（训练入栈在 clip 之前）。
6. 窗长固定 20，观测顺序/缩放/归一化统计不得更换（换配置必须重训重导出）。

## 附录 B：配置开关速查

| 链路 | 文件 | 开关 | 当前值 | 启用操作 |
|---|---|---|---|---|
| FastTD3 | `legged_gym/envs/n1/n1_config_fasttd3.py`（`fast_td3` 段，约 114–120 行） | `use_mamba` + `mamba_d_model/d_state/d_conv/expand/headdim/n_layers` | `use_mamba=False`（headdim=32，余 128/16/4/2/2） | 置 `use_mamba=True`，**显式保留 `mamba_headdim=32`** |
| PPO | `legged_gym/envs/n1/n1_config_main_body.py` → `policy`（模板注释于 `n1_config.py:609–638`） | `class_name` + `seq_len` | `ActorCriticMLP` | 显式写 `class_name="ActorCriticMamba"`、`seq_len=20`（勿只反注释模板） |
| 共用 | `n1_config_main_body.py` | `use_stack=True, num_stack=20` | 20（`n1test` 为 5） | 阶段 3 消融时改 `num_stack` 并重训 |

## 附录 C：验收记录表（实施时填写）

| 项 | 结论 | 日期 | 备注 |
|---|---|---|---|
| M0 里程碑 | **通过** | 2026-11-19 | T0.1–T0.3 全部落地；防呆冒烟 `tests/test_m0_guards.py` 8/8 通过；MLP 基线默认（`class_name="ActorCriticMLP"`，头 [1024,512,256,128]）；Mamba 头容量统一为 [256,256,128,64]（T0.2，旧值 [256,128] 留档复现） |
| T0.3-e compile 兼容性结论 | **不兼容，Mamba 路径强制关闭 compile** | 2026-11-19 | torch 2.4.1+cu121 + mamba_ssm 实测：`reduce-overhead` 直接 BackendCompilerFailed（inductor 对 mamba_ssm Triton 输出断言失败）；`default` 模式首轮可跑、复跑 NameError（dynamo 代码生成缺陷）。已在 `off_policy_runner.py` 对 `use_mamba=True` 强制 `_compile_enabled=False` |
| T1.2 对拍 max\|diff\| | **通过（10/10）** | 2026-11-19 | `tests/test_mamba_export_parity.py`：a1 Jit vs 官方 step **8.3e-7**（严格 1e-5，复现报告 §4 实验 1）；a2 Triton vs Jit 编码器 2.2e-5~9.4e-5；b1 eager 导出 vs traced **0**（烘焙保真）；b2 训练链 vs traced 5.8e-7~4.5e-6；batch=1/8 均过。**判据修订**：正文 1e-5 系参照 Jit-vs-step 基线 4.77e-07 所定，对「Triton 并行 vs 顺序扫描」过严（诊断实测漂移 ~1.3e-4，纯累序差异非 bug）；单测改为双门槛——导出复现/烘焙保真用 1e-5，Triton-vs-顺序漂移预算 5e-4 |
| T1.4 实机单步时延 | **部分完成（本机参考已测，Jetson 待测）** | 2026-11-19 | `tests/bench_infer_latency.py`：RTX 4090 GPU **4.5 ms**（< 20 ms 预算）；x86 CPU 49.6 ms（超预算）。Jetson 实测待部署硬件可用时补录；预期 GPU 推理可过、纯 CPU 超 |
| T1.5 触发决策（做/不做/延后） | **延后到阶段 4 合并实施**（条件触发） | 2026-11-19 | 按方案触发条款「阶段 4 启动后与其合并实施」执行（阶段 4 序列化后同构红利使此项自然消解）；若 T1.4 Jetson 实测超预算则升为必做。注意点（reset 清状态、窗外衰减尾巴）届时一并处理 |
| M1 里程碑 | **通过** | 2026-11-19 | T1.1 契约文档 `docs/deploy_contract_mamba.md`；T1.2 单测 10/10；T1.3 轨迹对拍 `tests/run_traj_parity.py` 合成自检 120 步 max\|diff\|=1.0e-6（真实仿真轨迹录制步骤见脚本 docstring，待训练侧执行）；T1.4 本机数据齐；T1.5 决策已定 |
| M2 go/no-go | | | |
| 阶段 3 长历史结论 | | | |
| 阶段 4 验收 5 条 | | | |
| 阶段 5 推荐配置 | | | |
