"""M0 验收冒烟：T0.1/T0.3 防呆断言行为检查（不含训练）。"""
import sys
sys.path.insert(0, "/home/mao/N1/rsl_rl")

import torch
from rsl_rl.modules.actor_critic_mamba import ActorCriticMamba
from rsl_rl.modules.actor_critic_mlp import ActorCriticMLP
from rsl_rl.modules.mamba_encoder import Mamba2Encoder

passed = failed = 0


def expect_raise(name, fn, needle):
    global passed, failed
    try:
        fn()
    except (ValueError, AssertionError) as e:
        if needle in str(e):
            print(f"[PASS] {name}: raised with expected message")
            passed += 1
        else:
            print(f"[FAIL] {name}: raised but message mismatch: {e}")
            failed += 1
    except Exception as e:
        print(f"[FAIL] {name}: wrong exception {type(e).__name__}: {e}")
        failed += 1
    else:
        print(f"[FAIL] {name}: no exception raised")
        failed += 1


def expect_ok(name, fn):
    global passed, failed
    try:
        fn()
    except Exception as e:
        print(f"[FAIL] {name}: unexpected {type(e).__name__}: {e}")
        failed += 1
    else:
        print(f"[PASS] {name}")
        passed += 1


# 1. seq_len 与 num_stack 不一致 → 抛错（T0.1）
expect_raise(
    "T0.1 seq_len!=num_stack",
    lambda: ActorCriticMamba(actor_num_input=1000, critic_num_input=183,
                             actor_num_output=13, seq_len=20, num_stack=10),
    "num_stack",
)

# 2. actor_num_input 不能被 seq_len 整除 → 抛错（T0.1）
expect_raise(
    "T0.1 actor_num_input%seq_len!=0",
    lambda: ActorCriticMamba(actor_num_input=1001, critic_num_input=183,
                             actor_num_output=13, seq_len=20, num_stack=20),
    "divisible",
)

# 3. 正常构造（seq_len==num_stack==20, headdim=32）→ 通过
expect_ok(
    "T0.1 normal construct",
    lambda: ActorCriticMamba(actor_num_input=1000, critic_num_input=183,
                             actor_num_output=13, seq_len=20, num_stack=20,
                             mamba_headdim=32),
)

# 4. mamba 键误入 ActorCriticMLP → 抛错（T0.1 静默回落防护）
expect_raise(
    "T0.1 mamba keys into MLP",
    lambda: ActorCriticMLP(actor_num_input=1000, critic_num_input=183,
                           actor_num_output=13, mamba_d_model=128),
    "class_name",
)

# 5. MLP 正常构造 → 通过
expect_ok(
    "MLP normal construct",
    lambda: ActorCriticMLP(actor_num_input=1000, critic_num_input=183,
                           actor_num_output=13),
)

# 6. headdim=64 + d_model=128 → nheads=4、D=548 → 抛错（T0.3-d，A1 修订后的真实约束）
#    GPU 实测该配置确实报 causal_conv1d channel-last stride 错误
expect_raise(
    "T0.3-d headdim=64 (D=548) rejected",
    lambda: Mamba2Encoder(d_input=50, d_model=128, d_state=16, d_conv=4,
                          expand=2, headdim=64, n_layers=1),
    "multiple of 8",
)

# 7. headdim=32（nheads=8，D=552）→ 通过
expect_ok(
    "T0.3-d headdim=32 ok",
    lambda: Mamba2Encoder(d_input=50, d_model=128, d_state=16, d_conv=4,
                          expand=2, headdim=32, n_layers=1),
)

# 7b. 架构消融用：d_model=96 / headdim=24（nheads=8，D=424）→ 通过
#     旧规则看 nheads=8 也通过；此例锁住「headdim=d_model*expand//8」这条可用配方
expect_ok(
    "T0.3-d ablation d_model=96 headdim=24 ok",
    lambda: Mamba2Encoder(d_input=50, d_model=96, d_state=16, d_conv=4,
                          expand=2, headdim=24, n_layers=1),
)

# 7c. nheads=8 但 d_state=2 → D=524 非 8 倍数 → 抛错
#     证明 nheads%8 只是默认 d_state=16 下的代理规则，不是充要条件
expect_raise(
    "T0.3-d nheads=8 but d_state=2 (D=524) rejected",
    lambda: Mamba2Encoder(d_input=50, d_model=128, d_state=2, d_conv=4,
                          expand=2, headdim=32, n_layers=1),
    "multiple of 8",
)

# 8. rmsnorm=False 门控分支不丢 z（T0.3-c）：构造 Jit 层并跑前向
from rsl_rl.modules.actor_critic_mamba import Mamba2LayerJit
enc = Mamba2Encoder(d_input=50, d_model=128, d_state=16, d_conv=4,
                    expand=2, headdim=32, n_layers=1)
layer = enc.mamba_layers[0]
layer.rmsnorm = False  # 模拟 rmsnorm=False 路径
jit_layer = Mamba2LayerJit(layer)


def run_rmsnorm_false():
    out = jit_layer(torch.randn(2, 5, 128))
    assert out.shape == (2, 5, 128), out.shape


expect_ok("T0.3-c rmsnorm=False forward", run_rmsnorm_false)

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
