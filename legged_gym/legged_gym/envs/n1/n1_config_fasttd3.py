from legged_gym.envs.n1.n1_config import N1CfgPPO as N1BaseCfgPPO
from legged_gym.envs.n1.n1_config_main_body import N1MainBodyCfg


class N1MainBodyCfgFastTD3(N1MainBodyCfg):
    """
    N1 Main Body configuration using FastTD3 (off-policy) algorithm.

    Usage:
        Register this config as a separate task, then train with
        python train.py --task N1_FastTD3
    """
    runner_class_name = "OffPolicyRunner"
    seed = -1  # random seed

    class env(N1MainBodyCfg.env):
        num_envs = 2048  # 4096→1024, free VRAM for replay buffer and Isaac Gym sim

    class runner(N1BaseCfgPPO.runner):
        experiment_name = "N1_FastTD3"
        num_steps_per_env = 1  # off-policy: one step per env per iteration
        save_interval = 10000
        max_iterations = 2000001
        run_name = ""

    # FastTD3 algorithm parameters (aligned with FastTD3 examples)
    class fast_td3:
        # --- Core hyperparameters ---
        buffer_size = 2048              # replay buffer capacity per env (aligned with MTBenchArgs)
        batch_size = 32768              # total batch size (aligned with BaseArgs)
        num_steps = 1                   # n-step return (1 = standard)
        gamma = 0.99                    # discount factor (aligned with MuJoCoPlaygroundArgs)
        tau = 0.1                       # target network soft-update rate (aligned with BaseArgs)
        recent_ratio = 0.5              # sample from recent 50% of buffer (0.0 = uniform)

        # --- Critic ---
        critic_learning_rate = 3e-4
        critic_learning_rate_end = 3e-4
        critic_hidden_dim = 1024
        num_atoms = 101                 # number of distributional atoms (aligned with BaseArgs)
        v_min = -10.0                   # value support lower bound (aligned with MuJoCoPlaygroundArgs/IsaacLabArgs)
        v_max = 10.0                    # value support upper bound (aligned with MuJoCoPlaygroundArgs/IsaacLabArgs)
        policy_noise = 0.001            # target policy smoothing noise std (aligned with BaseArgs)
        noise_clip = 0.5                # target policy smoothing noise clip
        use_cdq = True                  # Clipped Double Q-learning

        # --- Actor ---
        actor_learning_rate = 3e-4
        actor_learning_rate_end = 3e-4
        actor_hidden_dim = 1024          # hidden dim (aligned with BaseArgs)
        init_scale = 0.01               # final layer weight initialization scale (aligned with BaseArgs)
        std_min = 0.001                 # min exploration noise (aligned with BaseArgs)
        std_max = 0.4                   # max exploration noise (aligned with BaseArgs)
        policy_frequency = 2            # delayed policy update

        # --- Training control ---
        total_timesteps = 150000        # total environment steps
        learning_starts = 10            # steps before first update (aligned with BaseArgs)
        num_updates = 2                 # UTD ratio (aligned with BaseArgs)
        log_interval = 100              # console log every N iterations
        disable_bootstrap = False       # disable bootstrap from terminal states

        # --- Optimization ---
        weight_decay = 0.1
        compile = False                # torch.compile (disable if CUDA graph errors)
        compile_mode = "reduce-overhead"
        amp = True                     # automatic mixed precision
        amp_dtype = "bf16"
        use_grad_norm_clipping = False
        max_grad_norm = 0.0
        cpu_buffer = True              # store replay buffer on CPU to save GPU memory

        # --- Observation normalization ---
        obs_normalization = True

        # --- SimNorm (disabled by default, set sim_type to enable) ---
        sim_type = ""                  # "" | "sim_actor" | "sim_critic" | "sim_both"
        sim_dimension = 64
        critic_seq_len = 8
        actor_seq_len = 8

        # --- Mirror loss (for symmetric learning) ---
        enable_mirror = True           # enable mirror loss
        mirror_coef = 0.75              # mirror loss coefficient

        # --- Mamba-2 Actor (set use_mamba=True to enable) ---
        use_mamba = False               # use Mamba-2 backbone for Actor
        mamba_d_model = 128            # Mamba hidden dimension
        mamba_d_state = 16             # SSM state dimension
        mamba_d_conv = 4               # convolution width
        mamba_expand = 2               # expansion factor
        mamba_headdim = 32             # head dimension (d_model * expand / headdim must be multiple of 8)
        mamba_n_layers = 2             # number of Mamba layers

    class rewards(N1MainBodyCfg.rewards):
        class scales(N1MainBodyCfg.rewards.scales):
            torso_flat_orient = 0.35
            base_flat_orient = 0.35
            action_diff = -5
            action_diff_diff = -0.75
            dof_acc = -0.3
            dof_tor = -0.15
            feet_orient = 0.5
            feet_plane = 0.5
            cmd_diff_base_lin_vel_x = 3
            cmd_diff_base_lin_vel_y = 2
