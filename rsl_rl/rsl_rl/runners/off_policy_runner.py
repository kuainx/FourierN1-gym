import os
import sys
import time
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict

from rsl_rl.env import VecEnv

# Import FastTD3 components — add FastTD3 to path if needed
_FASTTD3_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "FastTD3")
_FASTTD3_ROOT = os.path.abspath(_FASTTD3_ROOT)
if _FASTTD3_ROOT not in sys.path:
    sys.path.insert(0, _FASTTD3_ROOT)

from fast_td3.fast_td3 import Actor, Critic
from fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
)

torch.set_float32_matmul_precision("high")


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    """Polyak averaging: tgt = (1-tau)*tgt + tau*src"""
    with torch.no_grad():
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)


class OffPolicyRunner:
    """Off-policy RL runner using FastTD3 algorithm.

    Implements the same interface as OnPolicyRunner so it can be dropped in
    via task_registry.make_alg_runner() by changing runner_class_name.

    Uses FastTD3's Actor (deterministic policy with learned exploration noise),
    Distributional Critic (Clipped Double Q-learning with distributional value),
    and SimpleReplayBuffer.

    Attributes:
        env (VecEnv): Isaac Gym vectorized environment.
        cfg (dict): Runner configuration.
        device (str): Device for computation.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg,
        log_dir=None,
        device="cpu",
    ):
        self.init(env, train_cfg, device)
        self.init_log(log_dir)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, env, train_cfg, device):
        print("----------------------------------")
        print("OffPolicyRunner (FastTD3)")

        self.cfg = train_cfg["runner"]
        self.fast_td3_cfg = train_cfg.get("fast_td3", {})
        self.log_interval = self.fast_td3_cfg.get("log_interval", 1)
        self.device = device
        self.env = env

        # Observation / action dimensions
        # Note: legged_gym may use frame stacking (use_stack + num_stack),
        # so actual obs dim = num_obs * num_stack when stacked
        n_obs = env.num_obs
        if hasattr(env.cfg, "env") and env.cfg.env.use_stack:
            n_obs *= env.cfg.env.num_stack
        n_act = env.num_actions
        n_critic_obs = env.num_pri_obs if env.num_pri_obs is not None else n_obs
        num_envs = env.num_envs

        print(f"  n_obs: {n_obs}, n_critic_obs: {n_critic_obs}, n_act: {n_act}")
        print(f"  num_envs: {num_envs}")
        print(f"  fast_td3_cfg: {self.fast_td3_cfg}")

        # --- Observation normalization ---
        if self.fast_td3_cfg.get("obs_normalization", True):
            self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=n_critic_obs, device=device
            )
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # --- Actor ---
        actor_kwargs = dict(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            init_scale=self.fast_td3_cfg.get("init_scale", 0.01),
            hidden_dim=self.fast_td3_cfg.get("actor_hidden_dim", 512),
            std_min=self.fast_td3_cfg.get("std_min", 0.001),
            std_max=self.fast_td3_cfg.get("std_max", 0.4),
            sim_type=self.fast_td3_cfg.get("sim_type", ""),
            sim_dimension=self.fast_td3_cfg.get("sim_dimension", 64),
            seq_len=self.fast_td3_cfg.get("actor_seq_len", 8),
            device=device,
        )
        self.actor = Actor(**actor_kwargs)

        # --- Critic ---
        critic_kwargs = dict(
            n_obs=n_critic_obs,
            n_act=n_act,
            num_atoms=self.fast_td3_cfg.get("num_atoms", 101),
            v_min=self.fast_td3_cfg.get("v_min", -250.0),
            v_max=self.fast_td3_cfg.get("v_max", 250.0),
            hidden_dim=self.fast_td3_cfg.get("critic_hidden_dim", 1024),
            sim_type=self.fast_td3_cfg.get("sim_type", ""),
            sim_dimension=self.fast_td3_cfg.get("sim_dimension", 64),
            seq_len=self.fast_td3_cfg.get("critic_seq_len", 8),
            device=device,
        )
        self.qnet = Critic(**critic_kwargs)
        self.qnet_target = Critic(**critic_kwargs)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # --- Optimizers ---
        weight_decay = self.fast_td3_cfg.get("weight_decay", 0.1)
        self.q_optimizer = optim.AdamW(
            self.qnet.parameters(),
            lr=torch.tensor(
                self.fast_td3_cfg.get("critic_learning_rate", 3e-4), device=device
            ),
            weight_decay=weight_decay,
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=torch.tensor(
                self.fast_td3_cfg.get("actor_learning_rate", 3e-4), device=device
            ),
            weight_decay=weight_decay,
        )

        # --- LR schedulers ---
        total_timesteps = self.fast_td3_cfg.get("total_timesteps", 150000)
        self.q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q_optimizer,
            T_max=total_timesteps,
            eta_min=torch.tensor(
                self.fast_td3_cfg.get("critic_learning_rate_end", 3e-4),
                device=device,
            ),
        )
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=total_timesteps,
            eta_min=torch.tensor(
                self.fast_td3_cfg.get("actor_learning_rate_end", 3e-4),
                device=device,
            ),
        )

        # --- Replay buffer ---
        self.asymmetric_obs = env.num_pri_obs is not None
        self.rb = SimpleReplayBuffer(
            n_env=num_envs,
            buffer_size=self.fast_td3_cfg.get("buffer_size", 1024 * 50),
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_critic_obs,
            asymmetric_obs=self.asymmetric_obs,
            playground_mode=False,
            n_steps=self.fast_td3_cfg.get("num_steps", 1),
            gamma=self.fast_td3_cfg.get("gamma", 0.99),
            device=device,
            cpu_buffer=self.fast_td3_cfg.get("cpu_buffer", False),
        )

        # --- AMP setup ---
        self.amp_enabled = (
            self.fast_td3_cfg.get("amp", True)
            and device != "cpu"
            and torch.cuda.is_available()
        )
        amp_dtype_str = self.fast_td3_cfg.get("amp_dtype", "bf16")
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
        self.scaler = GradScaler(
            enabled=self.amp_enabled and amp_dtype_str == "float16"
        )

        # --- Compile ---
        self._compile_enabled = self.fast_td3_cfg.get("compile", True)
        self._compiled = False

        # --- Algorithm hyperparams (cached for update loops) ---
        self.policy_noise = self.fast_td3_cfg.get("policy_noise", 0.001)
        self.noise_clip = self.fast_td3_cfg.get("noise_clip", 0.5)
        self.tau = self.fast_td3_cfg.get("tau", 0.1)
        self.gamma = self.fast_td3_cfg.get("gamma", 0.99)
        self.use_cdq = self.fast_td3_cfg.get("use_cdq", True)
        self.learning_starts = self.fast_td3_cfg.get("learning_starts", 10)
        self.num_updates = self.fast_td3_cfg.get("num_updates", 2)
        self.policy_frequency = self.fast_td3_cfg.get("policy_frequency", 2)
        self.disable_bootstrap = self.fast_td3_cfg.get("disable_bootstrap", False)
        self.use_grad_norm_clipping = self.fast_td3_cfg.get(
            "use_grad_norm_clipping", False
        )
        self.max_grad_norm = self.fast_td3_cfg.get("max_grad_norm", 0.0)

        # --- Track state ---
        self.action_low = -1.0
        self.action_high = 1.0

        print("OffPolicyRunner initialized.")

    def init_log(self, log_dir):
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop.

        Args:
            num_learning_iterations: Total environment steps (each = one policy step).
            init_at_random_ep_len: If True, randomize initial episode progress.
        """
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        # Compile update functions (first call only)
        if self._compile_enabled and not self._compiled:
            compile_mode = self.fast_td3_cfg.get("compile_mode", "reduce-overhead")
            self._update_main = torch.compile(
                self._update_main_impl, mode=compile_mode
            )
            self._update_pol = torch.compile(
                self._update_pol_impl, mode=compile_mode
            )
            self._compiled = True
        else:
            self._update_main = self._update_main_impl
            self._update_pol = self._update_pol_impl

        # --- Reset and get initial obs ---
        obs, pri_obs = self.env.reset()
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            obs = obs.to(self.device)
            pri_obs = pri_obs.to(self.device) if pri_obs is not None else None
        critic_obs = pri_obs if self.asymmetric_obs else obs

        # --- Episode tracking ---
        ep_infos = []
        rew_buffer = deque(maxlen=100)
        len_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        global_step = 0
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        prev_dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        for it in range(start_iter, tot_iter):
            start = time.time()

            # --- Record obs into buffer with previous step's next_obs ---
            # (On first iteration there's no previous step; we skip recording)

            # --- Rollout one step ---
            with torch.no_grad(), autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                norm_obs = self.obs_normalizer(obs)
                actions = self.actor.explore(norm_obs, dones=prev_dones)

            # Step the environment
            next_obs, next_pri_obs, rewards, dones, extras = self.env.step(
                actions.float()
            )
            if not (self.device == "cpu" or self.device.startswith("cuda")):
                next_obs = next_obs.to(self.device)
                next_pri_obs = (
                    next_pri_obs.to(self.device) if next_pri_obs is not None else None
                )
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
            next_critic_obs = next_pri_obs if self.asymmetric_obs else next_obs

            # Timeout info (legged_gym puts this in extras when send_timeouts=True)
            time_outs = extras.get("time_outs", torch.zeros_like(dones))

            # --- Store transition ---
            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(
                        actions, device=self.device, dtype=torch.float
                    ),
                    "next": {
                        "observations": next_obs,
                        "rewards": torch.as_tensor(
                            rewards, device=self.device, dtype=torch.float
                        ),
                        "truncations": time_outs.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(self.env.num_envs,),
                device=self.device,
            )
            if self.asymmetric_obs:
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = next_critic_obs
            self.rb.extend(transition)

            # --- Update obs pointers ---
            obs = next_obs
            critic_obs = next_critic_obs
            prev_dones = dones

            # --- Logging ---
            if self.log_dir is not None:
                if "episode" in extras:
                    ep_infos.append(extras["episode"])
                elif "log" in extras:
                    ep_infos.append(extras["log"])

                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rew_buffer.extend(
                    cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                )
                len_buffer.extend(
                    cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                )
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

            # --- Learning step ---
            logs_dict = {}
            if global_step > self.learning_starts:
                for i in range(self.num_updates):
                    batch_size = max(
                        1,
                        int(self.fast_td3_cfg.get("batch_size", 32768)
                        / self.env.num_envs),
                    )
                    data = self.rb.sample(batch_size)

                    # Normalize observations
                    data["observations"] = self.obs_normalizer(
                        data["observations"]
                    )
                    data["next"]["observations"] = self.obs_normalizer(
                        data["next"]["observations"]
                    )
                    if self.asymmetric_obs:
                        data["critic_observations"] = (
                            self.critic_obs_normalizer(
                                data["critic_observations"]
                            )
                        )
                        data["next"]["critic_observations"] = (
                            self.critic_obs_normalizer(
                                data["next"]["critic_observations"]
                            )
                        )

                    # Critic update
                    logs_dict = self._update_main(data, logs_dict)

                    # Actor update (delayed)
                    if self.num_updates > 1:
                        if i % self.policy_frequency == 1:
                            logs_dict = self._update_pol(data, logs_dict)
                    else:
                        if global_step % self.policy_frequency == 0:
                            logs_dict = self._update_pol(data, logs_dict)

                    # Soft-update target
                    _soft_update(self.qnet, self.qnet_target, self.tau)

            global_step += 1
            self.actor_scheduler.step()
            self.q_scheduler.step()

            stop = time.time()
            iteration_time = stop - start

            # --- Logging ---
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self._log(locals(), rew_buffer, len_buffer, ep_infos)
                ep_infos.clear()

            # --- Save ---
            save_interval = self.cfg.get("save_interval", 100)
            if it % save_interval == 0 and it > 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        # Final save
        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )

    # ------------------------------------------------------------------
    # Update logic (matches FastTD3 train.py)
    # ------------------------------------------------------------------

    def _update_main_impl(self, data, logs_dict):
        """Critic update: distributional Q-learning with Clipped Double Q."""
        with autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if self.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()

            if self.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                # Don't bootstrap for timeouts: legged_gym auto-resets env on
                # timeout, so next_obs has resampled commands (different context).
                # Bootstrapping with mismatched commands corrupts Q-values.
                bootstrap = (~dones).float()

            # Target policy smoothing
            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_state_actions = (
                self.actor(next_observations) + clipped_noise
            ).clamp(self.action_low, self.action_high)
            discount = self.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    self.qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = self.qnet_target.get_value(
                    qf1_next_target_projected
                )
                qf2_next_target_value = self.qnet_target.get_value(
                    qf2_next_target_projected
                )
                if self.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = (
                        qf_next_target_dist
                    )
                else:
                    qf1_next_target_dist = qf1_next_target_projected
                    qf2_next_target_dist = qf2_next_target_projected

            qf1, qf2 = self.qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * torch.nn.functional.log_softmax(qf1, dim=1),
                dim=1,
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * torch.nn.functional.log_softmax(qf2, dim=1),
                dim=1,
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(qf_loss).backward()
        self.scaler.unscale_(self.q_optimizer)

        if self.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.qnet.parameters(),
                max_norm=self.max_grad_norm
                if self.max_grad_norm > 0
                else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.q_optimizer)
        self.scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach().clone()
        logs_dict["qf_loss"] = qf_loss.detach().clone()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach().clone()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach().clone()
        return logs_dict

    def _update_pol_impl(self, data, logs_dict):
        """Actor update: maximize Q-value."""
        with autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            critic_observations = (
                data["critic_observations"]
                if self.asymmetric_obs
                else data["observations"]
            )

            qf1, qf2 = self.qnet(critic_observations, self.actor(data["observations"]))
            qf1_value = self.qnet.get_value(
                torch.nn.functional.softmax(qf1, dim=1)
            )
            qf2_value = self.qnet.get_value(
                torch.nn.functional.softmax(qf2, dim=1)
            )
            if self.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        if self.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.max_grad_norm
                if self.max_grad_norm > 0
                else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach().clone()
        logs_dict["actor_loss"] = actor_loss.detach().clone()
        return logs_dict

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, locs, rew_buffer, len_buffer, ep_infos):
        """Log metrics to tensorboard (every step) and console (every log_interval)."""
        self.tot_timesteps += self.env.num_envs
        self.tot_time += locs["iteration_time"]
        iteration_time = locs["iteration_time"]

        # Tensorboard (always)
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])

        logs_dict = locs["logs_dict"]
        if "qf_loss" in logs_dict:
            self.writer.add_scalar(
                "Loss/qf_loss", logs_dict["qf_loss"].mean(), locs["it"]
            )
        if "actor_loss" in logs_dict:
            self.writer.add_scalar(
                "Loss/actor_loss", logs_dict["actor_loss"].mean(), locs["it"]
            )
        if "qf_max" in logs_dict:
            self.writer.add_scalar(
                "Loss/qf_max", logs_dict["qf_max"].mean(), locs["it"]
            )
        if "qf_min" in logs_dict:
            self.writer.add_scalar(
                "Loss/qf_min", logs_dict["qf_min"].mean(), locs["it"]
            )

        self.writer.add_scalar(
            "Loss/critic_lr",
            self.q_scheduler.get_last_lr()[0],
            locs["it"],
        )
        self.writer.add_scalar(
            "Loss/actor_lr",
            self.actor_scheduler.get_last_lr()[0],
            locs["it"],
        )
        self.writer.add_scalar(
            "Perf/iteration_time", iteration_time, locs["it"]
        )

        if len(rew_buffer) > 0:
            self.writer.add_scalar(
                "Train/mean_reward",
                statistics.mean(rew_buffer),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(len_buffer),
                locs["it"],
            )

        # Console output (every log_interval iterations)
        it = locs["it"]
        if it % self.log_interval != 0 and it != locs["tot_iter"] - 1:
            return

        fps = int(self.env.num_envs / iteration_time) if iteration_time > 0 else 0
        width = 80
        pad = 35
        str_title = (
            f" \033[1m Learning iteration {it}/{locs['tot_iter']} \033[0m "
        )
        log_string = (
            f"{'#' * width}\n"
            f"{str_title.center(width, ' ')}\n\n"
            f"{'Computation (FPS):':>{pad}} {fps:.0f} steps/s ({iteration_time:.3f}s)\n"
        )

        if "qf_loss" in logs_dict:
            log_string += (
                f"{'Q-function loss:':>{pad}} {logs_dict['qf_loss'].mean().item():.4f}\n"
            )
        if "actor_loss" in logs_dict:
            log_string += (
                f"{'Actor loss:':>{pad}} {logs_dict['actor_loss'].mean().item():.4f}\n"
            )
        if len(rew_buffer) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} {statistics.mean(rew_buffer):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(len_buffer):.2f}\n"
            )

        eta_seconds = (
            self.tot_time
            / (it - locs["start_iter"] + 1)
            * (locs["num_learning_iterations"] - it)
        )
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"
        )
        print(log_string)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path, infos=None):
        """Save trained model to path."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "qnet_state_dict": self.qnet.state_dict(),
            "qnet_target_state_dict": self.qnet_target.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "obs_normalizer_state_dict": (
                self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict")
                else None
            ),
            "critic_obs_normalizer_state_dict": (
                self.critic_obs_normalizer.state_dict()
                if hasattr(self.critic_obs_normalizer, "state_dict")
                else None
            ),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(saved_dict, path)
        print(f"Saved model to {path}")

    def load(self, path, load_optimizer=True):
        """Load saved model from path."""
        loaded_dict = torch.load(path)

        # Handle noise_scales shape mismatch: training uses many envs,
        # inference may use fewer. noise_scales is only used for exploration,
        # so we can safely resize it to match the current actor.
        actor_state = loaded_dict["actor_state_dict"]
        if "noise_scales" in actor_state:
            current_shape = self.actor.noise_scales.shape
            loaded_shape = actor_state["noise_scales"].shape
            if current_shape != loaded_shape:
                # Drop stale noise_scales — only used for exploration, not inference.
                # Shape may differ in num_envs (train vs play) or n_act (per-env
                # vs per-joint noise). Re-inject the model's own values.
                actor_state["noise_scales"] = self.actor.noise_scales

        self.actor.load_state_dict(actor_state)
        self.qnet.load_state_dict(loaded_dict["qnet_state_dict"])
        self.qnet_target.load_state_dict(loaded_dict["qnet_target_state_dict"])

        if load_optimizer:
            self.q_optimizer.load_state_dict(loaded_dict["q_optimizer_state_dict"])
            self.actor_optimizer.load_state_dict(
                loaded_dict["actor_optimizer_state_dict"]
            )

        # Load normalizers if available
        for key in ["obs_normalizer_state_dict", "critic_obs_normalizer_state_dict"]:
            if key in loaded_dict and loaded_dict[key] is not None:
                normalizer = getattr(self, key.replace("_state_dict", ""), None)
                if normalizer is not None and hasattr(normalizer, "load_state_dict"):
                    normalizer.load_state_dict(loaded_dict[key])

        self.current_learning_iteration = loaded_dict.get("iter", 0)
        infos = loaded_dict.get("infos", None)
        print(f"Loaded model from {path}")
        return infos

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_inference_policy(self, device=None):
        """Return deterministic inference policy.

        Returns a callable that takes obs and returns actions without exploration noise.
        """
        self.actor.eval()
        if device is not None:
            self.actor.to(device)

        def _policy(obs):
            with torch.no_grad():
                norm_obs = self.obs_normalizer(obs)
                return self.actor(norm_obs)

        return _policy

    def train_mode(self):
        self.actor.train()
        self.qnet.train()

    def eval_mode(self):
        self.actor.eval()
        self.qnet.eval()
