import torch
import torch.nn as nn

from .ppo_mirror import PPOMirror


class PPOSelfImitate(PPOMirror):
    """
    PPO variant that adds a self-imitative (early-standing) benefit-cloning term.

    Mounted exactly like PPOMirror: an extra loss term returned from
    `calculate_other_loss`, which PPO.update() adds to the total loss per mini-batch.

    Imitation unit = trajectory (per paper):
      - during rollout we accumulate, per env, the (obs, action) frames of the
        current episode together with a cumulative stability score (from
        env.compute_stability_score(), physics-based, NOT depending on commands);
      - when an env terminates (done/timeout) its frames are pushed into a bounded
        high-score history buffer along with the shared trajectory score;
      - per update we sample top-k high-score frames, weight them by
            w = sim_coef * exp((score - G)/G) * w_iter(iter),
        with G = EMA of the historical best trajectory score (paper's R_max),
        and add a behavior-cloning term w * log pi(a|s).

    The w_iter schedule (soft or hard decay) makes the term strongest early and
    fade later, matching "only take effect in early training".

    The term is DISABLED when enable_self_imitate=False (zero added loss).
    """

    def __init__(
            self,
            actor_critic=None,
            # self-imitation config (explicitly parsed, not passed through **kwargs)
            enable_self_imitate=False,
            sim_coef=1.0,
            sim_top_k=512,
            sim_ema_decay=0.001,
            sim_start_iter=0,
            sim_end_iter=4000,
            sim_decay="soft",
            sim_history_capacity=4096,
            **kwargs,
    ):
        # Pass only non-sim kwargs to PPOMirror so nothing leaks through **kwargs
        mirror_kwargs = {
            k: kwargs.pop(k)
            for k in ["device", "mirror", "mirror_coef"]
            if k in kwargs
        }
        super().__init__(actor_critic=actor_critic, **mirror_kwargs, **kwargs)

        # ---- self-imitation config ----
        self.enable_self_imitate = enable_self_imitate
        self.sim_coef = float(sim_coef)
        self.sim_top_k = int(sim_top_k)
        self.sim_ema_decay = float(sim_ema_decay)
        self.sim_start_iter = int(sim_start_iter)
        self.sim_end_iter = int(sim_end_iter)
        self.sim_decay = sim_decay
        self.sim_history_capacity = int(sim_history_capacity)

        # ---- env reference for per-step stability scoring ----
        self.sim_env = None

        # ---- trajectory accumulation state ----
        self.num_envs_sim = 0
        self._sim_steps = 0
        self._sim_obs = None        # (num_envs, steps, obs_dim)
        self._sim_actions = None    # (num_envs, steps, act_dim)
        self._sim_score = None      # (num_envs,) cumulative stability score
        self._sim_reset_mask = None # (num_envs,) bool: envs to reset accum

        # ---- history buffer (bounded) ----
        self._hist_obs = []         # list of (n_frames, obs_dim)
        self._hist_actions = []
        self._hist_score = []       # list of (n_frames,) shared trajectory score
        self._hist_frames = 0

        # ---- EMA of historical best trajectory score (G) ----
        self._score_G = None

        # ---- logging ----
        self.sim_loss = torch.zeros(1, device=self.device)
        self.mean_sim_loss = 0.0
        self.sim_iter = 0
        self._mean_sim_count = 0

        if self.enable_self_imitate:
            print("PPOSelfImitate: self-imitation ENABLED")
            print(f"  sim_coef={self.sim_coef} top_k={self.sim_top_k} "
                  f"ema={self.sim_ema_decay} start={self.sim_start_iter} "
                  f"end={self.sim_end_iter} decay={self.sim_decay}")
        else:
            print("PPOSelfImitate: self-imitation DISABLED")

    def get_sim_iter(self):
        return self.sim_iter

    def set_sim_iter(self, it):
        self.sim_iter = int(it)

    # ------------------------------------------------------------------
    # Called by OnPolicyRunnerMirror during rollout
    # ------------------------------------------------------------------
    def reset_sim(self, num_envs, num_steps_per_env):
        self.num_envs_sim = num_envs
        self._max_sim_len = int(num_steps_per_env)
        self._sim_steps = 0
        # allocate single trajectory tracking buffers (num_envs, steps)
        obs_dim = getattr(self, "_sim_obs_dim", None)
        self._sim_obs = None
        self._sim_actions = None
        self._sim_score = torch.zeros(num_envs, device=self.device)
        self._sim_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def sim_append(self, obs, actions):
        """Record one rollout step's obs+actions for self-imitation."""
        if self._sim_obs is None:
            self._sim_obs_dim = obs.shape[-1]
            self._sim_act_dim = actions.shape[-1]
            self._sim_obs = torch.zeros(
                (obs.shape[0], self._max_sim_len, obs.shape[-1]), device=self.device
            )
            self._sim_actions = torch.zeros(
                (actions.shape[0], self._max_sim_len, actions.shape[-1]), device=self.device
            )
        if self._sim_steps >= self._max_sim_len:
            return
        self._sim_obs[:, self._sim_steps] = obs.detach()
        self._sim_actions[:, self._sim_steps] = actions.detach()
        self._sim_steps += 1

    def sim_accumulate_score(self, stability_score, dones):
        """Accumulate per-step stability score into per-env trajectory score."""
        if self._sim_score is None:
            return
        self._sim_score += stability_score.detach()
        # envs that just finished: push their frames to history and reset accum
        done_mask = (dones > 0).bool()
        if done_mask.any():
            self._finalize(done_mask)

    def sim_finalize_remaining(self):
        """At end of a rollout, finalize any still-running trajectories."""
        if self._sim_score is None:
            return
        still = ~self._sim_reset_mask
        if still.any():
            self._finalize(still)

    def clear_history(self):
        self._hist_obs = []
        self._hist_actions = []
        self._hist_score = []
        self._hist_frames = 0

    # internal: push frames of done envs into history buffer
    def _finalize(self, env_mask):
        if self._sim_score is None or self._sim_steps == 0:
            return
        env_mask = env_mask.bool()
        idx = env_mask.nonzero(as_tuple=False).squeeze(1)
        for e in idx.tolist():
            obs = self._sim_obs[e, : self._sim_steps]
            acts = self._sim_actions[e, : self._sim_steps]
            s = self._sim_score[e].detach()
            # only keep episodes that have at least a couple frames and a positive score
            if self._sim_steps >= 2 and s > 1e-3:
                self._hist_obs.append(obs.clone())
                self._hist_actions.append(acts.clone())
                self._hist_score.append(s.clone().expand(self._sim_steps))
                self._hist_frames += self._sim_steps
        # trim history to capacity (drop old episodes, keep recent)
        while self._hist_frames > self.sim_history_capacity and len(self._hist_obs) > 1:
            dropped = self._hist_obs.pop(0).shape[0]
            self._hist_actions.pop(0)
            self._hist_score.pop(0)
            self._hist_frames -= dropped
        # reset accum for done envs
        self._sim_score[env_mask] = 0.0
        self._sim_reset_mask[env_mask] = True

    # ------------------------------------------------------------------
    # The extra loss, mounted exactly like mirror
    # ------------------------------------------------------------------
    def calculate_other_loss(self, obs_batch, critic_obs_batch, actions_batch):
        loss = super().calculate_other_loss(obs_batch, critic_obs_batch, actions_batch)

        if not (self.enable_self_imitate and self.sim_coef > 0):
            return loss

        w_iter = self._schedule_weight(self.sim_iter)
        if w_iter <= 0:
            return loss

        obs_pool, act_pool, score_pool = self._sample_history()
        if obs_pool is None or obs_pool.shape[0] == 0:
            return loss

        # G = EMA of historical best trajectory score (paper's R_max)
        if self._score_G is None:
            self._score_G = score_pool.max().detach()
        else:
            best = score_pool.max()
            self._score_G = (1 - self.sim_ema_decay) * self._score_G + self.sim_ema_decay * best

        G = self._score_G.detach().clamp(min=1e-4)
        w = self.sim_coef * torch.exp((score_pool - G) / G).clamp(max=10.0)
        w = w * w_iter

        # behavior cloning on high-score (obs, action) pairs
        self.actor_critic.update_distribution(obs_pool)
        log_prob = self.actor_critic.get_actions_log_prob(act_pool)
        bc = -(w * log_prob).mean()

        self.sim_loss = bc.detach()
        return loss + bc

    def _sample_history(self):
        if self._hist_frames == 0:
            return None, None, None
        obs = torch.cat(self._hist_obs, dim=0)
        acts = torch.cat(self._hist_actions, dim=0)
        scores = torch.cat(self._hist_score, dim=0)
        k = min(self.sim_top_k, obs.shape[0])
        if k < obs.shape[0]:
            idx = torch.argsort(scores, descending=True)[:k]
            obs = obs[idx]
            acts = acts[idx]
            scores = scores[idx]
        return obs, acts, scores

    def _schedule_weight(self, it):
        if it < self.sim_start_iter:
            return 0.0
        if it >= self.sim_end_iter:
            return 1e-3 if self.sim_decay == "soft" else 0.0
        t = (it - self.sim_start_iter) / max(1, self.sim_end_iter - self.sim_start_iter)
        if self.sim_decay == "hard":
            return 1.0 - t
        # soft exponential decay
        return float(torch.exp(-3.0 * torch.tensor(t, device=self.device)).item())

    # ------------------------------------------------------------------
    # Logging (same pattern as PPOMirror)
    # ------------------------------------------------------------------
    def log_other_loss(self):
        super().log_other_loss()
        if self.enable_self_imitate:
            self.mean_sim_loss += self.sim_loss.item()
            self._mean_sim_count += 1

    def mean_other_loss(self):
        super().mean_other_loss()
        if self.enable_self_imitate:
            self.mean_sim_loss /= max(1, self._mean_sim_count)

    def update_sim(self):
        return self.mean_sim_loss

    # ------------------------------------------------------------------
    # Mirror-env linkage (get_mirror_observations/actions are on env)
    # ------------------------------------------------------------------
    def set_sim_env(self, env):
        """Called by the runner to attach the environment (used for stability score
        if not already provided through sim_accumulate_score)."""
        self.sim_env = env