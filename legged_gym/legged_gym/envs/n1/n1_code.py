from isaacgym.torch_utils import *
from legged_gym.utils.gym_math import get_euler_xyz

from legged_gym.envs.fftai.legged_robot_fftai_bipedal_code import LeggedRobotFFTAIBipedal
from legged_gym.envs.n1.n1_config import N1Cfg
from legged_gym.utils.math import wrap_to_pi


class N1(LeggedRobotFFTAIBipedal):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg: N1Cfg = cfg
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)

    def compute_observation_profile(self):

        obs_buf = torch.cat(
            (
                # command
                self.commands[:, 0:3] * self.commands_scale,

                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity * self.obs_scales.gravity,

                # dof related
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,

                # action related
                self.actions * self.obs_scales.action,
                self.swing_clock
            ), dim=-1)

        pri_obs_buf = torch.cat(
            (
                obs_buf,

                # base related
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_heights_offset * self.obs_scales.height_measurements,

                # foot related
                self.feet_contact,
                self.feet_height * self.obs_scales.height_measurements,
                self.avg_feet_speed_xyz[:, 0, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 0, 1:2] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 1:2] * self.obs_scales.lin_vel,

                # terrain related
                self.surround_heights_offset * self.obs_scales.height_measurements,
            ), dim=-1)

        self.obs_buf = obs_buf
        self.pri_obs_buf = pri_obs_buf

    def compute_obs_noise_scale_vec_profile(self):
        """
        Returns the noise scale vector for the observation vector.

        The noise scale vector is used to scale the noise vector to the same scale as the observation vector.

        Output:
        - noise_scale_vec: torch.Tensor
        """

        noise_vec = torch.zeros_like(self.obs_buf[0])

        # command
        start_index_of_commands = 0
        index_offset_of_commands = self.cfg.commands.num_commands
        noise_vec[start_index_of_commands + 0:
                  start_index_of_commands + self.cfg.commands.num_commands] = 0.  # commands x, y, yaw

        # base related
        start_index_of_base_related = start_index_of_commands + index_offset_of_commands
        index_offset_of_base_related = 6
        noise_vec[start_index_of_base_related + 0:
                  start_index_of_base_related + 3] = \
            self.noise_scales.ang_vel \
            * self.noise_level \
            * self.obs_scales.ang_vel  # base ang vel
        noise_vec[start_index_of_base_related + 3:
                  start_index_of_base_related + 6] = \
            self.noise_scales.gravity \
            * self.noise_level \
            * self.obs_scales.gravity  # base projected gravity

        # dof related
        start_index_of_dof_related = start_index_of_base_related + index_offset_of_base_related
        index_offset_of_dof_related = 2 * self.num_dofs
        noise_vec[start_index_of_dof_related + 0 * self.num_dofs:
                  start_index_of_dof_related + 1 * self.num_dofs] = \
            self.noise_scales.dof_pos \
            * self.noise_level \
            * self.obs_scales.dof_pos  # dof_pos_offset
        noise_vec[start_index_of_dof_related + 1 * self.num_dofs:
                  start_index_of_dof_related + 2 * self.num_dofs] = \
            self.noise_scales.dof_vel \
            * self.noise_level \
            * self.obs_scales.dof_vel  # dof_vel

        # action related
        start_index_of_action_related = start_index_of_dof_related + index_offset_of_dof_related
        index_offset_of_action_related = 1 * self.num_actions
        noise_vec[start_index_of_action_related + 0 * self.num_actions:
                  start_index_of_action_related + 1 * self.num_actions] = \
            self.noise_scales.action \
            * self.noise_level \
            * self.obs_scales.action  # actions

        return noise_vec

    # ----------------------------------------------

    def _resample_commands(self, env_ids=None, command_profile=None):
        super()._resample_commands(env_ids, command_profile)

        self.update_gait_generator_pattern()

    def set_commands(self, env_ids, commands):
        """
        Sets the commands for the specified environments.

        NOTE: should not be called in the training process!!!
        """
        self.commands[env_ids] = commands

        self._command_refinement(env_ids)
        self.update_gait_generator_pattern()

    def update_flags_of_stand_command(self):
        # situation 1: the norm of the command x, y lin_vel is less than 0.10
        flags_of_stand_command_s1 = torch.norm(self.commands[:, 0:2], dim=1) <= 0.10

        # situation 2: the value of the command yaw ang_vel is less than 0.10
        flags_of_stand_command_s2 = torch.abs(self.commands[:, 2]) <= 0.10

        # situation 1 and situation 2
        flags_of_stand_command = flags_of_stand_command_s1 * flags_of_stand_command_s2
        flags_of_stand_command = flags_of_stand_command.bool()

        return flags_of_stand_command

    def update_flags_of_walk_command(self):
        # situation 1: the norm of the command x, y lin_vel is greater than 0.10
        flags_of_walk_command_s1 = torch.norm(self.commands[:, 0:2], dim=1) > 0.10

        # situation 2: the value of the command yaw ang_vel is greater than 0.10
        flags_of_walk_command_s2 = torch.abs(self.commands[:, 2]) > 0.10

        # situation 1 and situation 2
        flags_of_walk_command = flags_of_walk_command_s1 + flags_of_walk_command_s2
        flags_of_walk_command = flags_of_walk_command.bool()

        return flags_of_walk_command

    def update_gait_generator_pattern(self):
        env_ids_of_off_command = torch.Tensor([]).int().to(self.device)
        env_ids_of_stand_command = torch.Tensor([]).int().to(self.device)
        env_ids_of_walk_command = torch.Tensor([]).int().to(self.device)

        if "stand" in self.cfg.commands.gait_patterns:
            env_ids_of_stand_command = torch.where(self.update_flags_of_stand_command())[0]

        if "walk" in self.cfg.commands.gait_patterns:
            env_ids_of_walk_command = torch.where(self.update_flags_of_walk_command())[0]

        # update the env_ids of different gait patterns
        self.env_ids_of_off_command = env_ids_of_off_command
        self.env_ids_of_stand_command = env_ids_of_stand_command
        self.env_ids_of_walk_command = env_ids_of_walk_command

    # ==========================================================================================================================
    # Reward functions
    def _reward_feet_orient(self):
        """
        奖励足部方向与躯干方向一致
        """
        base_link_rot = self.root_states[:, 3:7]
        left_foot_rot = self.rigid_body_states[:, self.feet_indices[0]][:,3:7]
        right_foot_rot = self.rigid_body_states[:, self.feet_indices[1]][:,3:7]
        base_euler = get_euler_xyz(base_link_rot)
        left_foot_euler = get_euler_xyz(left_foot_rot)
        right_foot_euler = get_euler_xyz(right_foot_rot)
        # print("base_euler",base_euler)
        # print("left_foot_euler",left_foot_euler)
        # print("right_foot_euler",right_foot_euler)
        # wrap 到 (-pi, pi]，避免 base/foot 欧拉角跨 ±pi 时虚假误差跳变
        error_left = torch.abs(wrap_to_pi(base_euler - left_foot_euler))
        error_right = torch.abs(wrap_to_pi(base_euler - right_foot_euler))
        # print("error_left",error_left)
        # print("error_right",error_right)
        reward_feet_rot = torch.exp(-1.0 * (error_left[:,2]+error_right[:,2]))
        return reward_feet_rot


    def _reward_feet_plane(self):
        """
        奖励足部平面与地面平行
        """
        base_link_rot = self.root_states[:, 3:7]
        left_foot_rot = self.rigid_body_states[:, self.feet_indices[0]][:,3:7]
        right_foot_rot = self.rigid_body_states[:, self.feet_indices[1]][:,3:7]
        base_euler = get_euler_xyz(base_link_rot)
        # 平地目标：roll/pitch 均以世界系水平面为参考（不再跟随躯干侧倾）
        base_euler[:,0] = 0  # roll 目标 = 0：脚掌与地面平行
        base_euler[:,1] = 0  # pitch 目标 = 0：世界系水平（下方 target 添加摆动上翘偏置）
        left_foot_euler = get_euler_xyz(left_foot_rot)
        right_foot_euler = get_euler_xyz(right_foot_rot)
        target_euler_left = torch.zeros_like(base_euler)
        target_euler_right = torch.zeros_like(base_euler)
        # 摆动相脚尖上翘目标：0.1 rad（原 0.2，上翘过大会让后跟成为最低点 → 抬脚瞬间拖地）
        target_euler_left[:,1] = 0.2 * self.swing_mask[:,0]
        target_euler_right[:,1] = 0.2 * self.swing_mask[:,1]

        # print("base_euler",base_euler)
        # print("left_foot_euler",left_foot_euler)
        # print("right_foot_euler",right_foot_euler)
        # wrap 到 (-pi, pi]，避免 base/foot 欧拉角跨 ±pi 时虚假误差跳变
        error_left = torch.abs(wrap_to_pi(base_euler - left_foot_euler - target_euler_left))
        error_right = torch.abs(wrap_to_pi(base_euler - right_foot_euler - target_euler_right))

        # print("error_left",error_left)
        # print("error_right",error_right)
        # error_plane = error_left + error_right
        # reward_feet_plane = torch.exp(-3.0 * (error_plane[:,0] + error_plane[:,1]))
        # roll/pitch 项均用线性饱和：exp(-10·err) 在 err>0.2 rad 时梯度消失，
        # 无法纠正 10~20° 的脚掌倾斜；0.3 rad 内线性递减，超出后保持恒定梯度推回
        reward_left_plane = 1. * (1.0 - torch.clamp(error_left[:,1] / 0.3, max=1.0)) + 0.5 * (1.0 - torch.clamp(error_left[:,0] / 0.3, max=1.0)) + 0.5 * torch.exp(-10.0 * error_left[:,2])
        reward_right_plane = 1. * (1.0 - torch.clamp(error_right[:,1] / 0.3, max=1.0)) + 0.5 * (1.0 - torch.clamp(error_right[:,0] / 0.3, max=1.0)) + 0.5 * torch.exp(-10.0 * error_right[:,2])
        return reward_left_plane + reward_right_plane


    def _reward_feet_swing_high(self):
        # 奖励高于target_height
        rew_pos = (self.cfg.rewards.target_feet_height - self.feet_height) * 100 # unit: mm
        rew_pos = torch.clip(rew_pos, -6, 100)
        rew_pos = torch.exp(-0.3 * rew_pos)
        rew_pos *= self.swing_mask
        rew_pos = torch.sum(rew_pos, dim=1)
        return rew_pos

    def _reward_feet_swing_low(self):
        # 惩罚低于target_height/2
        pen_pos = (self.cfg.rewards.target_feet_height/2 - self.feet_height) * 100 # unit: mm
        pen_pos = torch.clip(pen_pos, 0, 20)
        pen_pos = 1 - torch.exp(0.2 * pen_pos)
        pen_pos *= self.swing_mask
        pen_pos = torch.sum(pen_pos, dim=1)
        return pen_pos

    def _reward_feet_swing_too_high(self):
        # 惩罚高于target_height*2
        pen_pos_high = (self.feet_height - self.cfg.rewards.target_feet_height * 1.5) * 100 # unit: mm
        pen_pos_high = torch.clip(pen_pos_high, 0, 20)
        pen_pos_high = 1 - torch.exp(0.3 * pen_pos_high)
        pen_pos_high *= self.swing_mask
        pen_pos_high = torch.sum(pen_pos_high, dim=1)
        return pen_pos_high

    def _reward_feet_support_high(self):
        # 惩罚支撑脚高于target_height/2
        pen_pos_low = (self.feet_height - self.cfg.rewards.target_feet_height/2) * 100 # unit: mm
        pen_pos_low = torch.clip(pen_pos_low, 0, 20)
        pen_pos_low = 1 - torch.exp(0.3 * pen_pos_low)
        pen_pos_low *= self.stance_mask
        pen_pos_low = torch.sum(pen_pos_low, dim=1)
        return pen_pos_low

    def _reward_feet_support_low(self):
        # # 奖励支撑脚贴近地面
        rew_pos_low = (self.feet_height)*100 # unit: mm
        rew_pos_low = torch.clip(rew_pos_low, 0, 100)
        rew_pos_low = torch.exp(-0.2 * rew_pos_low)
        rew_pos_low *= self.stance_mask
        rew_pos_low = torch.sum(rew_pos_low, dim=1)
        return rew_pos_low

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        Uses a squared penalty to heavily punish large, brief spikes.
        """
        # 计算左右脚的接触力幅值 (num_envs, 2)
        forces_norm = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print(self.episode_length_buf)
        # print(forces_norm)
        # print(self.contact_forces_limit)

        # 超限部分：低于 0.75 * limit 的部分不计，超出的部分使用平方惩罚
        excess = (forces_norm - self.contact_forces_limit * 1.1).clamp(min=0,max=2000)
        penalty = torch.sum(excess, dim=1)  # 平方后对各脚求和

        steps = self.episode_length_buf.float()   # 确保为浮点，shape: (num_envs,)
        weight = torch.clamp((steps - 100) / 200.0, 0.0, 1.0)  # 计算权重，shape: (num_envs,)
        # print(penalty)
        # print(weight)

        return penalty * weight

    # ==========================================================================================================================
    # Self-imitation: stability scoring (early-standing self-imitation)
    # Pure physics quantities; does NOT depend on gait_patterns / stand command.
    def compute_stability_score(self):
        """
        Compute a per-env stability score in [0, 1] reflecting how close the robot is to
        stable standing: base height near target, small base roll/pitch, and both feet in contact.

        Returns:
            torch.Tensor: (num_envs,) float score.
        """
        # base height offset already computed in compute_observation_variables ([-1, 1])
        # only penalize deviation beyond tolerance
        err_h = (self.base_heights_offset - 0.0).abs()
        err_h = (err_h - self.cfg.rewards.base_height_offset_range_limit) * (err_h > self.cfg.rewards.base_height_offset_range_limit)
        r_h = torch.clamp(1.0 - err_h.squeeze(1) / 1.0, 0.0, 1.0)

        # base roll / pitch from projected gravity: [0,1,2] = world z in base frame
        # upright => base_projected_gravity ~ [0, 0, 1]
        g_proj = self.base_projected_gravity  # (num_envs, 3)
        # horizontal tilt magnitude: sqrt(gx^2 + gy^2), 0 when upright
        tilt = torch.norm(g_proj[:, :2], dim=-1)
        r_tilt = torch.clamp(1.0 - tilt / 0.4, 0.0, 1.0)

        # both feet in contact
        feet_contact = self.feet_contact  # (num_envs, 2) bool
        r_contact = (feet_contact[:, 0] & feet_contact[:, 1]).float()

        score = (
            self.cfg.self_imitation.score_w_h * r_h
            + self.cfg.self_imitation.score_w_tilt * r_tilt
            + self.cfg.self_imitation.score_w_contact * r_contact
        )
        # normalize weights to sum to 1
        total_w = (
            self.cfg.self_imitation.score_w_h
            + self.cfg.self_imitation.score_w_tilt
            + self.cfg.self_imitation.score_w_contact
        )
        return score / max(total_w, 1e-6)