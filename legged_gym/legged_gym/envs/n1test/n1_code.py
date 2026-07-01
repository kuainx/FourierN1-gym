from isaacgym.torch_utils import *

from legged_gym.envs.fftai.legged_robot_fftai_bipedal_code import LeggedRobotFFTAIBipedal
from legged_gym.envs.n1test.n1_config import N1Cfg
from legged_gym.utils.gym_math import get_euler_xyz

class N1(LeggedRobotFFTAIBipedal):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg: N1Cfg = cfg

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

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
    def _reward_feet_orientA(self):
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
        error_left = torch.abs(base_euler - left_foot_euler)
        error_right = torch.abs(base_euler - right_foot_euler)
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
        base_euler[:,1] = 0
        left_foot_euler = get_euler_xyz(left_foot_rot)
        right_foot_euler = get_euler_xyz(right_foot_rot)
        target_euler_left = torch.zeros_like(base_euler)
        target_euler_right = torch.zeros_like(base_euler)
        # target_euler_left[:,1] = 0.2 * self.swing_mask[:,0]
        # target_euler_right[:,1] = 0.2 * self.swing_mask[:,1]
        target_euler_left[:,1] = 0.2

        # print("base_euler",base_euler)
        print("left_foot_euler",left_foot_euler)
        print("right_foot_euler",right_foot_euler)
        error_left = torch.abs(base_euler - left_foot_euler - target_euler_left)
        error_right = torch.abs(base_euler - right_foot_euler - target_euler_right)

        # print("error_left",error_left)
        # print("error_right",error_right)
        # error_plane = error_left + error_right
        # reward_feet_plane = torch.exp(-3.0 * (error_plane[:,0] + error_plane[:,1]))
        reward_left_plane = 1. * torch.exp(-4.0 * error_left[:,1]) + 0.3 * torch.exp(-10.0 * error_left[:,0]) + 0.3 * torch.exp(-15.0 * error_left[:,2])
        reward_right_plane = 1. * torch.exp(-4.0 * error_right[:,1]) + 0.3 * torch.exp(-10.0 * error_right[:,0]) + 0.3 * torch.exp(-15.0 * error_left[:,2])
        return (reward_left_plane + reward_right_plane)/2

    def _reward_feet_orientB(self):
          """
          奖励足部方向一致
          """
          left_foot_rot = self.rigid_body_states[:, self.feet_indices[0]][:,3:7]
          right_foot_rot = self.rigid_body_states[:, self.feet_indices[1]][:,3:7]
          left_foot_rot_rev = quat_conjugate(left_foot_rot)
          right_foot_rot_rev = quat_conjugate(right_foot_rot)
          left_foot_direction = quat_rotate_inverse(left_foot_rot_rev,self.forward_vec)
          right_foot_direction = quat_rotate_inverse(right_foot_rot_rev,self.forward_vec)
          error_feet_rot = torch.abs(left_foot_direction - right_foot_direction)
          error_feet_rot = torch.sum(error_feet_rot[:,0:2], dim=1)  # dims 2->1

          reward_feet_rot = torch.exp(self.cfg.rewards.sigma_feet_orient * error_feet_rot)
          return reward_feet_rot
    def _reward_ref_action(self):
        cycle_time = 0.9
        phase = self.episode_length_buf * self.dt / cycle_time
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = 0.2
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        # sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        # self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        # self.ref_dof_pos[:, 5] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        # sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = -sin_pos_r * scale_1
        # self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        # self.ref_dof_pos[:, 11] = -sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos
        return 0