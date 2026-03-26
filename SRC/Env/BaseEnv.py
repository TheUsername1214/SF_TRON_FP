from SF_TRON_FP.SRC.Utils.Transformation import *


class BaseEnv:
    def __init__(self, EnvCfg, RobotCfg, PPOCfg):

        """初始化环境变量"""
        self.file_path = EnvCfg.EnvParam.file_path
        self.friction_coef = EnvCfg.EnvParam.friction_coef
        self.device = EnvCfg.EnvParam.device
        self.backend = EnvCfg.EnvParam.backend
        self.dt = EnvCfg.EnvParam.dt
        self.sub_step = EnvCfg.EnvParam.sub_step
        self.headless = EnvCfg.EnvParam.headless
        self.train = EnvCfg.EnvParam.train
        self.agents_num = (EnvCfg.EnvParam.agents_num - EnvCfg.EnvParam.agents_num_in_play) * self.train + \
                          EnvCfg.EnvParam.agents_num_in_play

        """导入Isaac Sim 库"""
        from ..Env.SoftwareSetup import App_Setup
        App_Setup(self.device, self.headless)
        from ..Env.SceneSetup import create_environment
        import time as t

        """初始化机器人执行器参数"""
        self.actuator_num = RobotCfg.ActuatorParam.actuator_num
        self.DomainRandomizationCfg = RobotCfg.DomainRandomizationCfg
        self.Kp = FT([RobotCfg.ActuatorParam.Kp] * self.agents_num)
        self.Kd = FT([RobotCfg.ActuatorParam.Kd] * self.agents_num)
        self.default_PD_angle = FT([RobotCfg.ActuatorParam.default_PD_angle] * self.agents_num)

        Kp_range = self.DomainRandomizationCfg.Kp_range
        Kd_range = self.DomainRandomizationCfg.Kd_range
        self.Kp = self.Kp * (1 + Kp_range * rand_num_like(self.Kp))
        self.Kd = self.Kd * (1 + Kd_range * rand_num_like(self.Kd))

        self.action_delay_range = self.DomainRandomizationCfg.action_delay_range
        self.external_body_force_range = self.DomainRandomizationCfg.external_body_force_range

        """初始化机器人位姿参数"""
        self.initial_body_linear_vel_range = RobotCfg.InitialState.initial_body_linear_vel_range
        self.initial_body_angular_vel_range = RobotCfg.InitialState.initial_body_angular_vel_range
        self.initial_joint_pos_range = RobotCfg.InitialState.initial_joint_pos_range
        self.initial_joint_vel_range = RobotCfg.InitialState.initial_joint_vel_range
        self.initial_height = RobotCfg.InitialState.initial_height
        self.initial_euler_angle_range = RobotCfg.InitialState.initial_euler_angle_range

        """初始化额外机器人参数"""
        self.vel_cmd = torch.zeros((self.agents_num, 1), device=self.device)  # 速度指令,0表示暂停，1表示前进
        self.target_ori = torch.zeros((self.agents_num, 3), device=self.device)
        self.time = torch.zeros((self.agents_num, 1), device=self.device)
        self.phase = torch.zeros((self.agents_num, 1), device=self.device)
        self.L_feet_air_time = torch.zeros((self.agents_num, 1), device=self.device)  # 左脚离地时间
        self.R_feet_air_time = torch.zeros((self.agents_num, 1), device=self.device)  # 右脚离地时间
        self.action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 动作
        self.prev_action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 上一次动作
        self.action_history = torch.zeros((self.agents_num, self.sub_step, self.actuator_num),
                                          device=self.device)  # 动作历史
        self.action_delay_idx = torch.randint(0, self.action_delay_range, (self.agents_num,),
                                              device=self.device)  # 延迟多少步
        self.external_body_force = torch.zeros((self.agents_num, 3),
                                               device=self.device)  # the dim 1 is necessary for isaac lab
        self.external_body_torques = torch.zeros((self.agents_num, 3), device=self.device)
        self.all_agent_indices = torch.arange(self.agents_num, device=self.device)

        self.reset_list = [self.time,
                           self.phase,
                           self.L_feet_air_time,
                           self.R_feet_air_time,
                           self.prev_action,
                           self.action,
                           self.action_history,
                           self.action_delay_idx]

        """奖励和"""
        self.max_step = PPOCfg.PPOParam.maximum_step
        self.vel_tracking_reward_sum = 0
        self.body_height_tracking_reward_sum = 0
        self.body_ori_tracking_reward_sum = 0
        self.foot_constraint_reward_sum = 0
        self.stand_still_reward_sum = 0
        self.effort_penalty_reward_sum = 0
        self.single_support_reward_sum = 0
        self.foot_air_time_reward_sum = 0
        self.Termination_reward_sum = 0
        self.t_module = t
        self.start_time = t.time()

        """初始化Isaac Sim环境"""
        self.sim, self.scene = create_environment(self.file_path,
                                                  self.dt,
                                                  self.sub_step,
                                                  self.agents_num,
                                                  self.device,
                                                  self.DomainRandomizationCfg)

    def prim_initialization(self, agent_index=None, reset_all=False):
        """
        :param reset_all:
        :param agent_index:  哪个序号的机器人挂了
        :return:
        重置指定序号机器人的位置和速度
        位置重置为初始位置，速度重置为随机小速度
        速度指令重置为随机值
        时间重置为0
        额外参数重置为0 比如power，action 等
        传感器数据会自动更新
        该函数在环境初始化和机器人挂掉时调用
        """
        if reset_all:
            agent_index = torch.arange(self.agents_num, device=self.device)

        num_agents = len(agent_index)
        if num_agents == 0:
            return

        # 生成随机初始化数据
        initial_linear_vel = self.initial_body_linear_vel_range * rand_num((num_agents, 3), device=self.device)
        initial_linear_vel[:, -1] = 0  # no push in z

        initial_angular_vel = self.initial_body_angular_vel_range * rand_num((num_agents, 3), device=self.device)

        initial_joint_pos = self.initial_joint_pos_range * rand_num((num_agents, 8), device=self.device)
        initial_joint_vel = self.initial_joint_vel_range * rand_num((num_agents, 8), device=self.device)
        initial_body_v_w = torch.concatenate((initial_linear_vel, initial_angular_vel), dim=1)

        initial_roll = self.initial_euler_angle_range[0] * rand_num((num_agents, 1), device=self.device)
        initial_pitch = self.initial_euler_angle_range[1] * rand_num((num_agents, 1), device=self.device)
        initial_yaw = self.initial_euler_angle_range[2] * rand_num((num_agents, 1), device=self.device)
        initial_euler_angle = torch.concatenate((initial_roll, initial_pitch, initial_yaw), dim=-1)
        initial_quat = euler_to_quaternion(initial_euler_angle)

        # 设置速度命令和时间
        """重新初始化额外机器人参数"""
        for i in range(len(self.reset_list)):
            self.reset_list[i][agent_index] = 0

        # 获取prim并设置身体速度
        # self.scene["robot"].reset(env_ids=agent_index.cpu().tolist())
        root_state = self.scene["robot"].data.default_root_state[agent_index].clone()
        root_state[:, :3] += self.scene.env_origins[agent_index]
        root_state[:, 2] += self.initial_height
        root_state[:, 3:7] = initial_quat
        self.scene["robot"].write_root_pose_to_sim(root_state[:, :7], env_ids=agent_index)
        self.scene["robot"].write_root_velocity_to_sim(root_velocity=initial_body_v_w, env_ids=agent_index)
        self.scene["robot"].write_joint_state_to_sim(position=initial_joint_pos,
                                                     velocity=initial_joint_vel,
                                                     env_ids=agent_index)
        self.scene.write_data_to_sim()
        self.scene.update(dt=0)

    def resample_command(self, activate=True):  # Only activate in walking, not stepping stone
        self.vel_cmd = torch.rand((self.agents_num, 1), device=self.device)
        self.vel_cmd = (self.vel_cmd > 0.3).float()  # 70%概率前进，30%概率原地不动
        if not activate:
            self.vel_cmd[:] = 1

    def apply_disturbance(self, activate=True):
        is_apply = torch.rand((self.agents_num, 1), device=self.device) > 0.8  # 给20%的人加外力
        self.external_body_force = rand_num((self.agents_num, 3), self.device) * is_apply.float()
        self.external_body_torques = rand_num((self.agents_num, 3), self.device) * is_apply.float()

        self.external_body_force[:, 0] *= self.external_body_force_range[0]
        self.external_body_force[:, 1] *= self.external_body_force_range[1]
        self.external_body_force[:, 2] *= self.external_body_force_range[2]

        external_body_force = rand_num((self.agents_num, 1, 3), self.device) * is_apply.float()
        external_body_torques = rand_num((self.agents_num, 1, 3), self.device) * is_apply.float()

        external_body_force[:, 0] = self.external_body_force
        external_body_torques[:] = 0  # 这么做是因为内存里张量维度和print出来的不一致

        self.scene["robot"].set_external_force_and_torque(external_body_force,
                                                          external_body_torques,
                                                          body_ids=[0] * self.agents_num,
                                                          is_global=True)

    def append_action_history(self, action):
        self.action_history[:, 1:, :] = self.action_history[:, :-1, :].clone()
        self.action_history[:, 0, :] = action.clone()
