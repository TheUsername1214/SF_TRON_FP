import torch

from SF_TRON_FP.utils.Useful_Function.Useful_Function import *


def yaw_transforming(x, y, yaw):
    x_new = x * torch.cos(yaw) - y * torch.sin(yaw)
    y_new = x * torch.sin(yaw) + y * torch.cos(yaw)
    return x_new.view(-1, 1), y_new.view(-1, 1)


class Tron_Env:
    def __init__(self, Env_Config, Robot_Config, PPO_Config):

        """初始化环境变量"""
        self.file_path = Env_Config.EnvParam.file_path
        self.friction_coef = Env_Config.EnvParam.friction_coef
        self.device = Env_Config.EnvParam.device
        self.backend = Env_Config.EnvParam.backend
        self.dt = Env_Config.EnvParam.dt
        self.sub_step = Env_Config.EnvParam.sub_step
        self.headless = Env_Config.EnvParam.headless
        self.train = Env_Config.EnvParam.train
        self.agents_num = (Env_Config.EnvParam.agents_num - Env_Config.EnvParam.agents_num_in_play) * self.train + \
                          Env_Config.EnvParam.agents_num_in_play

        """初始化机器人执行器参数"""
        self.actuator_num = Robot_Config.ActuatorParam.actuator_num
        self.ObsNoiseCfg = Robot_Config.ObservationNoiseCfg
        self.Kp = FT([Robot_Config.ActuatorParam.Kp] * self.agents_num)
        self.Kd = FT([Robot_Config.ActuatorParam.Kd] * self.agents_num)
        self.default_PD_angle_range = Robot_Config.ActuatorParam.default_PD_angle_range
        self.default_PD_angle = FT([Robot_Config.ActuatorParam.default_PD_angle] * self.agents_num)
        self.default_PD_angle += self.default_PD_angle_range*(2*torch.rand_like(self.default_PD_angle)-1)


        Kp_range = self.ObsNoiseCfg.Kp_range
        Kd_range = self.ObsNoiseCfg.Kd_range
        self.Kp = self.Kp * (1 + Kp_range * (2 * torch.rand_like(self.Kp) - 1))
        self.Kd = self.Kd * (1 + Kd_range * (2 * torch.rand_like(self.Kd) - 1))

        """导入Isaac Sim 库"""
        from ..Env.Software_Setup import App_Setup
        App_Setup(self.device, self.headless)

        """初始化机器人位姿参数"""
        self.initial_body_linear_vel_range = Robot_Config.InitialState.initial_body_linear_vel_range
        self.initial_body_angular_vel_range = Robot_Config.InitialState.initial_body_angular_vel_range
        self.initial_joint_pos_range = Robot_Config.InitialState.initial_joint_pos_range
        self.initial_joint_vel_range = Robot_Config.InitialState.initial_joint_vel_range
        self.initial_height = Robot_Config.InitialState.initial_height
        self.initial_euler_angle_range = Robot_Config.InitialState.initial_euler_angle_range
        self.frequency = Robot_Config.InitialState.frequency

        """初始化额外机器人参数"""
        self.vel_cmd = torch.zeros((self.agents_num, 1), device=self.device)  # 速度指令,0表示暂停，1表示前进
        self.target_ori = torch.zeros((self.agents_num, 3), device=self.device)
        self.time = torch.zeros((self.agents_num, 1), device=self.device)
        self.L_feet_air_time = torch.zeros((self.agents_num, 1), device=self.device)  # 左脚离地时间
        self.R_feet_air_time = torch.zeros((self.agents_num, 1), device=self.device)  # 右脚离地时间
        self.action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 动作
        self.prev_action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 上一次动作
        

        """奖励和"""
        self.max_step = PPO_Config.PPOParam.maximum_step
        self.vel_tracking_reward_sum = 0
        self.body_height_tracking_reward_sum = 0
        self.body_ori_tracking_reward_sum = 0
        self.foot_constraint_reward_sum = 0
        self.foot_step_length_reward_sum = 0
        self.effort_penalty_reward_sum = 0
        self.single_support_reward_sum = 0
        self.foot_air_time_reward_sum = 0
        self.Termination_reward_sum = 0

        import time as t
        self.t_module = t
        self.start_time = t.time()

        """初始化Isaac Sim环境"""
        from ..Env.Scene_Initialization import EnvSetup
        self.sim, self.scene = EnvSetup(self.file_path, self.dt, self.sub_step, self.agents_num, self.device,
                                        Robot_Config.DomainRandomizationCfg)

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
        initial_linear_vel = self.initial_body_linear_vel_range * (
                2 * torch.rand(num_agents, 3, device=self.device) - 1)
        initial_angular_vel = self.initial_body_angular_vel_range * (
                2 * torch.rand(num_agents, 3, device=self.device) - 1)
        initial_joint_pos = self.initial_joint_pos_range * (2 * torch.rand(num_agents, 8, device=self.device) - 1)
        initial_joint_vel = self.initial_joint_vel_range * (2 * torch.rand(num_agents, 8, device=self.device) - 1)
        initial_body_v_w = torch.concatenate((initial_linear_vel, initial_angular_vel), dim=1)

        initial_roll = self.initial_euler_angle_range[0]*(2 * torch.rand(num_agents, 1, device=self.device) - 1)
        initial_pitch = self.initial_euler_angle_range[1]*(2 * torch.rand(num_agents, 1, device=self.device) - 1)
        initial_yaw = self.initial_euler_angle_range[2]*(2 * torch.rand(num_agents, 1, device=self.device) - 1)
        initial_euler_angle = torch.concatenate((initial_roll,initial_pitch,initial_yaw),dim=-1)
        initial_quat = euler_to_quaternion(initial_euler_angle)
        

        # 设置速度命令和时间
        """重新初始化额外机器人参数"""
        self.vel_cmd[agent_index] = torch.rand((num_agents, 1), device=self.device)
        self.vel_cmd[agent_index] = (self.vel_cmd[agent_index] > 0.1).float()  # 90%概率前进，10%概率原地不动

        self.time[agent_index] = 0
        self.L_feet_air_time[agent_index] = 0
        self.R_feet_air_time[agent_index] = 0
        self.action[agent_index] = 0

        # 获取prim并设置身体速度
        # self.scene["robot"].reset(env_ids=agent_index.cpu().tolist())
        root_state = self.scene["robot"].data.default_root_state[agent_index].clone()
        root_state[:, :3] += self.scene.env_origins[agent_index]
        root_state[:, 2] += self.initial_height
        root_state[:,3:7] = initial_quat
        self.scene["robot"].write_root_pose_to_sim(root_state[:, :7], env_ids=agent_index)
        self.scene["robot"].write_root_velocity_to_sim(root_velocity=initial_body_v_w, env_ids=agent_index)
        self.scene["robot"].write_joint_state_to_sim(position=initial_joint_pos,
                                                     velocity=initial_joint_vel,
                                                     env_ids=agent_index)
        self.scene.write_data_to_sim()
        self.scene.update(dt=0)

    def resample_command(self,activate = True): # Only activate in walking, not stepping stone
        self.vel_cmd = torch.rand((self.agents_num, 1), device=self.device)
        self.vel_cmd = (self.vel_cmd > 0.3).float()  # 70%概率前进，30%概率原地不动
        if not activate:
            self.vel_cmd[:] = 1

    def apply_disturbance(self,activate = True):
        agent_index = torch.randperm(self.agents_num,device = self.device)[:int(0.3*self.agents_num)] # 20% push
        num_agents = len(agent_index)
        ang_vel_w = self.scene["robot"].data.root_com_ang_vel_w[agent_index]
        lin_vel_w = self.scene["robot"].data.root_com_lin_vel_w[agent_index]
        initial_linear_vel = self.initial_body_linear_vel_range * (
                2 * torch.rand(num_agents, 3, device=self.device) - 1)
        initial_angular_vel = self.initial_body_angular_vel_range * (
                2 * torch.rand(num_agents, 3, device=self.device) - 1)

        initial_linear_vel += lin_vel_w
        initial_angular_vel += ang_vel_w
        initial_body_v_w = torch.concatenate((initial_linear_vel, initial_angular_vel), dim=1)
        self.scene["robot"].write_root_velocity_to_sim(root_velocity=initial_body_v_w, env_ids=agent_index)

    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""

    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""

    """机器人状态更新"""

    def get_current_observations(self):
        # 获取机器人
        # #——————————————————————获取当前时刻状态————————————————————————————————##
        self.body_pos, self.body_ori = self.scene["imu_sensor"].data.pos_w, self.scene["imu_sensor"].data.quat_w
        self.body_ori = get_euler_angle(self.body_ori)
        self.angular_velocities = self.scene["imu_sensor"].data.ang_vel_b
        self.joint_pos = self.scene["robot"].data.joint_pos
        self.joint_vel = self.scene["robot"].data.joint_vel

        # 获取时间和时钟信号a
        clock_signal = 2 * torch.pi * self.time
        self.sine_clock = torch.sin(self.frequency*clock_signal)
        self.cosine_clock = torch.cos(self.frequency*clock_signal)

        # current_map = self.scene["height_scanner"].data.ray_hits_w[:, :, -1]
        # current_map = self.scene["Depth_Camera"].data.output['distance_to_image_plane'].reshape(self.agents_num, -1)
        current_map = torch.zeros((self.agents_num,11*18),device = self.device)
        # 拼接出下一时刻状态空间张量，并归一化
        current_noise = Add_ObsNoise(state=(self.joint_pos,
                                            self.joint_vel,
                                            self.body_ori,
                                            self.angular_velocities,
                                            current_map),
                                     ObsNoiseCfg=self.ObsNoiseCfg,
                                     device=self.device)
        # 添加噪声后的状态
        current_state = torch.concatenate(
            (self.joint_pos + current_noise[0],
             self.joint_vel*0.15 + current_noise[1],
             self.body_ori + current_noise[2],
             self.angular_velocities*0.15 + current_noise[3],
             self.sine_clock,
             self.cosine_clock,
             self.action,
             self.vel_cmd,
             current_map + current_noise[4]
             ), dim=1)

        # #——————————————————————获取当前时刻状态结束————————————————————————————————##

        # #——————————————————————获取额外机器人状态————————————————————————————————##
        # 获得机器人body高度 和 Abad 关节角度
        L_foot_pos, R_foot_pos = self.scene["L_imu_sensor"].data.pos_w, self.scene["R_imu_sensor"].data.pos_w


        self.L_foot_forward, self.L_foot_lateral = yaw_transforming(L_foot_pos[:, 0],
                                                                    L_foot_pos[:, 1],
                                                                    self.body_ori[:, 2])
        self.R_foot_forward, self.R_foot_lateral = yaw_transforming(R_foot_pos[:, 0],
                                                                    R_foot_pos[:, 1],
                                                                    self.body_ori[:, 2])

        L_foot_contact_force = self.scene["L_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        R_foot_contact_force = self.scene["R_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        self.L_foot_contact_situation = L_foot_contact_force > 1e-5
        self.R_foot_contact_situation = R_foot_contact_force > 1e-5
        # #——————————————————————获取额外机器人状态结束————————————————————————————————##
        return current_state

    def get_next_observations(self):

        # #——————————————————————获取下一时刻状态————————————————————————————————##
        # 获取机器人的关节身体信息
        self.next_body_pos, self.next_body_ori = self.scene["imu_sensor"].data.pos_w, self.scene[
            "imu_sensor"].data.quat_w
        self.next_body_ori = get_euler_angle(self.next_body_ori)
        self.next_angular_velocities = self.scene["imu_sensor"].data.ang_vel_b
        self.next_joint_pos = self.scene["robot"].data.joint_pos
        self.next_joint_vel = self.scene["robot"].data.joint_vel

        # 获取时间和时钟信号
        self.time += self.dt
        clock_signal = 2 * torch.pi * self.time
        self.sine_clock = torch.sin(self.frequency*clock_signal)
        self.cosine_clock = torch.cos(self.frequency*clock_signal)

        # next_map = self.scene["height_scanner"].data.ray_hits_w[:, :, -1]
        # next_map = self.scene["Depth_Camera"].data.output['distance_to_image_plane'].reshape(self.agents_num, -1)
        next_map = torch.zeros((self.agents_num,11*18),device = self.device)

        # 生成噪声
        next_noise = Add_ObsNoise(state=(self.next_joint_pos,
                                         self.next_joint_vel,
                                         self.next_body_ori,
                                         self.next_angular_velocities,
                                         next_map),
                                  ObsNoiseCfg=self.ObsNoiseCfg,
                                  device=self.device)

        # 拼接出下一时刻状态空间张量，并归一化
        next_state = torch.concatenate((self.next_joint_pos + next_noise[0],
                                        self.next_joint_vel*0.15 + next_noise[1],
                                        self.next_body_ori + +next_noise[2],
                                        self.next_angular_velocities*0.15 + next_noise[3],
                                        self.sine_clock,
                                        self.cosine_clock,
                                        self.action,
                                        self.vel_cmd,
                                        next_map + +next_noise[4]), dim=1)

        # #——————————————————————获取下一时刻状态结束————————————————————————————————##

        # #——————————————————————获取额外机器人状态————————————————————————————————##
        # 获得机器人body高度 和 Abad 关节角度
        self.next_body_height = self.next_body_pos[:, 2].view(-1, 1)
        L_foot_pos, R_foot_pos = (self.scene["L_imu_sensor"].data.pos_w,
                                  self.scene["R_imu_sensor"].data.pos_w)
        self.next_L_foot_angle, self.next_R_foot_angle = (get_euler_angle(self.scene["L_imu_sensor"].data.quat_w)[:,1],
                                get_euler_angle(self.scene["R_imu_sensor"].data.quat_w)[:,1])

        self.next_L_foot_forward, self.next_L_foot_lateral = yaw_transforming(L_foot_pos[:, 0],
                                                                              L_foot_pos[:, 1],
                                                                              self.next_body_ori[:, 2])
        self.next_R_foot_forward, self.next_R_foot_lateral = yaw_transforming(R_foot_pos[:, 0],
                                                                              R_foot_pos[:, 1],
                                                                              self.next_body_ori[:, 2])
        self.next_L_foot_z = L_foot_pos[:, 2].view(-1, 1)
        self.next_R_foot_z = R_foot_pos[:, 2].view(-1, 1)

        self.next_min_foot_z = torch.minimum(self.next_L_foot_z, self.next_R_foot_z)

        self.next_linear_vel = self.scene["robot"].data.root_lin_vel_w
        L_foot_contact_force = self.scene["L_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)
        R_foot_contact_force = self.scene["R_contact_sensor"].data.net_forces_w[:, 0, 2].view(-1, 1)

        self.next_L_foot_contact_situation = L_foot_contact_force > 1e-5
        self.next_R_foot_contact_situation = R_foot_contact_force > 1e-5
        self.L_feet_air_time += self.dt * (~self.next_L_foot_contact_situation)
        self.R_feet_air_time += self.dt * (~self.next_R_foot_contact_situation)
        # #——————————————————————获取额外机器人状态结束————————————————————————————————##
        return next_state

    """更新环境"""

    def update_world(self, scaled_action):
        """ 更新环境状态
        Args:
            action: 机器人动作
        """
        self.effort1 = (self.prev_action - scaled_action).abs().mean(dim=-1, keepdim=True)
        self.action = 1*scaled_action+0*self.prev_action
        self.prev_action = self.action.clone()

        for decimation in range(self.sub_step):
            joint_pos = self.scene["robot"].data.joint_pos
            joint_vel = self.scene["robot"].data.joint_vel
            torque = (self.action + self.default_PD_angle - joint_pos) * self.Kp - joint_vel * self.Kd
            torque[:, :6] = torque[:, :6].clip(-80, 80)
            torque[:, -2:] = torque[:, -2:].clip(-20, 20)
            self.scene["robot"].set_joint_effort_target(torque)
            self.scene.write_data_to_sim()
            if self.headless:
                render = False
            else:
                if decimation == self.sub_step - 1:
                    render = True
                else:
                    render = False
            self.sim.step(render=render)
            self.scene.update(self.dt / self.sub_step)

    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""

    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""

    """速度跟踪"""


    def vel_tracking_reward(self):

        vel_forward, vel_lateral = yaw_transforming(self.next_linear_vel[:, 0],
                                                    self.next_linear_vel[:, 1],
                                                    self.next_body_ori[:, 2])

        if_forward = (self.vel_cmd == 1).float()

        reward_vel_forward = - 1 * torch.abs(vel_forward - 0.5 * if_forward) + 0.5
        reward_vel_lateral = -0.5 * torch.abs(vel_lateral - 0)
        reward = reward_vel_lateral + reward_vel_forward

        return reward

    """高度跟踪"""

    def body_height_tracking_reward(self):

        below_min = torch.abs(self.next_body_height - 0.9)

        return -1 * below_min + 0.3

    """方位角跟踪（ZYX）"""

    def body_ori_tracking_reward(self):
        reward_ori_track = -0.2 * self.next_body_ori.norm(dim=1, keepdim=True)

        reward_ori_track += -0.1 * self.next_L_foot_angle.view(-1,1).norm(dim=1, keepdim=True)
        reward_ori_track += -0.1 * self.next_R_foot_angle.view(-1,1).norm(dim=1, keepdim=True)

        reward_ori_track += -0.1 * self.next_angular_velocities.norm(dim=1, keepdim=True)

        reward_ori_track += 0.2

        reward_ori_track *= 1

        return reward_ori_track

    """脚部限制"""

    def foot_constraint_reward(self):
        foot_regularization_reward = -0.2 * (
                torch.abs(self.next_L_foot_lateral - self.next_R_foot_lateral) < 0.1).float()
        foot_regularization_reward += -0.01 * (self.next_joint_pos[:, 0].view(-1,1) - 0.1).abs()
        foot_regularization_reward += -0.01 * (self.next_joint_pos[:, 1].view(-1,1) + 0.1).abs()

        return foot_regularization_reward

    """惩罚关节用力"""

    def effort_penalty_reward(self):
        joint_effort_reward = -0.25 * self.effort1
        return joint_effort_reward.view(-1, 1)



    """鼓励单脚着地"""

    def single_support_reward(self):
        single_support = self.next_L_foot_contact_situation != self.next_R_foot_contact_situation
        double_support = self.next_L_foot_contact_situation & self.next_R_foot_contact_situation
        flying = (~self.next_L_foot_contact_situation) & (~self.next_R_foot_contact_situation)

        walking_phase_reward = 0.2 * single_support.float()
        walking_phase_reward += -0.3 * flying.float()
        walking_phase_reward += -0.3 * double_support.float()

        walking_phase_reward *= (self.vel_cmd ==1).float() # only stepping when walking
        return walking_phase_reward.view(-1, 1)

    """鼓励脚悬空"""

    def foot_air_time_reward(self):
        L_touching_ground = self.next_L_foot_contact_situation & (~self.L_foot_contact_situation)
        R_touching_ground = self.next_R_foot_contact_situation & (~self.R_foot_contact_situation)

        foot_air_reward = ((self.L_feet_air_time - 1/self.frequency/2) * L_touching_ground +
                           (self.R_feet_air_time - 1/self.frequency/2) * R_touching_ground)
        too_long = (self.L_feet_air_time > 0.6) | (self.R_feet_air_time > 0.6)

        feet_air_time = 0.2 * foot_air_reward

        feet_air_time += -0.05 * too_long
        self.L_feet_air_time *= (~self.next_L_foot_contact_situation)
        self.R_feet_air_time *= (~self.next_R_foot_contact_situation)
        feet_air_time *= 8 *  (self.vel_cmd ==1).float()
        return feet_air_time.view(-1, 1)

    def foot_step_length(self):
        L_touching_ground = self.next_L_foot_contact_situation & (~self.L_foot_contact_situation)
        R_touching_ground = self.next_R_foot_contact_situation & (~self.R_foot_contact_situation)

        L_foot_step_reward = L_touching_ground * 5 * (self.next_L_foot_forward - self.next_R_foot_forward).clip(-0.5,
                                                                                                                0.5)
        R_foot_step_reward = R_touching_ground * 5 * (self.next_R_foot_forward - self.next_L_foot_forward).clip(-0.5,
                                                                                                                0.5)

        left_behind = L_touching_ground & (self.next_L_foot_forward < self.next_R_foot_forward)
        right_behind = R_touching_ground & (self.next_L_foot_forward > self.next_R_foot_forward)

        L_foot_step_reward += left_behind * (-0.1)
        R_foot_step_reward += right_behind * (-0.1)

        foot_step_reward = L_foot_step_reward + R_foot_step_reward
        foot_step_reward *= (self.vel_cmd == 1)
        return foot_step_reward

    """终止条件惩罚"""

    def Termination_reward(self):
        over1 = torch.abs(self.next_body_ori[:, 0].view(-1, 1)) > np.pi / 6
        over2 = torch.abs(self.next_body_ori[:, 1].view(-1, 1)) > np.pi / 6
        over3 = torch.abs(self.next_body_ori[:, 2].view(-1, 1)) > np.pi / 6
        over4 = (self.next_body_height - self.next_min_foot_z) < 0.6
        over5 = self.next_min_foot_z < -0.0
        self.over = over1 | over2 | over3 | over4 | over5
        reward_fall = - self.over.float() * 10
        return reward_fall.view(-1, 1)

    def compute_reward(self):
        reward = 0
        reward += 1 * self.vel_tracking_reward()
        reward += 1 * self.body_height_tracking_reward()
        reward += 1 * self.body_ori_tracking_reward()
        reward += 1 * self.foot_constraint_reward()
        reward += 1 * self.effort_penalty_reward()
        reward += 1 * self.single_support_reward()
        reward += 1 * self.foot_air_time_reward()
        reward += 1 * self.foot_step_length()
        reward += 1 * self.Termination_reward()
        reward += 0.2

        self.vel_tracking_reward_sum += self.vel_tracking_reward().mean().item() / self.max_step
        self.body_height_tracking_reward_sum += self.body_height_tracking_reward().mean().item() / self.max_step
        self.body_ori_tracking_reward_sum += self.body_ori_tracking_reward().mean().item() / self.max_step
        self.foot_constraint_reward_sum += self.foot_constraint_reward().mean().item() / self.max_step
        self.foot_step_length_reward_sum += self.foot_step_length().mean().item() / self.max_step

        self.effort_penalty_reward_sum += self.effort_penalty_reward().mean().item() / self.max_step
        self.single_support_reward_sum += self.single_support_reward().mean().item() / self.max_step
        self.foot_air_time_reward_sum += self.foot_air_time_reward().mean().item() / self.max_step
        self.Termination_reward_sum += self.Termination_reward().mean().item() / self.max_step

        return reward, self.over.float(), (self.time > 5).float()

    def print_reward_sum(self):
        print(f"vel_tracking_reward_sum: {self.vel_tracking_reward_sum:.4f}")
        print(f"body_height_tracking_reward_sum: {self.body_height_tracking_reward_sum:.4f}")
        print(f"body_ori_tracking_reward_sum: {self.body_ori_tracking_reward_sum:.4f}")
        print("")
        print(f"foot_constraint_reward_sum: {self.foot_constraint_reward_sum:.4f}")
        print(f"foot_step_length_reward_sum: {self.foot_step_length_reward_sum:.4f}")
        print("")
        print(f"effort_penalty_reward_sum: {self.effort_penalty_reward_sum:.4f}")
        print("")
        print(f"single_support_reward_sum: {self.single_support_reward_sum:.4f}")
        print(f"foot_air_time_reward_sum: {self.foot_air_time_reward_sum:.4f}")
        print("")
        print(f"Termination_reward_sum: {self.Termination_reward_sum:.4f}")
        print(f"simulation time:{self.t_module.time() - self.start_time:.2f} s")

        self.vel_tracking_reward_sum = 0
        self.body_height_tracking_reward_sum = 0
        self.body_ori_tracking_reward_sum = 0
        self.foot_constraint_reward_sum = 0
        self.foot_step_length_reward_sum = 0
        self.effort_penalty_reward_sum = 0
        self.single_support_reward_sum = 0
        self.foot_air_time_reward_sum = 0
        self.Termination_reward_sum = 0
