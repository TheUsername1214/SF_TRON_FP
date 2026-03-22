import torch


class EstimatorNetwork(torch.nn.Module):
    def __init__(self, state_dim, num_layers, output_dim):
        super(EstimatorNetwork, self).__init__()
        self.state_dim = state_dim
        self.num_layers = num_layers

        # 共享的主干网络
        self.fc1_x = torch.nn.Linear(self.state_dim, self.num_layers * 4)
        self.fc2_x = torch.nn.Linear(self.num_layers * 4, self.num_layers * 2)
        self.fc3_x = torch.nn.Linear(self.num_layers * 2, self.num_layers)
        self.fc4_x = torch.nn.Linear(self.num_layers, output_dim)

    def forward(self, input_):
        """处理输入，提取状态和地图特征"""
        x = input_
        # 通过主干网络
        x = torch.nn.functional.elu(self.fc1_x(x))
        x = torch.nn.functional.elu(self.fc2_x(x))
        x = torch.nn.functional.elu(self.fc3_x(x))
        x = torch.nn.functional.elu(self.fc4_x(x))

        return x


class Estimator:
    def __init__(self, PPOCfg, EnvCfg):
        self.state_dim = PPOCfg.CriticParam.state_dim
        self.output_dim = PPOCfg.EstimatorParam.output_dim
        self.agent_num = EnvCfg.EnvParam.agents_num
        self.device = EnvCfg.EnvParam.device
        self.max_step = PPOCfg.PPOParam.maximum_step
        self.batch_size = PPOCfg.PPOParam.batch_size
        self.history_length = PPOCfg.EstimatorParam.history_length
        self.estimator_layers_num = PPOCfg.EstimatorParam.estimator_layers_num
        self.estimator_lr = PPOCfg.EstimatorParam.estimator_lr
        self.estimator_update_frequency = PPOCfg.EstimatorParam.estimator_update_frequency

        self.state_buffer = torch.zeros((self.max_step, self.agent_num, self.state_dim * self.history_length),
                                        device=self.device)
        self.forward_state_buffer = torch.zeros((self.agent_num, self.state_dim * self.history_length),
                                        device=self.device)

        self.output_buffer = torch.zeros((self.max_step, self.agent_num, self.output_dim), device=self.device)

        self.Estimator = EstimatorNetwork(self.state_dim * self.history_length, self.estimator_layers_num,
                                          self.output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.Estimator.parameters(), lr=self.estimator_lr)
        self.loss_fn = torch.nn.MSELoss()

        self.idx = [torch.randperm(self.max_step * self.agent_num, device=self.device)[:self.batch_size]
                    for _ in range(self.estimator_update_frequency)]

    def store_new_state_and_output(self, state, output, step, over):
        self.state_buffer[step:, :, self.state_dim:] = over.float() * self.state_buffer[step:, :, :-self.state_dim]
        self.state_buffer[step:, :, :self.state_dim] = state

        self.forward_state_buffer[:,self.state_dim:] = over.float() * self.forward_state_buffer[:, :-self.state_dim]
        self.forward_state_buffer[:, :self.state_dim] = state
        self.output_buffer[step:, :, :] = output

    def update(self):
        state = self.state_buffer.view(-1, self.state_dim * self.history_length)
        output = self.output_buffer.view(-1, self.output_dim)

        # Estimator更新
        for i in range(self.estimator_update_frequency):
            idx = self.idx[i]
            state_batch, output_batch = state[idx], output[idx]
            estimated_output_batch = self.Estimator(state_batch)
            loss = self.loss_fn(output_batch, estimated_output_batch)
            # print(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    def reset(self):
        self.forward_state_buffer[:] = 0
        self.state_buffer[:] = 0
        self.output_buffer[:] = 0

    def estimate_output(self, historical_state):
        with torch.no_grad():
            return self.Estimator(historical_state)

    def get_estimate_output(self):
        with torch.no_grad():
            return self.Estimator(self.forward_state_buffer)

#
# class EnvCfg:
#     class EnvParam:  # 训练环境的参数
#         agents_num =2
#         agents_num_in_play = 10
#         file_path = "Model/Robot_Model/SF_TRON1A.usd"  # abs path, not relative path
#         dt = 0.02
#         sub_step = 10
#         friction_coef = 1
#         device = 'cuda'
#         backend = "torch"
#         train = False
#         headless = train  # True: no GUI, False: GUI
#
# class RobotCfg:
#     class ActuatorParam:  # 机T器人的参数
#         Kp = [80, 80, 80, 80, 80, 80, 40, 40]
#         Kd = [12, 12, 12, 12, 12, 12, 2, 2]  # Do not try to reduce Kd, because the action scale is not 0.25 but 1
#         default_PD_angle = [0.15, -0.15,
#                             0, 0,
#                             0, 0,
#                             0, 0]
#         actuator_num = 8
#
#     class InitialState:
#         initial_height = 0.85
#         initial_euler_angle_range = [0.1, 0.25, 0.1]
#         initial_body_linear_vel_range = 0.2
#         initial_body_angular_vel_range = 0.2
#         initial_joint_pos_range = 0.1
#         initial_joint_vel_range = 0
#         initial_joint_angle = [0, 0,
#                                -0, 0,
#                                0, 0,
#                                0, 0]
#
#     class DomainRandomizationCfg:
#         # relative
#         inertia_range = 0.2  # angular inertia
#         mass_range = 0.2  # unit in [Kg], only act on base mass
#         com_range = 0.2  # unit in [m]
#         # abs
#         friction_range = 1
#         restitution_range = 1
#         Kp_range = 0.1
#         Kd_range = 0.1
#
#
#
#
# class PPOCfg:
#     class CriticParam:  # Critic 神经网络 参数
#         state_dim = 3  # 机器人本体与外部指令感知
#         critic_layers_num = 256
#         critic_lr = 1e-4
#         critic_update_frequency = 300
#
#     class ActorParam:  # Actor 神经网络 参数
#         action_scale = [1, 1, 1, 1, 1, 1, 1, 1]
#         std_scale = 0.5
#         act_layers_num = 256
#         actuator_num = RobotCfg.ActuatorParam.actuator_num
#         actor_lr = 1e-4
#         actor_update_frequency = 40
#
#     class PPOParam:  # 强化学习 PPO算法 参数
#         gamma = 0.99
#         lam = 0.95
#         epsilon = 0.2
#         policy_smooth = 0.6
#         maximum_step = 3
#         episode = 3000
#         entropy_coef = 0.05  # positive means std increase, else decrease
#         batch_size = 20000
#
#     class EstimatorParam:
#         history_length = 3
#         output_dim = 3
#         estimator_layers_num = 256
#         estimator_lr = 1e-4
#         estimator_update_frequency = 300
#
#
# test = Estimator(PPOCfg, EnvCfg)
# state_list=[torch.ones(2,3) , 2*torch.ones(2,3), 3*torch.ones(2,3)]
# output_list = [torch.ones(2,3),3*torch.ones(2,3),6*torch.ones(2,3)]
# over = torch.ones(2,1)>0
# for i in range(3):
#     state = state_list[i].to("cuda")
#
#
#     output = output_list[i].to("cuda")
#
#     over = over.to("cuda")
#
#     test.store_new_state_and_output(state, output, i,over)
#
# print(test.state_buffer)
# print(test.output_buffer)
#
#
# test.update()
#
# historical_state = torch.zeros(1,9).to("cuda")
# historical_state[0,0:3] = 2
# historical_state[0,3:6] =1
# historical_state[0,6:] = 0
# print(test.estimate_output(historical_state))
#
# print(test.get_estimate_output())
# test.reset()
# print(test.get_estimate_output())
