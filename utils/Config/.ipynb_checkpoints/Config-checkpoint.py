class Env_Config:
    class EnvParam:  # 训练环境的参数
        agents_num = 4000
        agents_num_in_play = 10
        file_path = "model/Robot_Model/SF_TRON1A.usd"  # abs path, not relative path
        dt = 0.02
        sub_step = 4
        friction_coef = 1
        device = 'cuda'
        backend = "torch"
        headless = False  # True: no GUI, False: GUI
        train = headless


class Robot_Config:
    class ActuatorParam:  # 机器人的参数
        Kp = [45, 45, 45, 45, 45, 45, 30, 30]
        Kd = [3, 3, 3, 3, 3 , 3, 2, 2]  # Do not try to reduce Kd, because the action scale is not 0.25 but 1
        default_PD_angle = [0.1, -0.1,
                            0.15, -0.15,
                            -0.25, 0.25,
                            -0.0, -0.0]
        actuator_num = 8

    class InitialState:
        initial_height = 0.85
        initial_euler_angle_range = [0.1,0.2,0.1]
        initial_body_linear_vel_range = 0.3
        initial_body_angular_vel_range = 0.3
        initial_joint_pos_range = 0.2
        initial_joint_vel_range = 0.2
        initial_joint_angle = [0, 0,
                               -0, 0,
                               0, 0,
                               0, 0]

    class DomainRandomizationCfg:
        # relative
        mass_range = 0.1
        com_range = 0.1
        inertia_range = 0.1
        # abs
        friction_range = 1  
        restitution_range = 0.05 

    class ObservationNoiseCfg:
        # relative
        Kp_range = 0.1
        Kd_range = 0.1

        # abs noise
        joint_angle_noise = 0.02
        joint_angular_vel_noise = 0.1
        body_ori_noise = 0.05
        body_angular_vel_noise = 0.1
        depth_camera_noise = 0.2


class PPO_Config:
    class CriticParam:  # Critic 神经网络 参数
        state_dim = 33 + 18 * 11  # 机器人本体与外部指令感知
        critic_layers_num = 256
        critic_lr = 2e-4
        critic_update_frequency = 300

    class ActorParam:  # Actor 神经网络 参数
        action_scale = 1
        std_scale = 0.5
        act_layers_num = 256
        actuator_num = Robot_Config.ActuatorParam.actuator_num
        actor_lr = 2e-4
        actor_update_frequency = 100

    class PPOParam:  # 强化学习 PPO算法 参数
        gamma = 0.99
        lam = 0.95
        epsilon = 0.2
        maximum_step = 50
        episode = 3000
        entropy_coef = -0.01  # positive means std increase, else decrease
        batch_size = 20000
