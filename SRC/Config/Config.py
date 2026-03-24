class EnvCfg:
    class EnvParam:  # 训练环境的参数
        agents_num = 4000
        agents_num_in_play = 10
        file_path = "Model/Robot_Model/SF_TRON1A.usd"  # abs path, not relative path
        dt = 0.02
        sub_step = 10
        friction_coef = 1
        device = 'cuda'
        backend = "torch"
        train = False
        headless = train  # True: no GUI, False: GUI

class RobotCfg:
    class ActuatorParam:  # 机T器人的参数
        Kp = [80, 80, 80, 80, 80, 80, 40, 40]
        Kd = [12, 12, 12, 12, 12, 12, 2, 2]  # Do not try to reduce Kd, because the action scale is not 0.25 but 1
        default_PD_angle = [0.15, -0.15,
                            0, 0,
                            0, 0,
                            0, 0]
        actuator_num = 8

    class InitialState:
        initial_height = 0.85
        initial_euler_angle_range = [0.1, 0.25, 0.1]
        initial_body_linear_vel_range = 0.2
        initial_body_angular_vel_range = 0.2
        initial_joint_pos_range = 0.1
        initial_joint_vel_range = 0
        initial_joint_angle = [0, 0,
                               -0, 0,
                               0, 0,
                               0, 0]

    class DomainRandomizationCfg:
        # relative
        inertia_range = 0.2  # angular inertia
        mass_range = 0.2  # unit in [Kg], only act on base mass
        com_range = 0.2  # unit in [m]
        # abs
        friction_range = 1
        restitution_range = 1
        Kp_range = 0.1
        Kd_range = 0.1



class PPOCfg:
    class CriticParam:  # Critic 神经网络 参数
        state_dim = 33 + 18 * 11  # 机器人本体与外部指令感知
        critic_layers_num = 256
        critic_lr = 1e-4
        critic_update_frequency = 300

    class ActorParam:  # Actor 神经网络 参数
        action_scale = [1, 1, 1, 1, 1, 1, 1, 1]
        std_scale = 0.5
        act_layers_num = 256
        actuator_num = RobotCfg.ActuatorParam.actuator_num
        actor_lr = 1e-4
        actor_update_frequency = 40

    class PPOParam:  # 强化学习 PPO算法 参数
        gamma = 0.99
        lam = 0.95
        epsilon = 0.2
        policy_smooth = 0.6
        maximum_step = 25
        episode = 3000
        entropy_coef = 0.05  # positive means std increase, else decrease
        batch_size = 20000

    class EstimatorParam:
        state_dim = 33
        history_length = 10
        output_dim = 2
        estimator_layers_num = 256
        estimator_lr = 1e-4
        estimator_update_frequency = 300
