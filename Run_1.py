from SF_TRON_FP.utils.Env.Tron_Env import Tron_Env
from SF_TRON_FP.utils.PPO.Actor_Critic import Actor_Critic
from SF_TRON_FP.utils.Config.Config import *

maximum_step = PPO_Config.PPOParam.maximum_step
episode = PPO_Config.PPOParam.episode
train = Env_Config.EnvParam.train
AC = Actor_Critic(PPO_Config, Env_Config)
if not train:
    AC.load_best_model()

env = Tron_Env(Env_Config, Robot_Config, PPO_Config)
import torch

env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    if epi % 5 ==0:
        env.resample_command()
        env.apply_disturbance()
    for step in range(maximum_step):
        """获取当前状态"""

        if not train:
            env.vel_cmd[:] = 1
            if epi>6:
                env.vel_cmd[:] = 0
                print("stop!!!!!!!")
        state = env.get_current_observations()
        state[:,33:] = 0  # basic state 之后就是地图信息，第一阶段机器人盲走

        """做动作"""
        action, scaled_action = AC.sample_action(state,deterministic=not train)

        """更新环境"""
        env.update_world(scaled_action=scaled_action)

        """获取下一个状态"""

        next_state = env.get_next_observations()
        next_state[:, 33:] = 0

        """计算奖励 判断是否结束"""

        reward, over, extra_over = env.compute_reward()


        """存储经验"""
        if train:
            AC.store_experience(state,
                                action,
                                next_state,
                                reward,
                                over,
                                step)

        """重置挂掉的机器人"""
        over += extra_over
        env.prim_initialization(torch.nonzero(over.flatten()).flatten())

    """每个回合结束后训练一次"""
    if train:
        AC.update()
        env.print_reward_sum()
