import sys
import os

# 自动获取当前文件的目录，然后找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 假设脚本在项目子目录中
sys.path.insert(0, project_root)
from SF_TRON_FP.utils.Env.Tron_Env import Tron_Env
from SF_TRON_FP.utils.PPO.Actor_Critic import Actor_Critic
from SF_TRON_FP.utils.Config.Config import *

maximum_step = PPO_Config.PPOParam.maximum_step
episode = PPO_Config.PPOParam.episode
time_per_epi = Env_Config.EnvParam.dt*maximum_step
train = Env_Config.EnvParam.train

Trained_AC = Actor_Critic(PPO_Config, Env_Config,index=0)
Trained_AC.load_best_model()
AC = Actor_Critic(PPO_Config, Env_Config,index=1)
if not train:
    AC.load_best_model()
env = Tron_Env(Env_Config, Robot_Config, PPO_Config)
import torch

env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    if epi % int(5/time_per_epi+1) ==0:
        env.resample_command()
        env.apply_disturbance()
    for step in range(maximum_step):
        """获取当前状态"""
        state = env.get_current_observations()

        state_no_camera = state.clone()
        state_no_camera[:,33:] = 0

        """做动作"""

        action1, scaled_action1 = Trained_AC.sample_action(state_no_camera,deterministic=True)
        action2, scaled_action2 = AC.sample_action(state,deterministic=not train)

        """更新环境"""
        env.update_world(scaled_action=scaled_action1*1+scaled_action2*1)

        """获取下一个状态"""

        next_state = env.get_next_observations()

        """计算奖励 判断是否结束"""

        reward, over, extra_over = env.compute_reward()

        """存储经验"""
        if train:
            AC.store_experience(state,
                                action2,
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
