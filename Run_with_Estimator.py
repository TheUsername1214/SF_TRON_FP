import sys
import os

# 自动获取当前文件的目录，然后找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 假设脚本在项目子目录中
sys.path.insert(0, project_root)
from SF_TRON_FP.SRC.Env.TronEnv import TronEnv
from SF_TRON_FP.SRC.PPO.Actor_Critic import Actor_Critic
from SF_TRON_FP.SRC.Config.Config import *
from SF_TRON_FP.SRC.Estimator.Estimator import *

maximum_step = PPOCfg.PPOParam.maximum_step
episode = PPOCfg.PPOParam.episode
time_per_epi = EnvCfg.EnvParam.dt * maximum_step
train = EnvCfg.EnvParam.train
PPO_3 = Actor_Critic(PPOCfg, EnvCfg, index=2)
Estimator_1 = Estimator(PPOCfg, EnvCfg, index=1)
if not train:
    PPO_3.load_best_model()
    Estimator_1.load_best_model()

Env = TronEnv(EnvCfg, RobotCfg, PPOCfg)
import torch

Env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    if epi % int(5 / time_per_epi + 1) == 0:
        Env.resample_command()
        Env.apply_disturbance()
    over = torch.tensor([0], device="cuda")
    for step in range(maximum_step):
        """获取当前状态"""
        state = Env.get_current_observations()
        state[:, 33:] = 0  # basic state 之后就是地图信息，第一阶段机器人盲走
        privilege_state = Env.get_privilege()
        Estimator_1.store_new_state_and_output(state[:, :33], privilege_state / 100, step, over)

        if step == maximum_step - 1:
            print("estimate_error:", (Estimator_1.get_estimate_output() * 100 - privilege_state).abs().mean())

        """做动作"""
        action, scaled_action = PPO_3.sample_action(state, deterministic=not train)

        """更新环境"""
        Env.update_world(scaled_action=scaled_action)

        """获取下一个状态"""

        next_state = Env.get_next_observations()
        next_state[:, 33:] = 0

        """计算奖励 判断是否结束"""

        reward, over, extra_over = Env.compute_reward()

        """存储经验"""
        if train:
            PPO_3.store_experience(state,
                                   action,
                                   next_state,
                                   reward,
                                   over,
                                   step)

        """重置挂掉的机器人"""
        over += extra_over
        Env.prim_initialization(torch.nonzero(over.flatten()).flatten())

    """每个回合结束后训练一次"""
    if train:
        PPO_3.update()
        Estimator_1.update()
        Env.print_reward_sum()
