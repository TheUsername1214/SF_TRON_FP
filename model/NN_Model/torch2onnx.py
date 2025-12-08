from SF_TRON_FP.utils.Config.Config import *
from SF_TRON_FP.utils.PPO.Actor_Critic import Actor_Critic

import torch.onnx
import onnx
state_dim = PPO_Config.CriticParam.state_dim
# 1. 加载或定义PyTorch模型
model1 = Actor_Critic(PPO_Config, Env_Config)
model1.actor.load_state_dict(torch.load("actor0.pth"))
# model2 = Actor_Critic(PPO_Config, Env_Config)
# model2.actor.load_state_dict(torch.load("actor1.pth"))
# 2. 创建示例输入（dummy input）
# 注意：需要与实际推理时的输入形状一致
dummy_input = torch.randn(1,state_dim).to("cuda")  # 示例：图像输入

# 3. 导出为ONNX
onnx_path1 = "model1.onnx"
# onnx_path2 = "model2.onnx"
torch.onnx.export(
    model1.actor,                   # PyTorch模型
    dummy_input,             # 模型输入
    onnx_path1,               # 保存路径
    export_params=True,      # 导出训练好的参数
    opset_version=11,        # ONNX算子集版本（常用11或13）
    do_constant_folding=True,# 优化常量
    input_names=['input'],   # 输入节点名称
    output_names=['output'], # 输出节点名称
    dynamic_axes={           # 动态轴（支持可变batch_size）
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# torch.onnx.export(
#     model2.actor,                   # PyTorch模型
#     dummy_input,             # 模型输入
#     onnx_path2,               # 保存路径
#     export_params=True,      # 导出训练好的参数
#     opset_version=11,        # ONNX算子集版本（常用11或13）
#     do_constant_folding=True,# 优化常量
#     input_names=['input'],   # 输入节点名称
#     output_names=['output'], # 输出节点名称
#     dynamic_axes={           # 动态轴（支持可变batch_size）
#         'input': {0: 'batch_size'},
#         'output': {0: 'batch_size'}
#     }
# )

print(f"模型已导出")