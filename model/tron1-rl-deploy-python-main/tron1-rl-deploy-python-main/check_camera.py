import numpy as np
import matplotlib.pyplot as plt

VALUE_MIN = 0.0    # 设置最小显示值
VALUE_MAX = 1  # 设置最大显示值（根据你的数据调整）

while True:
    try:
        depth_image_raw = np.load("origin_array.npy")
        depth_image_f = np.load("filter_array.npy")
    except Exception as e:
        continue  # 如果加载失败，继续下一次循环
    
    # 创建1行2列的子图

    
    # 显示原始图像
    plt.subplot(1, 2, 1)  # 1行2列的第1个位置
    plt.imshow(depth_image_raw, vmin=VALUE_MIN, vmax=VALUE_MAX, cmap='viridis')
    plt.title("Original Depth Image")
    plt.colorbar()
    
    # 显示滤波后的图像
    plt.subplot(1, 2, 2)  # 1行2列的第2个位置
    plt.imshow(depth_image_f, vmin=VALUE_MIN, vmax=VALUE_MAX, cmap='viridis')
    plt.title("Filtered Depth Image")
    plt.colorbar()
    
    plt.suptitle("Depth Image Comparison", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()  # 清除当前图形