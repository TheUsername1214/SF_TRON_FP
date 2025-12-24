import torch
import numpy as np
def FT(variable):
    return torch.FloatTensor(variable).to(torch.device("cuda:0"))

def get_euler_angle(quat): # I have checked it, it is correct
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
    Args:
        quat (torch.Tensor): Tensor of shape (N, 4) representing quaternions
                             in the order (w, x, y, z).
    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing Euler angles
                      in radians in the order (roll, pitch, yaw).
    """


    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    # Roll (x), Pitch (y), Yaw (z)
    # Using the ZYX convention XYZ Euler

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Use 1.0 - 1e-6 to avoid NaN when |sinp| is slightly > 1.0 due to floating point
    sinp = torch.clamp(sinp, -1.0 + 1e-6, 1.0 - 1e-6)
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    angles = torch.stack([roll, pitch, yaw], dim=1)

    # Angle adjustments to keep them within [-pi/2, pi/2]
    angles = torch.where(angles < -np.pi / 2, angles + np.pi, angles)
    angles = torch.where(angles > np.pi / 2, angles - np.pi, angles)

    return angles

def euler_to_quaternion(euler_angles):
    """Convert Euler angles (roll, pitch, yaw) in radians to quaternion.
    Args:
        euler_angles (torch.Tensor): Tensor of shape (N, 3) representing Euler angles
                                     in radians in the order (roll, pitch, yaw) (ZYX convention).
    Returns:
        torch.Tensor: Tensor of shape (N, 4) representing quaternions
                      in the order (w, x, y, z).
    """
    
    roll = euler_angles[:, 0]  # x-axis rotation
    pitch = euler_angles[:, 1]  # y-axis rotation  
    yaw = euler_angles[:, 2]  # z-axis rotation
    
    # Compute half angles
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    
    # Trigonometric functions
    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)
    
    # Compute quaternion components (ZYX convention: yaw * pitch * roll)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    # Normalize quaternion (optional but recommended)
    quat = torch.stack([w, x, y, z], dim=1)
    quat = quat / torch.norm(quat, dim=1, keepdim=True)
    
    return quat

    
def Add_ObsNoise(state,ObsNoiseCfg,device):

    joint_angle_noise = ObsNoiseCfg.joint_angle_noise * (2 * torch.rand_like(state[0]).to(device) - 1)
    joint_angular_vel_noise = ObsNoiseCfg.joint_angular_vel_noise * (2 * torch.rand_like(state[1]).to(device) - 1)
    body_ori_noise = ObsNoiseCfg.body_ori_noise * (2 * torch.rand_like(state[2]).to(device) - 1)
    body_angular_vel = ObsNoiseCfg.body_angular_vel_noise * (2 * torch.rand_like(state[3]).to(device) - 1)
    depth_camera_noise = ObsNoiseCfg.depth_camera_noise * (2 * torch.rand_like(state[4]).to(device) - 1)

    hole_mask = torch.rand_like(state[4])<0.01
    depth_camera_noise += -state[4]*hole_mask

    return (joint_angle_noise,
            joint_angular_vel_noise,
            body_ori_noise,
            body_angular_vel,
            depth_camera_noise)



def difference(target_value, value, sigma):
    """计算两个值的归一化差异"""
    return torch.sum(((target_value - value) / sigma) ** 2,dim=1,keepdim=True)


def abs_sum(target_value, value):
    """计算两个值的绝对差异之和"""
    return -torch.sum(torch.abs(target_value - value),dim=1,keepdim=True)


def exp_sum(target_value, value, sigma):
    """计算两个值的指数差异之和"""
    return torch.exp(-difference(target_value, value, sigma))

