import torch
import numpy as np
from typing import Union


def FT(array: Union[np.ndarray, list, torch.Tensor]) -> torch.Tensor:
    """
    Transform numpy array or list to tensor
    :param array:
    :return:
    """
    return torch.FloatTensor(array).to(torch.device("cuda:0"))


def rand_num(shape: torch.Tensor, device) -> torch.Tensor:
    """
    return array of [-1,1] random number
    :param shape:
    :return:
    """
    return 2 * torch.rand(shape[0], shape[1], device=device) - 1


def rand_num_like(array: torch.Tensor) -> torch.Tensor:
    """
    return array of [-1,1] random number
    :param array:
    :return:
    """
    return 2 * torch.rand_like(array) - 1


def yaw_transforming(x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate points (x, y) by yaw angle.

    Args:
        x (torch.Tensor): x coordinates
        y (torch.Tensor): y coordinates
        yaw (torch.Tensor): rotation angles (in radians)

    Returns:
        tuple of torch.Tensor: rotated x and y coordinates, each with shape (-1, 1)
    """
    x_new = x * torch.cos(yaw) - y * torch.sin(yaw)
    y_new = x * torch.sin(yaw) + y * torch.cos(yaw)
    return x_new.view(-1, 1), y_new.view(-1, 1)


def get_euler_angle(quat: torch.Tensor) -> torch.Tensor:
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
    angles = torch.where(angles < -torch.pi / 2, angles + torch.pi, angles)
    angles = torch.where(angles > torch.pi / 2, angles - torch.pi, angles)

    return angles


def euler_to_quaternion(euler_angles: torch.Tensor) -> torch.Tensor:
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
