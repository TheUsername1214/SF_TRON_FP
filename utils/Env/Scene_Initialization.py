import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, ImuCfg, patterns, RayCasterCameraCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfSteppingStonesTerrainCfg, HfRandomUniformTerrainCfg
from isaaclab.sim import DomeLightCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np
import torch


def EnvSetup(file_path, dt, sub_step, agents_num, device, DomainRandomizationCfg):
    mass_range = DomainRandomizationCfg.mass_range
    com_range = DomainRandomizationCfg.com_range
    inertia_range = DomainRandomizationCfg.inertia_range
    friction_range = DomainRandomizationCfg.friction_range
    restitution_range = DomainRandomizationCfg.restitution_range

    num_row = 10
    """生成地形高度图"""
    sub_terrains = {}
    terrian_number = 10
    holes_depth = np.linspace(-0.1,-0.3,terrian_number)
    for i in range(terrian_number):
        # sub_terrains[f"stepping_stone{i}"] = HfSteppingStonesTerrainCfg(
        #     proportion=1.0,
        #     border_width=0.1,
        #     holes_depth=holes_depth[i],  # 这个不能给高，不然碰撞计算要花很久
        #     stone_height_max=0,
        #     stone_width_range=(0.4, 0.6), # 0.5的宽度实际上是0.4m
        #     stone_distance_range=(0.1, 0.2),#0.2的间距实际上是0.3m ，0.1的间距就是0.2m
        #     platform_width=0.8,
        #     )

        sub_terrains[f"flat_plane{i}"] = HfRandomUniformTerrainCfg(
            proportion=1.0,
            border_width=0.1,
            noise_range=(-0.01, 0.01),
            noise_step=0.01,
            )
    print(f"Environment initialization: num_row of terrains: {num_row}x{num_row}")

    """生成地形配置文件"""
    gen_cfg = TerrainGeneratorCfg(
        num_rows=num_row,  # 太多了会出问题，比如碰撞体失效
        num_cols=num_row,
        size=(10, 3),
        color_scheme="none",
        sub_terrains=sub_terrains,
        curriculum=False,
        border_width=20,
    )

    """添加仿真环境的参与物体"""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        """Design the scene with sensors on the robot."""

        """环境中的光照设置"""

        light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=DomeLightCfg(
                intensity=750.0,
                color=(0.9, 0.9, 0.9),
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        """地形设置"""
        imp_cfg = TerrainImporterCfg(
            prim_path="/World/defaultGroundPlane",
            terrain_type="generator",
            terrain_generator=gen_cfg,
            debug_vis=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1,
                dynamic_friction=1
            )
        )

        """机器人设置"""
        robot = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=file_path,
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0)),
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
            collision_group=0,

        )

        """脚部传感器"""
        L_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_L_Link",
            update_period=0.0, history_length=1, debug_vis=False
        )

        R_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_R_Link",
            update_period=0.0, history_length=1, debug_vis=False
        )

        imu_sensor = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            update_period=0,
        )

        L_imu_sensor = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_L_Link",
            update_period=0,
        )

        R_imu_sensor = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_R_Link",
            update_period=0,
        )

        Depth_Camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Camera_Frame",
            update_period=0.1,
            mesh_prim_paths=["/World/defaultGroundPlane"],
            max_distance=6,
            depth_clipping_behavior="max",
            debug_vis=True,
            offset=RayCasterCameraCfg.OffsetCfg(convention="world"),
            pattern_cfg=patterns.PinholeCameraPatternCfg(width=int(11 * (45.55 / 26.6)),
                                                         height=11,
                                                         horizontal_aperture=45.55,
                                                         vertical_aperture=26.6)
        )

    """初始化仿真 世界"""
    # 启动Isaac Sim 软件， 必须放在导入Isaac sim 库之前。
    sim_cfg = sim_utils.SimulationCfg(dt=dt / sub_step, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # design scene
    scene_cfg = SceneCfg(num_envs=agents_num, env_spacing=0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    scene.update(dt=0)

    """ 域随机化"""
    rb_view = scene["robot"].root_physx_view
    indices = torch.arange(agents_num)
    """ 质量随机化"""
    origin_mass = rb_view.get_masses().clone()
    origin_mass += mass_range * origin_mass * (2 * torch.rand_like(origin_mass) - 1)
    rb_view.set_masses(origin_mass.clamp(0.001, 10000), indices)

    """ 质心随机化"""  # 注意，com不仅包含xyz，还包含四元数
    origin_com = rb_view.get_coms().clone()
    origin_com[..., :3] += com_range * origin_com[..., :3] * (2 * torch.rand_like(origin_com[..., :3]) - 1)
    rb_view.set_coms(origin_com, indices)

    """ 惯性张量随机化"""
    origin_inertia = rb_view.get_inertias().clone()
    origin_inertia += inertia_range * origin_inertia * (2 * torch.rand_like(origin_inertia[..., -1]).unsqueeze(-1)
                                                        - 1.0)

    rb_view.set_inertias(origin_inertia, indices)

    """ 摩擦力与弹力随机化"""
    origin_material = rb_view.get_material_properties().clone()

    origin_material[..., 0:2] += (friction_range * torch.rand_like(origin_material[..., 0:2])) # only add friction
    origin_material[..., 0:2].clamp_(0.5, 2.0)
    origin_material[..., 2] += (restitution_range * torch.rand_like(origin_material[..., 2]))
    origin_material[..., 2].clamp_(0.0, 1) # restitution max=1
    rb_view.set_material_properties(origin_material, indices)

    return sim, scene
