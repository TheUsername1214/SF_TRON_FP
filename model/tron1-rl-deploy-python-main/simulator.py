# Copyright information
#
# © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

import os
import sys
import time
import mujoco
import mujoco.viewer as viewer
from functools import partial
import onnxruntime as ort
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
import numpy as np

class SimulatorMujoco:
    def __init__(self, asset_path, joint_sensor_names, robot): 
        self.robot_type = os.getenv("ROBOT_TYPE")
        self.rl_type = os.getenv("RL_TYPE")
        model_dir = f'{os.path.dirname(os.path.abspath(__file__))}/controllers/model'
        self.model1 = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/model1.onnx'
        self.initialize_onnx_models()
        self.robot = robot
        self.joint_sensor_names = joint_sensor_names
        self.joint_num = len(joint_sensor_names)
        
        # Load the MuJoCo model and data from the specified XML asset path
        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        
        # Launch the MuJoCo viewer in passive mode with custom settings
        self.viewer = viewer.launch_passive(self.mujoco_model, self.mujoco_data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
        self.viewer.cam.distance = 10  # Set camera distance
        self.viewer.cam.elevation = -20  # Set camera elevation
    
        self.dt = self.mujoco_model.opt.timestep  # Get simulation timestepf
        self.fps = 1 / self.dt  # Calculate frames per second (FPS)
        self.sim_time = 0



        # Initialize robot command data with default values
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [80. for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [12 for x in range(0, self.joint_num)]
        self.robot_cmd.Kp[3] = 40
        self.robot_cmd.Kp[7] = 40
        self.robot_cmd.Kd[3] = 2
        self.robot_cmd.Kd[7] = 2




        # Initialize robot state data with default values
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]
        self.initial_joint_positions = [-0.02843711, 0.03055508, -0.00893693,  0.03103702 , 0.12365698, -0.11625697,
 -0.03181694 ,-0.036117 ]
        self.initial_joint_positions   = self.swap_positions(self.initial_joint_positions,reverse = True) # lab 2 gym


        self.initial_base_euler = [-0.0015 ,     -0.1 ,-0.08969947]  # [roll, pitch, yaw]
        self.default_PD_angle = np.array([0.15, -0.15,
                            0., -0.,
                            -0., 0.,
                            -0., -0.])

        self.default_PD_angle = self.swap_positions(self.default_PD_angle,reverse = True)

        # Initialize IMU data structure
        self.imu_data = datatypes.ImuData()

        # Set up callback for receiving robot commands in simulation mode
        self.robotCmdCallbackPartial = partial(self.robotCmdCallback)
        self.robot.subscribeRobotCmdForSim(self.robotCmdCallbackPartial)

        self.actions = np.zeros(8)
        self.last_actions = np.zeros(8)
        self.last_actions = self.swap_positions(self.last_actions,reverse = True)
        self.action_scale = np.array([0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5,0.5])
        self.cmd = np.zeros(0)
        self.depth_image = np.zeros(11*18)
        self.imu_data.quat[0] = 1
        self.imu_data.quat[1] = 0
        self.imu_data.quat[2] = 0
        self.imu_data.quat[3] = 0
        self.imu_data.gyro[0] = 0
        self.imu_data.gyro[1] = 0
        self.imu_data.gyro[2] = 0

    def initialize_onnx_models(self):
        # Configure ONNX Runtime session options to optimize CPU usage
        session_options = ort.SessionOptions()
        # Limit the number of threads used for parallel computation within individual operators
        session_options.intra_op_num_threads = 1
        # Limit the number of threads used for parallel execution of different operators
        session_options.inter_op_num_threads = 1
        # Enable all possible graph optimizations to improve inference performance
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable CPU memory arena to reduce memory fragmentation
        session_options.enable_cpu_mem_arena = False
        # Disable memory pattern optimization to have more control over memory allocation
        session_options.enable_mem_pattern = False

        # Define execution providers to use CPU only, ensuring no GPU inference
        cpu_providers = ['CPUExecutionProvider']

        # Load the ONNX model and set up input and output names
        self.policy_session1 = ort.InferenceSession(self.model1, sess_options=session_options,
                                                   providers=cpu_providers)
        self.policy_input_names1 = [self.policy_session1.get_inputs()[i].name for i in
                                   range(self.policy_session1.get_inputs().__len__())]
        self.policy_output_names1 = [self.policy_session1.get_outputs()[i].name for i in
                                    range(self.policy_session1.get_outputs().__len__())]
        self.policy_input_shapes1 = [self.policy_session1.get_inputs()[i].shape for i in
                                    range(self.policy_session1.get_inputs().__len__())]
        self.policy_output_shapes1 = [self.policy_session1.get_outputs()[i].shape for i in
                                     range(self.policy_session1.get_outputs().__len__())]
    def compute_observation(self):
        # Convert IMU orientation from quaternion to Euler angles (ZYX convention)
        imu_orientation = np.array(self.imu_data.quat)
        euler_angles = self.get_euler_angle(imu_orientation)
        print(euler_angles)

        # Retrieve base angular velocity from the IMU data
        base_ang_vel = np.array(self.imu_data.gyro)
        

        # Retrieve joint positions and velocities from the robot state
        joint_positions = np.array(self.robot_state.q)

        joint_velocities = np.array(self.robot_state.dq)

        # Retrieve the last actions that were applied to the robot
        last_actions = np.array(self.last_actions)

        # Populate observation vector
        joint_pos_input = joint_positions
        # swap positions in joint_pos, joint_vel and actions if mode is isaaclab

        joint_pos_input = self.swap_positions(joint_pos_input,reverse = False)
        joint_velocities = self.swap_positions(joint_velocities,reverse = False)
        last_actions = self.swap_positions(last_actions,reverse = False)  # gym order 2 lab order
        
        period = 1
        phase = self.sim_time%period/period
        sine_clock = np.sin(2*np.pi*phase).reshape(1)
        cosine_clock = np.cos(2*np.pi*phase).reshape(1)

        self.depth_image  = self.depth_image.flatten()
        self.cmd = np.array([0])


        if self.sim_time>3:
            self.cmd = np.array([1])

        else:
             self.cmd = np.array([0])
        obs = np.concatenate([
            joint_pos_input,  # Scaled joint positions
            joint_velocities*0.05,  # Scaled joint velocities
            euler_angles*0.25,  # Scaled base orientation (Euler angles)
            base_ang_vel*0.25,  # Scaled base angular velocity
            sine_clock,
            cosine_clock,
            last_actions,  # Lab order
            self.cmd,
            self.depth_image*0
        ])
        
        self.euler_angles = euler_angles
        self.observations = obs.copy()

    def compute_actions(self):
        """
        Computes the actions based on the current observations using the policy session.
        """
        # Concatenate observations into a single tensor and convert to float32
        input_tensor1 = self.observations.reshape(1, -1).copy()
        input_tensor1[0,33:] = 0.00


        input_tensor1 = input_tensor1.astype(np.float32)
        # Create a dictionary of inputs for the policy session
        inputs1 = {self.policy_input_names1[0]: input_tensor1}

        # Run the policy session and get the output
        output1,_ = self.policy_session1.run(self.policy_output_names1, inputs1)


        # Flatten the output and store it as actions
        self.actions1 = np.array(output1).flatten()  # Lab order


        self.actions = self.actions1.copy()*self.action_scale 

        self.actions = self.swap_positions(self.actions,reverse = True)
        self.last_actions = self.actions.copy()

        self.actions += self.default_PD_angle
        self.robot_cmd.q = self.actions


    # Callback function for receiving robot command data
    def robotCmdCallback(self, robot_cmd: datatypes.RobotCmd):
        self.robot_cmd = robot_cmd

    def swap_positions(self, initial_array, reverse=False):
        # Gym顺序: [左腿4关节, 右腿4关节]
        # Lab顺序: [左右abad, 左右hip, 左右knee, 左右ankle]
        
        # 索引映射
        gym_to_lab = [0, 4, 1, 5, 2, 6, 3, 7]  # Gym索引 -> Lab索引
        
        new_array = np.zeros_like(initial_array)
        
        if not reverse:
            # Gym -> Lab
            for i in range(8):
                new_array[i] = initial_array[gym_to_lab[i]]
        else:
            # Lab -> Gym: 需要逆映射
            for i in range(8):
                new_array[gym_to_lab[i]] = initial_array[i]
        
        return new_array

    # Callback for keypress events in the MuJoCo viewer (currently does nothing)
    def key_callback(self, keycode):
        pass

    def run(self):
        frame_count = 0
        self.rate = Rate(self.fps)  # Set the update rate according to FPS
        self.set_initial_state()

        for i in range(self.joint_num):
            self.robot_state.q[i] = self.mujoco_data.qpos[i + 7]
            self.robot_state.dq[i] = self.mujoco_data.qvel[i + 6]
        while self.viewer.is_running():    
            self.compute_observation()
            self.compute_actions()
            self.sim_time += 0.02
            print(self.sim_time)
            for _ in range(10):
                # Step the MuJoCo physics simulation
                for i in range(self.joint_num):
                    self.robot_state.q[i] = self.mujoco_data.qpos[i + 7]
                    self.robot_state.dq[i] = self.mujoco_data.qvel[i + 6]
                    self.robot_state.tau[i] = self.mujoco_data.ctrl[i]

                    # Apply control commands to the robot based on the received robot command data
                    self.mujoco_data.ctrl[i] = (
                        self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) + 
                        self.robot_cmd.Kd[i] * (0- self.robot_state.dq[i])
                    )


   

                    
                # Set the timestamp for the current robot state and publish it
                self.robot_state.stamp = time.time_ns()
                self.robot.publishRobotStateForSim(self.robot_state)
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)
                # Extract IMU data (orientation, gyro, and acceleration) from simulation
                imu_quat_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
                self.imu_data.quat[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 0]
                self.imu_data.quat[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 1]
                self.imu_data.quat[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 2]
                self.imu_data.quat[3] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 3]
                imu_gyro_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
                self.imu_data.gyro[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 0]
                self.imu_data.gyro[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 1]
                self.imu_data.gyro[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 2]

                if np.any(self.euler_angles>0.5):

                    mujoco.mj_resetData(self.mujoco_model, self.mujoco_data)
                    self.set_initial_state()
                # Set the timestamp for the current IMU data and publish it
                self.imu_data.stamp = time.time_ns()
                self.robot.publishImuDataForSim(self.imu_data)
                # Sync the viewer every 20 frames for smoother visualization
                if frame_count % 1 == 0:
                    self.viewer.sync()


  

                frame_count += 1
                #self.rate.sleep()  # Maintain the simulation loop at the correct rate
    def get_euler_angle(self, quat):  # I have checked it, it is correct
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
        Args:
            quat (np.Tensor): Tensor of shape (N, 4) representing quaternions
                                in the order (x,y,z,w).
        Returns:
            np.Tensor: Tensor of shape (N, 3) representing Euler angles
                        in radians in the order (roll, pitch, yaw).
        """
        quat = np.array(quat).reshape(4)
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]

        # Roll (x), Pitch (y), Yaw (z)
        # Using the ZYX convention XYZ Euler

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Use 1.0 - 1e-6 to avoid NaN when |sinp| is slightly > 1.0 due to floating point
        sinp = np.clip(sinp, -1.0 + 1e-6, 1.0 - 1e-6)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        angles = np.array([roll, pitch, yaw])

        # Angle adjustments to keep them within [-pi/2, pi/2]
        angles = np.where(angles < -np.pi / 2, angles + np.pi, angles)
        angles = np.where(angles > np.pi / 2, angles - np.pi, angles)
        
        return angles






    def set_initial_state(self):
        """设置初始状态：关节位置和机身姿态"""
        # 设置关节初始位置
        for i in range(self.joint_num):
            self.mujoco_data.qpos[i + 7] = self.initial_joint_positions[i]
            self.mujoco_data.qvel[i + 6] = 0  # 初始关节速度设为0
            
        # 将欧拉角转换为四元数
        roll, pitch, yaw = self.initial_base_euler
        
        # 使用ZYX顺序（先绕Z轴，再绕Y轴，最后绕X轴）
        # 注意：MuJoCo中通常使用wxyz顺序
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        # 四元数 (w, x, y, z)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        # 设置机身姿态（位置和方向）
        # MuJoCo中，前7个自由度是：前3个是位置(x,y,z)，后4个是四元数(w,x,y,z)
        self.mujoco_data.qpos[0:3] = [0, 0, 0.85]  # 初始位置，可以根据需要调整
        self.mujoco_data.qpos[3:7] = [qw, qx, qy, qz]  # 四元数顺序: w,x,y,z
        
        # 重置速度和加速度
        self.mujoco_data.qvel[:] = 0
        self.mujoco_data.qacc[:] = 0
        self.mujoco_data.ctrl[:] = 0



        # 更新MuJoCo内部状态
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        imu_quat_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
        self.imu_data.quat[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 0]
        self.imu_data.quat[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 1]
        self.imu_data.quat[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 2]
        self.imu_data.quat[3] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 3]





if __name__ == '__main__': 
    robot_type = os.getenv("ROBOT_TYPE")

    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    # Create a Robot instance of the PointFoot type
    robot = Robot(RobotType.PointFoot, True)

    # Default IP address for the robot
    robot_ip = "127.0.0.1"
    
    # Check if command-line argument is provided for robot IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    # Initialize the robot with the provided IP address
    if not robot.init(robot_ip):
        sys.exit()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the robot model XML file based on the robot type
    model_path = f'{script_dir}/robot-description/pointfoot/{robot_type}/xml/robot.xml'

    # Check if the model file exists, otherwise exit with an error
    if not os.path.exists(model_path):
        print(f"Error: The file {model_path} does not exist. Please ensure the ROBOT_TYPE is set correctly.")
        sys.exit(1)

    print(f"*** Model File Loaded: robot-description/pointfoot/{robot_type}/xml/robot.xml ***")

    # Define the names of the joint sensors used in the robot

    joint_sensor_names = [
        "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint"
    ]


    # Create and run the MuJoCo simulator instance
    simulator = SimulatorMujoco(model_path, joint_sensor_names, robot)
    simulator.run()
