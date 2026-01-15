
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import mujoco
from dm_control import mjcf
from vpg.envs.mujoco_env import MJMocapBaseEnv
from vpg.assets import *
def angle_to_quat_z(angle_deg):
    rot = R.from_euler('z', angle_deg, degrees=True)
    x, y, z, w = rot.as_quat()  # scipy 输出: [x, y, z, w]
    return np.array([w, x, y, z])  # MuJoCo 需要: [w, x, y, z]

class UR5EPNPENV(MJMocapBaseEnv):
    def __init__(
        self,
        frame_skip:int = 5,
        render_mode:str = "rgbd_array",
        ) -> None:
        self._initial_qpos = np.array([
            1.3246831, -1.4734432, 1.45530837, -1.55266095, -1.57079637, -0.24611371
        ])
        self._initial_mocap_pose = np.array([
            0, 0.35, 0.313, 0, -1, 0, 0
        ])
        mjcf_root = self._build_mjcf_root()
        super().__init__(
            mjcf_root,
            frame_skip,
            render_mode,
        )
        # self.action_space = gym.spaces.MultiDiscrete([3,3,3,3],start=[-1,-1,-1,-1])
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.object_state = {
            "apple":False,
            "banana":False
        }
        self.object_lifted = 0 
        self.workspace_low = np.array([-0.28, 0.4, 0])
        self.workspace_high = np.array([0.28, 0.8, 0.3])
        self.incremental_unit = 0.02
        self.pixel_unit = 0.001
        self.handeye_renderer = mujoco.Renderer(self.model,224,224)
        self._open_distance = 0.150
        self._close_distance = 0.059
    def _build_mjcf_root(self):
        apple_mjcf = mjcf.from_path("vpg/assets/objects/apple.xml")
        banana_mjcf = mjcf.from_path("vpg/assets/objects/banana.xml")
        # lemon_mjcf = mjcf.from_path("vpg/assets/objects/lemon.xml")
        # peach_mjcf = mjcf.from_path("vpg/assets/objects/peach.xml")
        mjcf_root = mjcf.from_path("vpg/assets/scene/basic_scene_b.xml")
        self.robot = UR5E()
        self.gripper = AG95()
        self.robot.attach_tool(self.gripper.mjcf_root)
        mjcf_root.attach(self.robot.mjcf_root)
        self.banana_frame = mjcf_root.attach(banana_mjcf)
        self.banana_frame.pos = [0,0.6,0.01]
        self.apple_frame = mjcf_root.attach(apple_mjcf)
        self.apple_frame.pos = [0.12,0.52,0.01]
        # lemon_frame = mjcf_root.attach(lemon_mjcf)
        # lemon_frame.pos = [0.12,0.68,0.01]
        # peach_frame = mjcf_root.attach(peach_mjcf)
        # peach_frame.pos = [-0.12,0.68,0.01]
        self.object_handles = [self.apple_frame,self.banana_frame]
        return mjcf_root
    
    def reset_model(self):
        """
        Set the robot to the desired initial joint configuration.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        for i, jnt_id in enumerate(self.robot_jnt_id):
            self.data.qpos[jnt_id] = self._initial_qpos[i]
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = self._initial_mocap_pose[:3]
        self.data.mocap_quat[self.mocap_id] = self._initial_mocap_pose[3:]
        self.data.ctrl[-1] = 0

    def reset_arm(self):
        for i, jnt_id in enumerate(self.robot_jnt_id):
            self.data.qpos[jnt_id] = self._initial_qpos[i]
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = self._initial_mocap_pose[:3]
        self.data.mocap_quat[self.mocap_id] = self._initial_mocap_pose[3:]
        self.data.ctrl[-1] = 0


    def reset_banana(self):
        x = np.random.uniform(-0.20, 0.20)
        y = np.random.uniform(0.48, 0.72)
        z = 0.01
        pos = [x, y, z]
        angle_z = np.random.uniform(0, 45)  
        quat = np.array([np.cos(angle_z / 2), 0, 0, np.sin(angle_z / 2)])
        quat /= np.linalg.norm(quat)

        self._physics.bind(self.banana_frame).pos = pos
        self._physics.bind(self.banana_frame).quat = quat
        mujoco.mj_forward(self.model, self.data)

    def reset_objects(self):
        x_min, x_max = -0.25, 0.25
        y_min, y_max = 0.43, 0.77
        x_mid = 0.0
        y_mid = 0.6
        region_box1 = [
            (x_min, x_mid, y_min, y_max), #left
            (x_mid, x_max, y_min, y_max), #right
        ]
        np.random.shuffle(region_box1)

        for i,obj_handle in enumerate(self.object_handles):
            x_low,x_high,y_low,y_high = region_box1[i]
            x = np.random.uniform(x_low, x_high)
            y = np.random.uniform(y_low, y_high)
            z = 0.01  # 略高于桌面
            pos = [x, y, z]
            self._physics.bind(obj_handle).pos = pos

            angle_z = np.random.uniform(0, 10)  
            quat = np.array([np.cos(angle_z / 2), 0, 0, np.sin(angle_z / 2)])
            quat /= np.linalg.norm(quat)
            self._physics.bind(obj_handle).quat = quat

            mujoco.mj_forward(self.model, self.data)

    def get_mocap_pos(self):
        pos = self.data.mocap_pos[self.mocap_id]
        quat = self.data.mocap_quat[self.mocap_id]
        return pos,quat

    def set_mocap_pose(self,pos:np.ndarray,angle_deg:float=0):
        self.data.mocap_pos[self.mocap_id] = pos
        self.rotate_mocap(angle_deg)
        mujoco.mj_forward(self.model, self.data)

    def rotate_mocap(self, degree):
        """
        绕 Z 轴旋转 mocap body，**顺时针为正角度**，逆时针为负。
        """
        theta = np.deg2rad(degree)
        rotate_xmat = np.array([
            [ np.cos(theta),  np.sin(theta), 0],
            [-np.sin(theta),  np.cos(theta), 0],
            [              0,               0, 1]
        ])
        mocap_quat = self.data.mocap_quat[self.mocap_id].copy()  # 避免原地修改问题
        mocap_xmat = np.zeros(9)
        mujoco.mju_quat2Mat(mocap_xmat, mocap_quat)
        mocap_xmat = mocap_xmat.reshape(3, 3)
        new_xmat = (rotate_xmat @ mocap_xmat).flatten()
        new_quat = np.zeros(4)
        mujoco.mju_mat2Quat(new_quat, new_xmat)
        self.data.mocap_quat[self.mocap_id] = new_quat

    def step(self,action):
        # action = self._scale_action(action)
        # self.set_mocap_pose(action[:3])
        # self.data.ctrl[-1] = action[-1]
        self._step_simulation()

    def _check_grasp_success(self):
        # left_finger_id = self.model.body(self.gripper.left_finger_element.full_identifier).id
        # right_finger_id = self.model.body(self.gripper.right_finger_element.full_identifier).id
        # left_finger_pos = self.data.body(left_finger_id).xpos
        # right_finger_pos = self.data.body(right_finger_id).xpos
        # distance = np.linalg.norm(left_finger_pos - right_finger_pos)
        # distance = round(distance,3)
        # if self._close_distance < distance < self._open_distance:
        #     return True
        # else:
        #     False
        object_state = {
            'banana': False,
            'apple': False,
        }
        mujoco.mj_forward(self.model, self.data)
        banana_xpos = self._physics.named.data.xpos["banana/object"]
        apple_xpos = self._physics.named.data.xpos["apple/object"]
        if banana_xpos[-1] >= 0.1:
            object_state["banana"] = True
            self._physics.bind(self.banana_frame).pos = [0.6,0.1,-0.3]
            self.object_lifted += 1  
        elif apple_xpos[-1] >= 0.1:
            object_state["apple"] = True
            self._physics.bind(self.apple_frame).pos = [0.6,0.1,-0.3]
            self.object_lifted += 1 
        mujoco.mj_forward(self.model, self.data)
        if self.object_lifted == len(self.object_handles):
            return True,object_state
        else:
            return False,object_state

    def _scale_action(self, raw_action: np.ndarray) -> np.ndarray:
        scaled_action = raw_action[:4] * self.incremental_unit
        current_pos,_ = self.get_mocap_pos()
        next_pos = current_pos + scaled_action[:3]
        next_pos = np.clip(next_pos,self.workspace_low,self.workspace_high)
        current_gripper_pos = self.data.ctrl[-1]
        next_gripper_pos = np.array(current_gripper_pos+scaled_action[-1])
        next_gripper_pos = np.clip(next_gripper_pos,[0],[0.943])
        return np.concatenate([next_pos, next_gripper_pos])

    def grasp(self,pixel_x,pixel_y,angle_deg):
        pos_x = self.workspace_low[0] + (pixel_x * self.pixel_unit)
        pos_y = self.workspace_high[1] - (pixel_y * self.pixel_unit)
        pos_z = self.workspace_high[2]
        action = np.array([pos_x,pos_y,pos_z])
        self.set_mocap_pose(action,angle_deg)
        for i in range(30):
            self._step_simulation()
            self.render("pick_view")
        action[-1] = 0.0
        self.set_mocap_pose(action)
        for i in range(30):
            self._step_simulation()
            self.render("pick_view")
        self.data.ctrl[-1] = 0.943
        for i in range(30):
            self._step_simulation()
            self.render("pick_view")
        action[-1] = 0.3
        self.set_mocap_pose(action)
        for i in range(30):
            self._step_simulation()
            rgb_img, _, _ = self.render("ur5e/eyeinhand")
        terminated, grasp_state =  self._check_grasp_success()
        return terminated, grasp_state, rgb_img            


    def render(self, camera_name: str = None):
        valid_cameras = {
            'pick_view': self.pick_renderer,
            'place_view': self.place_renderer,
            'ur5e/eyeinhand': self.handeye_renderer
        }

        if camera_name not in valid_cameras:
            raise ValueError(f"camera_name must be one of {list(valid_cameras.keys())}")

        if self.render_mode == "human":
            self._render_simulation()

        renderer = valid_cameras[camera_name]
        return self._render_camera(renderer, camera_name)
    
    def _render_camera(self, renderer: mujoco.Renderer, camera_name: str):
        """
        使用指定的 renderer 渲染 RGB、Depth、Segmentation 三通道。
        """
        renderer.update_scene(self.data, camera_name)

        # Depth
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        # RGB
        rgb = renderer.render()
        # Segmentation
        renderer.enable_segmentation_rendering()
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()

        return rgb, depth, seg


    def reset(self):
        self.object_lifted = 0 
        self.reset_model()
        self.reset_objects()
        # self.reset_apple()

    @property
    def mocap_target_name(self):
        return "mocap"

    @property
    def tcp_site_name(self):
        return self.gripper.tcp_site_element.full_identifier
    
    @property
    def handeye_camera_name(self):
        return self.robot.camera_element.full_identifier

    @property
    def robot_body_id(self):
        return [self.model.body(self.robot.mjcf_root.model+"/"+body_name).id for body_name in self.robot.body_names]
    @property
    def robot_jnt_id(self):
        return [self.model.joint(self.robot.mjcf_root.model+"/"+joint_name).id for joint_name in self.robot.joint_names]
 
    @property
    def robot_nv(self):
        return len(self.robot_jnt_id)
    
    @property
    def gripper_actuator_id(self):
        return self.model.actuator(self.gripper.gpactuator_element.full_identifier).id



