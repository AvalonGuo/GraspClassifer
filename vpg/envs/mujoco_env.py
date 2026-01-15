import cv2
from typing import Any, Dict, Optional, Tuple, Union


import time
import numpy as np

from dm_control import mjcf

import mujoco
import mujoco.viewer


class MJMocapBaseEnv:
    render_modes = ["human","rgbd_array"]
    def __init__(
        self,
        mjcf_root,
        frame_skip : int = 5,
        render_mode : str = None, 
        ) -> None:
        self.mjcf_root = mjcf_root
        self.frame_skip = frame_skip
        if render_mode is not None and render_mode not in self.render_modes:
            raise ValueError(
                f"Invalid render_mode '{render_mode}'. "
                f"Supported modes are: {self.render_modes}"
            )
        self.render_mode = render_mode
        self.passive_viewer = None
        self._initialize_simulation()
        self.pick_renderer = mujoco.Renderer(self.model,400,560)
        self.place_renderer = mujoco.Renderer(self.model,600,400)
        

    def _initialize_simulation(self):
        self._physics  = mjcf.Physics.from_mjcf_model(self.mjcf_root)
        self.model = self._physics.model.ptr
        self.data = self._physics.data.ptr
        self.model.vis.global_.offwidth = 1024
        self.model.vis.global_.offheight = 1024
        self.mocap_id = self.model.body(self.mocap_target_name).mocapid[0]
        self.tcp_site_id = self.model.site(self.tcp_site_name).id
        gravity_compensation = True 
        if gravity_compensation:
            self.model.body_gravcomp[self.robot_body_id] = 1.0
    
    def _step_simulation(self):
        ctrl = self.data.ctrl.copy()
        ctrl[:self.robot_nv] = self.resolve_joint_rate_ik()
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _render_simulation(self):
        if self.passive_viewer == None and self.render_mode == "human":
            self.passive_viewer =  mujoco.viewer.launch_passive(model=self.model,data=self.data,show_left_ui=True,show_right_ui=True)
        step_start = time.time()
        self.passive_viewer.sync()
        time_until_next_step = self.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def resolve_joint_rate_ik(self):
        # Pre-define params
        damping = 1e-4
        max_angvel = 30
        # Pre-allocate numpy arrays.
        jac = np.zeros((6, self.model.nv))
        diag = damping * np.eye(6)
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        # Position error.
        error_pos[:] = self.data.mocap_pos[self.mocap_id] - self.data.site(self.tcp_site_id).xpos
        # Orientation error.
        mujoco.mju_mat2Quat(site_quat, self.data.site(self.tcp_site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, self.data.mocap_quat[self.mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 2.0)
        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.tcp_site_id)

        # Solve system of equations: J @ dq = error.
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        # Scale down joint velocities if they exceed maximum.
        if max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, 1.0)

        # Set the control signal.
        q = q[:self.robot_nv*2]
        jnt_range_min = self.model.jnt_range.T[0][:self.robot_nv*2]
        jnt_range_max = self.model.jnt_range.T[1][:self.robot_nv*2]

        np.clip(q,jnt_range_min,jnt_range_max, out=q)
        q = q[self.robot_jnt_id]
        return q

    def close_viewer(self):
        if self.passive_viewer != None:
            self.passive_viewer.close()

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    @property
    def mocap_target_name(self):
        raise NotImplementedError

    @property
    def tcp_site_name(self):
        raise NotImplementedError
    
    @property
    def robot_body_id(self):
        raise NotImplementedError
    
    @property
    def robot_jnt_id(self):
        raise NotImplementedError
    @property
    def robot_nv(self):
        raise NotImplementedError