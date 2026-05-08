import argparse
import os
import sys
from contextlib import nullcontext

import mujoco
import mujoco.viewer
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PACKAGE_SRC = os.path.join(PROJECT_ROOT, "src")
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from se3_impedance.geometry_impedance_control import geometry_impedance_control
from se3_impedance.paths import get_franka_model_path
from se3_impedance.se3_controller import SE3ImpedanceController

XML_PATH = str(get_franka_model_path("scene_se3.xml"))


EE_BODY_NAME = "link7"
NV_ARM = 7
VIEWER_SYNC_INTERVAL = 5
INITIAL_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def parse_args():
    parser = argparse.ArgumentParser(description="Franka SE(3) impedance test.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def transform_from_pose(position, quat):
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    transform = np.eye(4)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = position
    return transform


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    gic = geometry_impedance_control()
    controller = SE3ImpedanceController(gic)
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

    data.qpos[:NV_ARM] = INITIAL_Q
    mujoco.mj_forward(model, data)

    T_d = transform_from_pose(data.xpos[ee_body_id].copy(), data.xquat[ee_body_id].copy())
    V_d = np.zeros(6)
    V_d_dot = np.zeros(6)
    M_full = np.zeros((model.nv, model.nv))
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))

    print("\n开始运行 SE(3) 阻抗控制仿真...")
    print(f"MuJoCo viewer = {'off (--headless)' if args.headless else 'on'}。\n")

    viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)

    with viewer_context as viewer:
        step = 0
        while (args.max_steps <= 0 or step < args.max_steps) and (viewer is None or viewer.is_running()):
            step += 1

            q = data.qpos[:NV_ARM].copy()
            q_dot = data.qvel[:NV_ARM].copy()
            T = transform_from_pose(data.xpos[ee_body_id].copy(), data.xquat[ee_body_id].copy())
            R = T[:3, :3]

            mujoco.mj_jacBody(model, data, jac_p, jac_r, ee_body_id)
            J_world = np.vstack((jac_p[:, :NV_ARM], jac_r[:, :NV_ARM]))
            J = np.vstack((R.T @ J_world[:3, :], R.T @ J_world[3:, :]))
            V = J @ q_dot

            mujoco.mj_fullM(model, M_full, data.qM)
            M = M_full[:NV_ARM, :NV_ARM]
            qfrc_bias = data.qfrc_bias[:NV_ARM].copy()

            tau = controller.compute_torque(
                q=q,
                q_dot=q_dot,
                T=T,
                V=V,
                J=J,
                M=M,
                qfrc_bias=qfrc_bias,
                J_dot_q_dot=np.zeros(6),
                T_d=T_d,
                V_d=V_d,
                V_d_dot=V_d_dot,
                F_ext=np.zeros(6),
            )

            data.ctrl[:NV_ARM] = tau
            mujoco.mj_step(model, data)
            if viewer is not None and step % VIEWER_SYNC_INTERVAL == 0:
                viewer.sync()


if __name__ == "__main__":
    main()
