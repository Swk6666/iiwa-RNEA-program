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

from nric_torque_controller import NRICTorqueController

XML_PATH = str(get_franka_model_path("scene_se3.xml"))
NOMINAL_XML_PATH = str(get_franka_model_path("panda_nohand_torque.xml"))


EE_BODY_NAME = "link7"
SENSOR_SITE_NAME = "attachment_site"
NRIC_WRENCH_FRAME = "link7"
FORCE_SENSOR_NAME = "attachment_site_force"
TORQUE_SENSOR_NAME = "attachment_site_torque"
NV_ARM = 7
VIEWER_SYNC_INTERVAL = 5
ENABLE_NRIC = True
INITIAL_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def read_wrench_sensor(model, data, force_sensor_id, torque_sensor_id):
    force_adr = model.sensor_adr[force_sensor_id]
    force_dim = model.sensor_dim[force_sensor_id]
    torque_adr = model.sensor_adr[torque_sensor_id]
    torque_dim = model.sensor_dim[torque_sensor_id]
    force = data.sensordata[force_adr:force_adr + force_dim].copy()
    torque = data.sensordata[torque_adr:torque_adr + torque_dim].copy()
    # MuJoCo site force/torque sensors report the child-parent interaction wrench.
    # The Pinocchio nominal model expects the external wrench applied on the
    # end-effector by the environment, so we flip the sign here.
    return -np.concatenate([force, torque])


def transform_from_xmat(position, rotation_flat):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation_flat, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def transform_wrench_between_frames(wrench_source, source_transform, target_transform):
    source_transform = np.asarray(source_transform, dtype=float)
    target_transform = np.asarray(target_transform, dtype=float)
    wrench_source = np.asarray(wrench_source, dtype=float)

    target_from_source = np.linalg.inv(target_transform) @ source_transform
    rotation = target_from_source[:3, :3]
    translation = target_from_source[:3, 3]

    force_target = rotation @ wrench_source[:3]
    torque_target = rotation @ wrench_source[3:] + np.cross(translation, force_target)
    return np.concatenate([force_target, torque_target])


def parse_args():
    parser = argparse.ArgumentParser(description="Franka SE(3) impedance test with optional NRIC.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--joint7-frictionloss",
        type=float,
        default=0.1,
        help="MuJoCo dry frictionloss applied to joint7.",
    )
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
    joint7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    if joint7_id == -1:
        raise ValueError("未找到 joint7，无法设置 frictionloss")
    model.dof_frictionloss[model.jnt_dofadr[joint7_id]] = args.joint7_frictionloss
    gic = geometry_impedance_control()
    controller = SE3ImpedanceController(gic)
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    sensor_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SENSOR_SITE_NAME)
    force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, FORCE_SENSOR_NAME)
    torque_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, TORQUE_SENSOR_NAME)
    if sensor_site_id == -1:
        raise ValueError(f"未找到力传感器 site: {SENSOR_SITE_NAME}")
    if force_sensor_id == -1 or torque_sensor_id == -1:
        raise ValueError(
            f"未找到六维力传感器: {FORCE_SENSOR_NAME} / {TORQUE_SENSOR_NAME}"
        )

    data.qpos[:NV_ARM] = INITIAL_Q
    mujoco.mj_forward(model, data)

    T_d = transform_from_pose(data.xpos[ee_body_id].copy(), data.xquat[ee_body_id].copy())
    V_d = np.zeros(6)
    V_d_dot = np.zeros(6)
    M_full = np.zeros((model.nv, model.nv))
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    nric = NRICTorqueController(
        nominal_mjcf_path=NOMINAL_XML_PATH,
        end_effector_frame_name=NRIC_WRENCH_FRAME,
        time_step=model.opt.timestep,
        gravity_z=model.opt.gravity[2],
        enabled=ENABLE_NRIC,
        gic=gic,
    )
    nric.reset(data.qpos[:NV_ARM].copy(), data.qvel[:NV_ARM].copy())

    print("\n开始运行带摩擦的 SE(3) 阻抗控制仿真...")
    if ENABLE_NRIC:
        print(
            "控制律 = MuJoCo 当前状态下的 SE(3) 阻抗力矩 tau_nominal + "
            "Pinocchio 无摩擦名义模型产生的 NRIC 补偿 tau_aux。"
        )
    else:
        print("控制律 = MuJoCo 当前状态下计算的纯 Python SE(3) 阻抗力矩。")
    print(
        f"MuJoCo viewer = {'off (--headless)' if args.headless else 'on'}; "
        f"NRIC = {'on (ENABLE_NRIC=True)' if ENABLE_NRIC else 'off (ENABLE_NRIC=False)'}; "
        f"joint7 frictionloss = {args.joint7_frictionloss:g}; "
        "外力来自 MuJoCo site force/torque sensor。\n"
    )

    viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)

    with viewer_context as viewer:
        step = 0
        while (args.max_steps <= 0 or step < args.max_steps) and (viewer is None or viewer.is_running()):
            step += 1

            q = data.qpos[:NV_ARM].copy()
            dq = data.qvel[:NV_ARM].copy()
            T = transform_from_pose(data.xpos[ee_body_id].copy(), data.xquat[ee_body_id].copy())
            T_site = transform_from_xmat(
                data.site_xpos[sensor_site_id].copy(),
                data.site_xmat[sensor_site_id].copy(),
            )
            R = T[:3, :3]
            F_ext_site = read_wrench_sensor(model, data, force_sensor_id, torque_sensor_id)
            F_ext_link7 = transform_wrench_between_frames(F_ext_site, T_site, T)

            mujoco.mj_jacBody(model, data, jac_p, jac_r, ee_body_id)
            J_world = np.vstack((jac_p[:, :NV_ARM], jac_r[:, :NV_ARM]))
            J = np.vstack((R.T @ J_world[:3, :], R.T @ J_world[3:, :]))
            V = J @ dq

            mujoco.mj_fullM(model, M_full, data.qM)
            M = M_full[:NV_ARM, :NV_ARM]
            qfrc_bias = data.qfrc_bias[:NV_ARM].copy()

            tau_nominal = controller.compute_torque(
                q=q,
                q_dot=dq,
                T=T,
                V=V,
                J=J,
                M=M,
                qfrc_bias=qfrc_bias,
                J_dot_q_dot=np.zeros(6),
                T_d=T_d,
                V_d=V_d,
                V_d_dot=V_d_dot,
                F_ext=F_ext_link7,
            )

            control = nric.compute_torque(
                q=q,
                dq=dq,
                tau_nominal_mujoco=tau_nominal,
                T_d=T_d,
                V_d=V_d,
                V_d_dot=V_d_dot,
                F_ext=F_ext_link7,
            )

            data.ctrl[:NV_ARM] = control["tau"]
            mujoco.mj_step(model, data)

            if viewer is not None and step % VIEWER_SYNC_INTERVAL == 0:
                viewer.sync()


if __name__ == "__main__":
    main()
