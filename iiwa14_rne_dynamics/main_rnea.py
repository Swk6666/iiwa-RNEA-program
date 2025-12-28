import numpy as np
import mujoco
import mujoco.viewer
from scipy.io import savemat
from tqdm import tqdm
from mujoco import minimize
from ik_solver import MujocoIKSolver
from Polynolial_traj import plan_quintic, quintic_coeffs, sample_trajectory


xml_path = "iiwa_description/urdf/iiwa14.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# 设定仿真时间步长（默认 0.002s，会导致时间翻倍），与轨迹 dt 保持一致
SIM_DT = 0.001
mj_model.opt.timestep = SIM_DT


def get_mj_body_pos_by_name( name: str) -> np.ndarray:
    """获取MuJoCo body位置"""

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise ValueError(f"未找到名为 {name} 的 body")
    return mj_data.xpos[body_id].copy()

def get_mj_body_quat_by_name( name: str) -> np.ndarray:
    """获取MuJoCo body四元数"""

    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise ValueError(f"未找到名为 {name} 的 body")
    return mj_data.xquat[body_id].copy()

# 定义目标位置和姿态
target_pos = np.array([0.62, -0.103, 0.796])
target_quat = np.array([0.597, -0.115, 0.748, -0.267])

# 定义机械臂末端站点名称
EE_SITE = "attachment_site"

# 创建IK求解器实例
solver = MujocoIKSolver(xml_path, EE_SITE)

initial_q = np.zeros(7)
# 求解IK问题
sol, trace = solver.solve(
    target_pos, # 目标位置
    target_quat, # 目标四元数
    radius=0.04, # 位置容差
    reg=1e-3, # 正则化项权重
    reg_target=initial_q, # 初始迭代开始关节角度
    verbose=minimize.Verbosity.FINAL, # 日志输出
)

print("IK求解结果:", np.round(sol, 6))


# 关节空间内进行五次多项式的规划
p0 = np.zeros(7)  # 初始位置
pT = sol
T = 2.0  # 终止时间
dt = SIM_DT
num = int(T / dt) + 1  # 20001

coeffs, t, pos, vel, acc = plan_quintic(p0, pT, T=T, num=num)

traj_len = pos.shape[0]
torque = np.zeros((traj_len, mj_model.nv))
for i in range(traj_len):
    mj_data.qpos[:mj_model.nq] = pos[i]
    mj_data.qvel[:mj_model.nv] = vel[i]
    mj_data.qacc[:mj_model.nv] = acc[i]
    mujoco.mj_inverse(mj_model, mj_data)
    torque[i] = mj_data.qfrc_inverse[:mj_model.nv]
print("pos[1000]:", pos[1000])
print("vel[1000]:", vel[1000])
print("acc[1000]:", acc[1000])
print("torque[1000]:", torque[1000])
savemat("torque_mujoco.mat", {"torque": torque, "t": t})

def pinocchio_rnea_from_urdf(urdf_path, q, v, a, gravity=None):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    q = np.asarray(q).reshape((model.nq,))
    v = np.asarray(v).reshape((model.nv,))
    a = np.asarray(a).reshape((model.nv,))

    if gravity is not None:
        model.gravity.linear = np.asarray(gravity).reshape((3,))

    return pin.rnea(model, data, q, v, a)


def pinocchio_rnea_trajectory(urdf_path, pos, vel, acc, gravity=None):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    pos = np.asarray(pos)
    vel = np.asarray(vel)
    acc = np.asarray(acc)

    if pos.shape != vel.shape or pos.shape != acc.shape:
        raise ValueError("pos, vel, acc must have the same shape.")
    if pos.shape[1] != model.nq or vel.shape[1] != model.nv:
        raise ValueError("pos/vel/acc second dimension must match nq/nv.")

    if gravity is not None:
        model.gravity.linear = np.asarray(gravity).reshape((3,))

    tau = np.zeros((pos.shape[0], model.nv))
    for i in range(pos.shape[0]):
        tau[i] = pin.rnea(model, data, pos[i], vel[i], acc[i])

    return tau

tau_pin = pinocchio_rnea_from_urdf(
    "iiwa_description/urdf/iiwa14.urdf",
    pos[1000], vel[1000], acc[1000],
    gravity=[0.0, 0.0, 0.0],
)
print("pinocchio tau:", tau_pin)
