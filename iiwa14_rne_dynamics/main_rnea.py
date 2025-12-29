import numpy as np
import mujoco
import mujoco.viewer
from scipy.io import savemat
from tqdm import tqdm
from mujoco import minimize
from ik_solver import MujocoIKSolver
from Polynolial_traj import plan_quintic, quintic_coeffs, sample_trajectory


xml_path = "kuka_iiwa_14/scene_torque.xml"
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
T = 20.0  # 终止时间
dt = SIM_DT
num = int(T / dt) + 1  # 20001

coeffs, t, pos, vel, acc = plan_quintic(p0, pT, T=T, num=num)

traj_len = pos.shape[0]



'''
前面的代码不变，下面这段代码是用mujoco的rnea方法计算逆动力学，并保存为mat文件
'''
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




'''
下面的代码是用pinocchio计算逆动力学
'''
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

traj_len = pos.shape[0]


'''
下面这段代码是用pinocchio计算逆动力学，作为前馈，加上反馈，实现力矩控制下的轨迹跟踪
'''

# PD控制器增益
Kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])/1000
Kd = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])/1000

mj_data.qpos[:7] = np.zeros(7) 
mj_data.qvel[:7] = np.zeros(7) 
mujoco.mj_forward(mj_model, mj_data)
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer, tqdm(total=0, dynamic_ncols=True) as pbar:
    step_count = 0
    print_interval = 100  # 每100步更新一次 tqdm 显示
    
    while viewer.is_running():
        idx = min(step_count, traj_len - 1)  # 超出轨迹后保持末端姿态
        
        # 获取当前关节状态
        q_current = mj_data.qpos[:7].copy()
        v_current = mj_data.qvel[:7].copy()
        
        # 获取期望轨迹状态
        q_desired = pos[idx]
        v_desired = vel[idx]
        a_desired = acc[idx]
        
        # 使用pinocchio计算前馈力矩（逆动力学）
        tau_feedforward = pinocchio_rnea_from_urdf(
            "iiwa_description/urdf/iiwa14.urdf",
            q_desired, v_desired, a_desired,
            gravity=[0.0, 0.0, 0.0],
        )
        
        # 计算PD反馈力矩
        tau_feedback = Kp * (q_desired - q_current) + Kd * (v_desired - v_current)
        
        # 总控制力矩 = 前馈 + 反馈
        tau_total = tau_feedforward + tau_feedback
        
        # 设置力矩控制输入
        mj_data.ctrl[:] = tau_total
        
        mujoco.mj_step(mj_model, mj_data)
        step_count += 1

        # 轨迹执行完毕后自动退出
        if step_count >= traj_len:
            break

        if step_count % print_interval == 0:
            actual_ee_pos = get_mj_body_pos_by_name("link7")
            actual_ee_quat = get_mj_body_quat_by_name("link7")
            pbar.set_description(f"time: {mj_data.time:.3f}s | pose: [{np.round(actual_ee_pos, 3)}] | quat: [{np.round(actual_ee_quat, 3)}]")
            pbar.update(1)  # 更新一次进度条（动态刷新）

        viewer.sync()

