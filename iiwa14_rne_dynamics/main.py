import numpy as np
import mujoco
import mujoco.viewer
from tqdm import tqdm
from mujoco import minimize
from ik_solver import MujocoIKSolver
from Polynolial_traj import plan_quintic, quintic_coeffs, sample_trajectory


xml_path = "kuka_iiwa_14/scene.xml"
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

# 求解IK问题
sol, trace = solver.solve(
    target_pos,
    target_quat,
    radius=0.04,
    reg=1e-3,
    reg_target=solver.q0,
    verbose=minimize.Verbosity.FINAL,
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

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer, tqdm(total=0, dynamic_ncols=True) as pbar:
    step_count = 0
    print_interval = 100  # 每100步更新一次 tqdm 显示
    
    while viewer.is_running():
        idx = min(step_count, traj_len - 1)  # 超出轨迹后保持末端姿态
        mj_data.ctrl[:] = pos[idx]  # 设置当前时间步的关节位置作为控制输入
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
