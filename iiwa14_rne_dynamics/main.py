import numpy as np
import mujoco
import mujoco.viewer
from tqdm import tqdm

xml_path = "kuka_iiwa_14/scene.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)


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


with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer, tqdm(total=0, dynamic_ncols=True) as pbar:
    step_count = 0
    print_interval = 100  # 每100步更新一次 tqdm 显示
    
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
       # mj_data.ctrl[:] = np.array([0.0] * mj_model.nu)  # 控制器输出为零
        step_count += 1

        if step_count % print_interval == 0:
            actual_ee_pos = get_mj_body_pos_by_name("link7")
            actual_ee_quat = get_mj_body_quat_by_name("link7")
            pbar.set_description(f"time: {mj_data.time:.3f}s | pose: [{np.round(actual_ee_pos, 3)}] | quat: [{np.round(actual_ee_quat, 3)}]")
            pbar.update(1)  # 更新一次进度条（动态刷新）

        viewer.sync()