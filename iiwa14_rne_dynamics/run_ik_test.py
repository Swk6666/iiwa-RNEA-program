import numpy as np
import mujoco
from mujoco import minimize

from ik_solver import MujocoIKSolver


XML_PATH = "kuka_iiwa_14/iiwa14.xml"
EE_SITE = "attachment_site"


def main():
    target_pos = np.array([0.62, -0.103, 0.796])
    target_quat = np.array([0.597, -0.115, 0.748, -0.267])

    solver = MujocoIKSolver(XML_PATH, EE_SITE)

    sol, trace = solver.solve(
        target_pos,
        target_quat,
        radius=0.04,
        reg=1e-3,
        reg_target=solver.q0,
        verbose=minimize.Verbosity.FINAL,
    )

    ee_pos, ee_quat = solver.forward_kinematics(sol)
    pos_err, quat_err = solver.pose_error(ee_pos, ee_quat, target_pos, target_quat)

    print("Solution qpos:", np.round(sol, 6))
    print("EE position:", np.round(ee_pos, 6))
    print("Target position:", target_pos)
    print("Position error (norm):", np.linalg.norm(pos_err))
    print("Quaternion (wxyz):", np.round(ee_quat, 6))
    print("Target quaternion:", solver._normalize_quat(target_quat))
    print("Quat residual scaled:", np.round(quat_err, 6))
    print("Quat residual norm:", np.linalg.norm(quat_err))
    print("Iterations:", len(trace))


if __name__ == "__main__":
    main()
