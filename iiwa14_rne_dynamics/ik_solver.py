import numpy as np
import mujoco
from mujoco import minimize


class MujocoIKSolver:
    """General inverse kinematics solver backed by MuJoCo analytic Jacobians."""

    def __init__(self, xml_path: str, effector_site: str, init_qpos=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.eff_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, effector_site)
        if self.eff_sid == -1:
            raise ValueError(f"Cannot find site '{effector_site}' in model.")

        if init_qpos is not None:
            self.q0 = np.array(init_qpos, dtype=float)
        else:
            self.q0 = self._default_qpos()

        self.bounds = self._joint_bounds()

    def _default_qpos(self) -> np.ndarray:
        try:
            return self.model.key("home").qpos.copy()
        except Exception:
            return np.zeros(self.model.nq, dtype=float)

    def _joint_bounds(self):
        lower = self.model.jnt_range[:, 0].copy()
        upper = self.model.jnt_range[:, 1].copy()
        # If a joint is not limited, mjModel stores zeros; replace with wide bounds.
        limited = self.model.jnt_limited.astype(bool)
        lower[~limited] = -np.inf
        upper[~limited] = np.inf
        return [lower, upper]

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        quat = np.array(quat, dtype=float)
        norm = np.linalg.norm(quat)
        if norm == 0:
            raise ValueError("Quaternion has zero norm.")
        return quat / norm

    @staticmethod
    def _as_batch(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        return x

    def residual(self, x: np.ndarray, pos: np.ndarray, quat: np.ndarray,
                 radius: float = 0.04, reg: float = 1e-3, reg_target=None) -> np.ndarray:
        """Compute IK residual for batched joint vectors."""
        x = self._as_batch(x)
        pos = np.asarray(pos, dtype=float)
        target_quat = self._normalize_quat(quat)
        reg_target = self.q0 if reg_target is None else np.asarray(reg_target, dtype=float)

        residuals = []
        for i in range(x.shape[1]):
            self.data.qpos = x[:, i]
            mujoco.mj_kinematics(self.model, self.data)

            res_pos = self.data.site_xpos[self.eff_sid] - pos

            eff_quat = np.empty(4)
            mujoco.mju_mat2Quat(eff_quat, self.data.site_xmat[self.eff_sid])

            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, target_quat, eff_quat)
            res_quat *= radius

            res_reg = reg * (x[:, i] - reg_target)

            res_i = np.hstack((res_pos, res_quat, res_reg))
            residuals.append(np.atleast_2d(res_i).T)

        return np.hstack(residuals)

    def jacobian(self, x: np.ndarray, res: np.ndarray, pos: np.ndarray, quat: np.ndarray,
                 radius: float = 0.04, reg: float = 1e-3) -> np.ndarray:
        del res  # Not used, kept for API compatibility.

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        jac_pos = np.empty((3, self.model.nv))
        jac_rot = np.empty((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.eff_sid)

        eff_quat = np.empty(4)
        mujoco.mju_mat2Quat(eff_quat, self.data.site_xmat[self.eff_sid])
        target_quat = self._normalize_quat(quat)
        d_eff = np.empty((3, 3))
        # （mjd_subQuat本质：右雅可比 / 右雅可比逆）
        mujoco.mjd_subQuat(target_quat, eff_quat, None, d_eff)

        target_mat = np.empty(9)
        mujoco.mju_quat2Mat(target_mat, target_quat)
        target_mat = target_mat.reshape(3, 3)

        rot_scale = radius * d_eff.T @ target_mat.T
        jac_rot = rot_scale @ jac_rot

        jac_reg = reg * np.eye(self.model.nv)

        return np.vstack((jac_pos, jac_rot, jac_reg))

    def solve(self, target_pos, target_quat, *, radius=0.04, reg=1e-3,
              reg_target=None, q0=None, verbose=0, check_derivatives=False):
        q_init = self.q0 if q0 is None else np.asarray(q0, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = self._normalize_quat(target_quat)

        ik_fn = lambda x: self.residual(x, target_pos, target_quat,
                                        radius=radius, reg=reg, reg_target=reg_target)
        jac_fn = lambda x, r: self.jacobian(x, r, target_pos, target_quat,
                                            radius=radius, reg=reg)

        solution, trace = minimize.least_squares(q_init, ik_fn, self.bounds,
                                                 jacobian=jac_fn, verbose=verbose,
                                                 check_derivatives=check_derivatives)
        return solution, trace

    def forward_kinematics(self, qpos: np.ndarray):
        qpos = np.asarray(qpos, dtype=float)
        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.eff_sid].copy()
        mat = self.data.site_xmat[self.eff_sid]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat

    def pose_error(self, current_pos: np.ndarray, current_quat: np.ndarray,
                   target_pos: np.ndarray, target_quat: np.ndarray, radius: float = 0.04):
        pos_err = current_pos - target_pos
        quat_err = np.empty(3)
        mujoco.mju_subQuat(quat_err, self._normalize_quat(target_quat),
                           self._normalize_quat(current_quat))
        quat_err *= radius
        return pos_err, quat_err
