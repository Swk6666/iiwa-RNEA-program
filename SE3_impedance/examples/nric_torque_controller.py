import os

import numpy as np

from se3_impedance.geometry_impedance_control import geometry_impedance_control
from se3_impedance.se3_controller import SE3ImpedanceController


NV_ARM = 7


def load_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise ImportError(
            "NRIC 已开启，但当前 Python 环境没有 pinocchio。"
            "请使用包含 pinocchio 的环境运行，例如 "
            "`/opt/miniconda3/envs/swkpy310/bin/python examples/franka_se3_test_friction.py`。"
        ) from exc
    return pin


def build_pinocchio_model(pin, mjcf_path, gravity_z):
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"Pinocchio nominal MJCF not found: {mjcf_path}")
    model = pin.buildModelFromMJCF(mjcf_path)
    model.gravity.linear[:] = 0.0
    model.gravity.angular[:] = 0.0
    model.gravity.linear[2] = float(gravity_z)
    if model.nq != NV_ARM or model.nv != NV_ARM:
        raise RuntimeError(
            f"Expected a 7-DoF fixed-base Franka model, got nq={model.nq}, nv={model.nv}."
        )
    return model


def as_vector7(value, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (NV_ARM,):
        raise ValueError(f"{label} must be shape ({NV_ARM},).")
    return value


def as_diag7(value, default, label):
    if value is None:
        return np.eye(NV_ARM) * default
    value = np.asarray(value, dtype=float)
    if value.shape == (NV_ARM,):
        return np.diag(value)
    if value.shape == (NV_ARM, NV_ARM):
        return value
    raise ValueError(f"{label} must be shape ({NV_ARM},) or ({NV_ARM}, {NV_ARM}).")


class PinocchioNominalSE3Controller:
    def __init__(self, mjcf_path, end_effector_frame_name, gravity_z, gic, q_nominal=None):
        self.pin = load_pinocchio()
        self.model = build_pinocchio_model(self.pin, mjcf_path, gravity_z)
        self.data = self.model.createData()
        if not self.model.existFrame(end_effector_frame_name):
            raise RuntimeError(f"End-effector frame not found: {end_effector_frame_name}")
        self.frame_id = self.model.getFrameId(end_effector_frame_name)
        self.controller = SE3ImpedanceController(gic)
        if q_nominal is not None:
            self.controller.q_nominal = as_vector7(q_nominal, "q_nominal").copy()

    def compute_torque(self, q, dq, T_d, V_d, V_d_dot, F_ext):
        q = as_vector7(q, "q")
        dq = as_vector7(dq, "dq")
        J = np.asarray(
            self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_id,
                self.pin.ReferenceFrame.LOCAL,
            ),
            dtype=float,
        )
        self.pin.forwardKinematics(self.model, self.data, q, dq, np.zeros(NV_ARM))
        self.pin.updateFramePlacements(self.model, self.data)

        T = np.asarray(self.data.oMf[self.frame_id].homogeneous, dtype=float)
        V = np.asarray(
            self.pin.getFrameVelocity(
                self.model,
                self.data,
                self.frame_id,
                self.pin.ReferenceFrame.LOCAL,
            ).vector,
            dtype=float,
        )
        J_dot_q_dot = np.asarray(
            self.pin.getFrameAcceleration(
                self.model,
                self.data,
                self.frame_id,
                self.pin.ReferenceFrame.LOCAL,
            ).vector,
            dtype=float,
        )
        qfrc_bias = np.asarray(self.pin.nonLinearEffects(self.model, self.data, q, dq), dtype=float)
        M = np.asarray(self.pin.crba(self.model, self.data, q), dtype=float)
        M = np.triu(M) + np.triu(M, 1).T

        return self.controller.compute_torque(
            q=q,
            q_dot=dq,
            T=T,
            V=V,
            J=J,
            M=M,
            qfrc_bias=qfrc_bias,
            J_dot_q_dot=J_dot_q_dot,
            T_d=T_d,
            V_d=V_d,
            V_d_dot=V_d_dot,
            F_ext=F_ext,
        )


class PinocchioNominalSimulator:
    def __init__(
        self,
        mjcf_path,
        end_effector_frame_name,
        time_step,
        gravity_z,
        include_joint_damping=True,
    ):
        self.pin = load_pinocchio()
        self.model = build_pinocchio_model(self.pin, mjcf_path, gravity_z)
        self.data = self.model.createData()
        if not self.model.existFrame(end_effector_frame_name):
            raise RuntimeError(f"End-effector frame not found: {end_effector_frame_name}")
        self.frame_id = self.model.getFrameId(end_effector_frame_name)
        self.parent_joint_id = self.model.frames[self.frame_id].parentJoint
        if self.parent_joint_id == 0:
            raise RuntimeError(
                f"The end-effector frame must be attached to a non-root joint: {end_effector_frame_name}"
            )
        self.dt = float(time_step)
        if self.dt <= 0.0:
            raise ValueError("time_step must be strictly positive.")
        self.include_joint_damping = bool(include_joint_damping)
        self.fext = [self.pin.Force.Zero() for _ in range(self.model.njoints)]
        self.q = np.zeros(NV_ARM)
        self.dq = np.zeros(NV_ARM)
        self.ddq = np.zeros(NV_ARM)
        self.initialized = False

    def reset(self, q0, dq0=None):
        self.q = as_vector7(q0, "q0").copy()
        self.dq = np.zeros(NV_ARM) if dq0 is None else as_vector7(dq0, "dq0").copy()
        self.ddq = np.zeros(NV_ARM)
        self._clear_external_forces()
        self.initialized = True

    def step(self, tau, F_ext=None):
        if not self.initialized:
            raise RuntimeError("Call reset(q0) before stepping the nominal simulator.")
        tau = as_vector7(tau, "tau").copy()
        F_ext = np.zeros(6) if F_ext is None else np.asarray(F_ext, dtype=float)
        if F_ext.shape != (6,):
            raise ValueError("F_ext must be shape (6,).")

        self._clear_external_forces()
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        self.fext[self.parent_joint_id] = self._map_external_wrench_to_joint(F_ext)

        tau_effective = tau.copy()
        if self.include_joint_damping:
            tau_effective -= np.asarray(self.model.damping, dtype=float) * self.dq

        self.ddq = np.asarray(
            self.pin.aba(self.model, self.data, self.q, self.dq, tau_effective, self.fext),
            dtype=float,
        )
        self.dq = self.dq + self.dt * self.ddq
        self.q = np.asarray(self.pin.integrate(self.model, self.q, self.dt * self.dq), dtype=float)
        return self.q.copy(), self.dq.copy()

    def _clear_external_forces(self):
        for index in range(self.model.njoints):
            self.fext[index] = self.pin.Force.Zero()

    def _map_external_wrench_to_joint(self, F_ext):
        force_local = self.pin.Force(F_ext[:3], F_ext[3:])
        force_world = self.data.oMf[self.frame_id].act(force_local)
        return self.data.oMi[self.parent_joint_id].actInv(force_world)


class NRICTorqueController:
    def __init__(
        self,
        nominal_mjcf_path,
        end_effector_frame_name="link7",
        time_step=1e-3,
        gravity_z=-9.81,
        enabled=False,
        gic=None,
        k_gamma=None,
        k_p=None,
        k_i=None,
        integral_limit=0.2,
        integral_deadband=5e-3,
        integral_leak_rate=0.2,
        zero_cross_reset_ratio=0.2,
        integral_unwind_rate=5.0,
        q_nominal=None,
    ):
        self.enabled = bool(enabled)
        self.dt = float(time_step)
        self.K_gamma = as_diag7(k_gamma, 2.0, "k_gamma")
        self.K_p = as_diag7(k_p, 5.0, "k_p")
        self.K_i = as_diag7(k_i, 0.1, "k_i")
        self.integral_limit = float(integral_limit)
        self.integral_deadband = float(integral_deadband)
        self.integral_leak_rate = float(integral_leak_rate)
        self.zero_cross_reset_ratio = float(zero_cross_reset_ratio)
        self.integral_unwind_rate = float(integral_unwind_rate)
        self.integral_error = np.zeros(NV_ARM)
        self.prev_error = np.zeros(NV_ARM)
        self.initialized = False
        self.nominal_controller = None
        self.nominal_simulator = None

        if self.enabled:
            if gic is None:
                gic = geometry_impedance_control()
            self.nominal_controller = PinocchioNominalSE3Controller(
                nominal_mjcf_path,
                end_effector_frame_name,
                gravity_z,
                gic,
                q_nominal=q_nominal,
            )
            self.nominal_simulator = PinocchioNominalSimulator(
                nominal_mjcf_path,
                end_effector_frame_name,
                time_step,
                gravity_z,
                include_joint_damping=True,
            )

    def reset(self, q0, dq0=None):
        if not self.enabled:
            self.initialized = True
            return
        dq0 = np.zeros(NV_ARM) if dq0 is None else dq0
        self.nominal_simulator.reset(q0, dq0)
        self.integral_error.fill(0.0)
        self.prev_error.fill(0.0)
        self.initialized = True

    def get_nominal_state(self):
        if not self.enabled:
            raise RuntimeError("NRIC is disabled; no nominal state is available.")
        if not self.initialized:
            raise RuntimeError("NRIC nominal simulator has not been initialized yet.")
        return self.nominal_simulator.q.copy(), self.nominal_simulator.dq.copy()

    def compute_torque(self, q, dq, tau_nominal_mujoco, T_d, V_d=None, V_d_dot=None, F_ext=None):
        q = as_vector7(q, "q")
        dq = as_vector7(dq, "dq")
        tau_nominal_mujoco = as_vector7(tau_nominal_mujoco, "tau_nominal_mujoco")
        T_d = np.asarray(T_d, dtype=float)
        V_d = np.zeros(6) if V_d is None else np.asarray(V_d, dtype=float)
        V_d_dot = np.zeros(6) if V_d_dot is None else np.asarray(V_d_dot, dtype=float)
        F_ext = np.zeros(6) if F_ext is None else np.asarray(F_ext, dtype=float)

        if not self.enabled:
            return self._disabled_output(tau_nominal_mujoco)
        if not self.initialized:
            self.reset(q, dq)

        q_nominal = self.nominal_simulator.q.copy()
        dq_nominal = self.nominal_simulator.dq.copy()
        tau_nominal_pin = self.nominal_controller.compute_torque(
            q_nominal,
            dq_nominal,
            T_d,
            V_d,
            V_d_dot,
            F_ext,
        )

        e_nr = q_nominal - q
        e_nr_dot = dq_nominal - dq
        self._update_integral_error(e_nr)

        tau_aux_integral = self.K_gamma @ (self.K_i @ self.integral_error)
        tau_aux_pd = self.K_gamma @ (e_nr_dot + self.K_p @ e_nr)
        tau_aux = tau_aux_pd + tau_aux_integral

        self.nominal_simulator.step(tau_nominal_pin, F_ext)

        return {
            "tau": tau_nominal_mujoco + tau_aux,
            "tau_nominal": tau_nominal_mujoco,
            "tau_nominal_mujoco": tau_nominal_mujoco,
            "tau_nominal_pin": tau_nominal_pin,
            "tau_aux": tau_aux,
            "tau_aux_pd": tau_aux_pd,
            "tau_aux_integral": tau_aux_integral,
            "e_nr": e_nr,
            "e_nr_dot": e_nr_dot,
            "integral_error": self.integral_error.copy(),
            "q_nominal": q_nominal,
            "dq_nominal": dq_nominal,
        }

    def _disabled_output(self, tau_nominal_mujoco):
        zeros = np.zeros(NV_ARM)
        return {
            "tau": tau_nominal_mujoco,
            "tau_nominal": tau_nominal_mujoco,
            "tau_nominal_mujoco": tau_nominal_mujoco,
            "tau_nominal_pin": zeros.copy(),
            "tau_aux": zeros.copy(),
            "tau_aux_pd": zeros.copy(),
            "tau_aux_integral": zeros.copy(),
            "e_nr": zeros.copy(),
            "e_nr_dot": zeros.copy(),
            "integral_error": self.integral_error.copy(),
            "q_nominal": zeros.copy(),
            "dq_nominal": zeros.copy(),
        }

    def _update_integral_error(self, e_nr):
        leak = np.clip(1.0 - self.integral_leak_rate * self.dt, 0.0, 1.0)
        self.integral_error *= leak

        zero_cross_mask = self.prev_error * e_nr < 0.0
        if np.any(zero_cross_mask):
            self.integral_error[zero_cross_mask] *= self.zero_cross_reset_ratio

        small_error_mask = np.abs(e_nr) < self.integral_deadband
        oppose_mask = e_nr * self.integral_error < 0.0
        unwind_mask = small_error_mask & oppose_mask
        if np.any(unwind_mask):
            unwind = np.clip(1.0 - self.integral_unwind_rate * self.dt, 0.0, 1.0)
            self.integral_error[unwind_mask] *= unwind

        integrate_mask = ~unwind_mask
        self.integral_error[integrate_mask] += e_nr[integrate_mask] * self.dt
        self.integral_error = np.clip(
            self.integral_error, -self.integral_limit, self.integral_limit
        )
        self.prev_error = e_nr.copy()
