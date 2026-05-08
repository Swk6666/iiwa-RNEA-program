"""Small MuJoCo runtime helpers used by the example scripts."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Optional

import mujoco
import numpy as np


class NullCapture:
    """No-op context manager kept where older scripts expected frame capture."""

    output_path = None

    def __enter__(self) -> "NullCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def capture_frame(self, force: bool = False) -> None:
        return None

    def set_external_wrench_overlay(self, *args, **kwargs) -> None:
        return None


class ExternalWrenchOverlay:
    """Draw the current world-frame wrench applied to a MuJoCo body."""

    _FORCE_RGBA = np.array([0.9, 0.2, 0.2, 0.85], dtype=np.float32)
    _TORQUE_RGBA = np.array([0.2, 0.8, 0.9, 0.85], dtype=np.float32)
    _IDENTITY_MAT = np.eye(3, dtype=np.float64).ravel()
    _ZEROS3 = np.zeros(3, dtype=np.float64)

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
        force_scale: float = 0.01,
        torque_scale: float = 0.03,
        arrow_radius: float = 0.006,
        torque_offset: float = 0.035,
    ):
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        self.force_scale = force_scale
        self.torque_scale = torque_scale
        self.arrow_radius = arrow_radius
        self.torque_offset = torque_offset

    @staticmethod
    def _scaled_vector(vector: np.ndarray, scale: float, min_length: float = 0.03, max_length: float = 0.35):
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            return None
        length = np.clip(scale * norm, min_length, max_length)
        return vector / norm * length

    def draw(self, scene: mujoco.MjvScene) -> int:
        if scene.ngeom >= scene.maxgeom:
            return 0

        origin = self.data.xpos[self.body_id]
        wrench_world = self.data.xfrc_applied[self.body_id]
        drawn = 0

        force_vec = self._scaled_vector(wrench_world[:3], self.force_scale)
        if force_vec is not None and scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                self._ZEROS3,
                self._ZEROS3,
                self._IDENTITY_MAT,
                self._FORCE_RGBA,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                self.arrow_radius,
                origin,
                origin + force_vec,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1
            drawn += 1

        torque_vec = self._scaled_vector(wrench_world[3:], self.torque_scale)
        if torque_vec is not None and scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                self._ZEROS3,
                self._ZEROS3,
                self._IDENTITY_MAT,
                self._TORQUE_RGBA,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                self.arrow_radius,
                origin,
                origin + torque_vec,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1
            drawn += 1

        return drawn


class SimulationSession:
    """Context manager for optional MuJoCo passive visualization."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        enable_visualization: bool = True,
    ):
        self.model = model
        self.data = data
        self.enable_visualization = enable_visualization
        self.viewer: Optional[object] = None
        self._stack: Optional[ExitStack] = None

    def __enter__(self) -> "SimulationSession":
        self._stack = ExitStack()
        if self.enable_visualization:
            from mujoco import viewer as mj_viewer

            self.viewer = self._stack.enter_context(mj_viewer.launch_passive(self.model, self.data))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stack is not None:
            self._stack.close()
        self.viewer = None

    def post_step(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def capture_frame(self, force: bool = False) -> None:
        return None
