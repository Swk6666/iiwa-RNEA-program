#!/usr/bin/env python3
"""Compare Pinocchio RNEA results from the IIWA14 URDF and MJCF models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin


SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "iiwa_description" / "urdf" / "iiwa14.urdf"
MJCF_PATH = SCRIPT_DIR / "kuka_iiwa_14" / "iiwa14.xml"

Q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
QD = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
QDD = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
EXPECTED_TAU = np.array(
    [0.0312, 0.4054, 0.0956, -0.0304, 0.0075, 0.0003, 0.0011],
    dtype=float,
)
ZERO_GRAVITY = np.array([0,0,-9.81])


def load_model(model_path: Path) -> pin.Model:
    suffix = model_path.suffix.lower()
    if suffix == ".urdf":
        return pin.buildModelFromUrdf(str(model_path))
    if suffix == ".xml":
        return pin.buildModelFromMJCF(str(model_path))
    raise ValueError(f"Unsupported model format: {model_path}")


def compute_rnea(model_path: Path, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    model = load_model(model_path)
    model.gravity.linear = ZERO_GRAVITY.copy()
    data = model.createData()
    return pin.rnea(model, data, q, qd, qdd)


def print_result(label: str, tau: np.ndarray) -> None:
    print(f"{label} full precision:")
    print(np.array2string(tau, precision=10, separator=", "))
    print(f"{label} rounded to 4 decimals:")
    print(np.array2string(np.round(tau, 4), precision=4, separator=", "))
    print()


def main() -> None:
    np.set_printoptions(suppress=True)

    print(f"URDF path: {URDF_PATH}")
    print(f"MJCF path: {MJCF_PATH}")
    print(f"gravity: {ZERO_GRAVITY}")
    print()

    urdf_tau = compute_rnea(URDF_PATH, Q, QD, QDD)
    mjcf_tau = compute_rnea(MJCF_PATH, Q, QD, QDD)

    print_result("URDF", urdf_tau)
    print_result("MJCF", mjcf_tau)

    print("Expected tau (given answer):")
    print(np.array2string(EXPECTED_TAU, precision=4, separator=", "))
    print()

    urdf_matches_expected = np.allclose(urdf_tau, EXPECTED_TAU, atol=5e-5, rtol=0.0)
    mjcf_matches_expected = np.allclose(mjcf_tau, EXPECTED_TAU, atol=5e-5, rtol=0.0)
    urdf_matches_mjcf = np.allclose(urdf_tau, mjcf_tau, atol=1e-12, rtol=0.0)

    print("URDF - expected:")
    print(np.array2string(urdf_tau - EXPECTED_TAU, precision=10, separator=", "))
    print(f"match within atol=5e-5: {urdf_matches_expected}")
    print()

    print("MJCF - expected:")
    print(np.array2string(mjcf_tau - EXPECTED_TAU, precision=10, separator=", "))
    print(f"match within atol=5e-5: {mjcf_matches_expected}")
    print()

    print("URDF - MJCF:")
    print(np.array2string(urdf_tau - mjcf_tau, precision=10, separator=", "))
    print(f"URDF and MJCF identical within atol=1e-12: {urdf_matches_mjcf}")


if __name__ == "__main__":
    main()
