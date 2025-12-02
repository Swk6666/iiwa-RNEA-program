"""
Compute the spatial transforms (X matrices) and spatial inertias (Imat) for
the iiwa14 arm directly from the URDF using Pinocchio.

The resulting X matrices match the ones produced by xmat0-6.m, and Imat
matches the matrices hard-coded in main.m.
"""

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pinocchio as pin


# Path to the URDF used by the MATLAB code
URDF_PATH = Path(__file__).parent / "URDFParser" / "iiwa.urdf"

# Joint names in the order expected by main.m (iiwa_joint_1 ... iiwa_joint_7)
JOINT_NAMES = [f"iiwa_joint_{i}" for i in range(1, 8)]

# Build the model once
MODEL = pin.buildModelFromUrdf(str(URDF_PATH))
DATA = MODEL.createData()
JOINT_IDS = [MODEL.getJointId(name) for name in JOINT_NAMES]

# Permutation matrix to switch between Pinocchio's spatial order [v; w]
# and the MATLAB code's order [w; v]
PERMUTE = np.block(
    [
        [np.zeros((3, 3)), np.eye(3)],
        [np.eye(3), np.zeros((3, 3))],
    ]
)


def fill_q(q_list: Iterable[float]) -> np.ndarray:
    """
    Place the provided joint values into a full Pinocchio configuration vector.
    Fixed joints contribute zero dof so we only overwrite revolute joints.
    """
    q = pin.neutral(MODEL)
    for jid, q_val in zip(JOINT_IDS, q_list):
        idx = MODEL.idx_qs[jid]
        q[idx : idx + MODEL.nqs[jid]] = q_val
    return q


def compute_xmat_and_inertia(
    q_list: Iterable[float],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
        xmat_list: list of 6x6 spatial motion transforms (parent -> child)
                   in the [w; v] ordering used by the MATLAB code
        inertia_list: list of 6x6 spatial inertia matrices (in [w; v] order)
    """
    q = fill_q(q_list)
    pin.forwardKinematics(MODEL, DATA, q)

    xmat_list: List[np.ndarray] = []
    inertia_list: List[np.ndarray] = []

    for jid in JOINT_IDS:
        parent = MODEL.parents[jid]
        # Transform from parent joint frame to current joint frame
        pMi = DATA.oMi[parent].inverse() * DATA.oMi[jid]
        # Convert Pinocchio's [v; w] convention to MATLAB's [w; v]
        xmat_list.append(PERMUTE @ pMi.inverse().toActionMatrix() @ PERMUTE)
        inertia_list.append(PERMUTE @ MODEL.inertias[jid].matrix() @ PERMUTE)

    return xmat_list, inertia_list


def _decompose_entry(a0: float, a_pi: float, a_pi2: float, tol: float = 1e-12) -> str:
    """
    Decompose a linear expression a*c + b*s + d from samples at:
        theta=0   -> a0 (c=1,s=0)
        theta=pi  -> a_pi (c=-1,s=0)
        theta=pi/2-> a_pi2 (c=0,s=1)
    Returns a MATLAB-friendly string.
    """
    d = (a0 + a_pi) / 2.0
    a = (a0 - a_pi) / 2.0
    b = a_pi2 - d

    def fmt_coeff(val: float) -> str:
        if abs(val) < tol:
            return ""
        if abs(val - 1) < tol:
            return ""
        if abs(val + 1) < tol:
            return "-"
        return f"{val:.12g}*"

    terms = []
    coeff = fmt_coeff(a)
    if coeff or abs(a) >= tol:
        terms.append(f"{coeff}cos(theta)" if coeff not in ("", "-") else f"{coeff}cos(theta)".replace("**", "*"))

    coeff = fmt_coeff(b)
    if coeff or abs(b) >= tol:
        terms.append(f"{coeff}sin(theta)" if coeff not in ("", "-") else f"{coeff}sin(theta)".replace("**", "*"))

    if abs(d) >= tol:
        terms.append(f"{d:.12g}")

    if not terms:
        return "0"

    expr = " + ".join(terms)
    expr = expr.replace("+ -", "- ")
    expr = expr.replace("  ", " ")
    return expr


def generate_matlab_xmat_functions() -> str:
    """
    Produce MATLAB function text xmat0 ... xmat6 matching the existing format,
    using Pinocchio model data. Each function returns the [w; v] spatial
    transform from parent to child.
    """
    lines: List[str] = []
    for func_idx, jid in enumerate(JOINT_IDS):
        # Sample X(theta) at 3 angles to recover linear terms in cos/sin
        def x_at(theta: float) -> np.ndarray:
            q = np.zeros(MODEL.nq)
            q[MODEL.idx_qs[jid] : MODEL.idx_qs[jid] + MODEL.nqs[jid]] = theta
            pin.forwardKinematics(MODEL, DATA, q)
            parent = MODEL.parents[jid]
            pMi = DATA.oMi[parent].inverse() * DATA.oMi[jid]
            return PERMUTE @ pMi.inverse().toActionMatrix() @ PERMUTE

        X0 = x_at(0.0)
        Xpi = x_at(np.pi)
        Xpi2 = x_at(np.pi / 2.0)

        lines.append(f"function X = xmat{func_idx}(theta)")
        lines.append("    X = [")
        for r in range(6):
            row_terms = []
            for c in range(6):
                expr = _decompose_entry(X0[r, c], Xpi[r, c], Xpi2[r, c])
                row_terms.append(expr)
            row_str = "        " + ", ".join(row_terms) + ";"
            lines.append(row_str)
        lines.append("    ];")
        lines.append("end")
        lines.append("")  # blank line between functions

    return "\n".join(lines)


if __name__ == "__main__":
    # Example: inertia matrices in the same order as main.m (w; v ordering)
    _, I_list = compute_xmat_and_inertia([0.0] * 7)

    np.set_printoptions(precision=6, suppress=True)
    for idx, name in enumerate(JOINT_NAMES):
        print(f"{name} spatial inertia (Imat):\n{I_list[idx]}\n")

    print("Generated MATLAB xmat* functions:\n")
    print(generate_matlab_xmat_functions())
