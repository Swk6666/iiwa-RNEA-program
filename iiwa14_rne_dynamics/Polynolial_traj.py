"""
Quintic polynomial trajectory generator with zero start/end velocity and acceleration.

传入两个等长数组作为初末位置，默认终止时间 T=1，可指定采样时间序列获取轨迹（位置/速度/加速度）。
"""

import numpy as np
from typing import Iterable, Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]


def _to_numpy(vec: ArrayLike) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    return arr


def quintic_coeffs(p0: ArrayLike, pT: ArrayLike, T: float = 1.0) -> np.ndarray:
    """
    计算满足 v(0)=v(T)=0, a(0)=a(T)=0 的五次多项式系数。

    p(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
    """
    if T <= 0:
        raise ValueError("T 必须为正数")

    p0 = _to_numpy(p0)
    pT = _to_numpy(pT)
    if p0.shape != pT.shape:
        raise ValueError("p0 与 pT 形状必须一致")

    dp = pT - p0
    a0 = p0
    a1 = np.zeros_like(dp)
    a2 = np.zeros_like(dp)
    a3 = 10 * dp / (T**3)
    a4 = -15 * dp / (T**4)
    a5 = 6 * dp / (T**5)
    return np.vstack((a0, a1, a2, a3, a4, a5))  # shape (6, dof)


def sample_trajectory(coeffs: np.ndarray, t: ArrayLike):
    """
    根据系数和时间序列采样位置、速度、加速度。
    """
    t = np.asarray(t, dtype=float)
    a0, a1, a2, a3, a4, a5 = coeffs

    pos = (
        a0
        + a1 * t[..., None]
        + a2 * t[..., None] ** 2
        + a3 * t[..., None] ** 3
        + a4 * t[..., None] ** 4
        + a5 * t[..., None] ** 5
    )
    vel = (
        a1
        + 2 * a2 * t[..., None]
        + 3 * a3 * t[..., None] ** 2
        + 4 * a4 * t[..., None] ** 3
        + 5 * a5 * t[..., None] ** 4
    )
    acc = (
        2 * a2
        + 6 * a3 * t[..., None]
        + 12 * a4 * t[..., None] ** 2
        + 20 * a5 * t[..., None] ** 3
    )
    return pos, vel, acc


def plan_quintic(
    p0: ArrayLike,
    pT: ArrayLike,
    T: float = 1.0,
    num: int = 100,
):
    """
    生成五次多项式轨迹。

    参数:
      p0: 初始位置数组
      pT: 末端位置数组
      T: 终止时间
      num: 采样点数量

    返回:
      coeffs: (6, dof) 的系数矩阵
      t: 时间数组
      pos, vel, acc: 位置/速度/加速度采样
    """
    coeffs = quintic_coeffs(p0, pT, T)
    t = np.linspace(0, T, num)
    pos, vel, acc = sample_trajectory(coeffs, t)
    return coeffs, t, pos, vel, acc


if __name__ == "__main__":
    # 示例：三关节从 [0, 0, 0] 规划到 [1, -0.5, 0.8]，总时长 2 秒
    coeffs, t, pos, vel, acc = plan_quintic([0, 0, 0], [1, -0.5, 0.8], T=2.0, num=5)
    print("系数 a0..a5:\n", coeffs)
    print("采样时间:", t)
    print("位置采样:\n", pos)
    print("速度采样:\n", vel)
    print("加速度采样:\n", acc)
