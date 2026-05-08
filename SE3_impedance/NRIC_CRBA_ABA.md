# NRIC、CRBA、ABA 使用说明

## 仓库作用

这个仓库实现了一个基于 `SE(3)` 的几何阻抗控制器，并提供 Franka Panda 机械臂的 MuJoCo 仿真示例。

控制器参考的文献是：

`Impedance_Control_Design_Framework_Using_Commutative_Map_Between_SE3_and_mathfrak_se3.pdf`

主要代码结构如下：

- `src/se3_impedance/geometry_impedance_control.py`：实现 `SE(3)` 和 `se(3)` 相关的几何运算，包括指数映射、对数映射、伴随变换、`dexp` 及其微分等。
- `src/se3_impedance/se3_controller.py`：实现 `SE3ImpedanceController`，把末端位姿误差和速度误差转换为关节力矩。
- `src/se3_impedance/models/franka_panda/`：存放 Franka Panda 的 MuJoCo 模型文件。
- `examples/`：存放可以直接运行的仿真示例。

## 主要 Franka 示例

### `examples/franka_se3_test.py`

这个文件是无摩擦情况下的 Franka `SE(3)` 阻抗控制示例。

在无摩擦仿真中，控制器可以把末端执行器驱动到期望平衡位置，因此稳态误差理论上可以收敛到 0。

### `examples/franka_se3_test_friction.py`

这个文件是带摩擦情况下的 Franka `SE(3)` 阻抗控制示例。目前主要是在 MuJoCo 中给 `joint7` 加入 dry friction / `frictionloss`。

可以自由设置 NRIC 的启用与否；在 Mujoco 可视化中拖动末端执行器处的小球（小球很小，需要自己找一下），观察开启和关闭 NRIC 时的控制效果差异。

在有干摩擦的情况下，纯 `SE(3)` 阻抗控制器可能无法完全抹平稳态误差。原因是关节静摩擦会抵消一部分控制力矩，使机器人在误差还没有完全归零时就停住。

因此，这个示例中引入了 NRIC 控制器：

`examples/nric_torque_controller.py`

NRIC 控制器内部维护一个无摩擦的名义机器人模型，然后比较名义模型和真实 MuJoCo 机器人的状态：

```python
e_nr = q_nominal - q
e_nr_dot = dq_nominal - dq
```

再根据这个误差生成辅助力矩：

```python
tau = tau_nominal_mujoco + tau_aux
```

其中：

- `tau_nominal_mujoco` 是基于当前 MuJoCo 真实状态计算得到的原始 `SE(3)` 阻抗控制力矩。
- `tau_aux` 是 NRIC 根据名义模型误差生成的补偿力矩。
- 最终下发给 MuJoCo 机器人的控制力矩是两者之和。

## CRBA 算法用在哪里

CRBA 当前用在：

`examples/nric_torque_controller.py`

在 `PinocchioNominalSE3Controller.compute_torque()` 中，目前通过 Pinocchio 调用：

```python
M = np.asarray(self.pin.crba(self.model, self.data, q), dtype=float)
```

这一步计算的是名义机器人在当前构型 `q` 下的关节空间质量矩阵：

```python
M(q)
```

这个质量矩阵随后会传入：

`src/se3_impedance/se3_controller.py`

在 `SE3ImpedanceController.compute_torque()` 中，质量矩阵会被用于计算：

- `M_inv`
- 操作空间惯量相关项
- 动态一致伪逆
- 零空间投影力矩

也就是说，当前 NRIC 的名义模型要计算 `tau_nominal_pin`，就需要质量矩阵 `M(q)`，因此需要 CRBA。

## ABA 算法用在哪里

ABA 当前也用在：

`examples/nric_torque_controller.py`

在 `PinocchioNominalSimulator.step()` 中，目前通过 Pinocchio 调用：

```python
self.ddq = np.asarray(
    self.pin.aba(self.model, self.data, self.q, self.dq, tau_effective, self.fext),
    dtype=float,
)
```

这一步计算的是无摩擦名义机器人在当前状态和输入力矩下的关节加速度：

```python
ddq_nominal = ABA(q_nominal, dq_nominal, tau_nominal_pin, f_ext)
```

然后名义仿真器会对名义状态做积分：

```python
dq_nominal += dt * ddq_nominal
q_nominal = integrate(q_nominal, dt * dq_nominal)
```

因此，ABA 的作用是推进 NRIC 内部的无摩擦名义机器人模型。如果没有这一步，`q_nominal` 和 `dq_nominal` 就不会按照无摩擦动力学演化，NRIC 也就无法构造有意义的“名义模型 vs 真实带摩擦机器人”的误差。

## 当前依赖

目前 CRBA 和 ABA 都来自 Pinocchio：

- `pin.crba(...)`：计算质量矩阵 `M(q)`
- `pin.aba(...)`：计算前向动力学加速度 `ddq`

因此，当前 `examples/nric_torque_controller.py` 中的 NRIC 控制器仍然依赖 Pinocchio 来完成这两部分动力学计算。

## 接下来需要做的工作

下一步需要把这里的 Pinocchio 实现替换为仓库中的手写算法实现。

具体来说，需要用手写 CRBA 和 ABA 替换：

```python
self.pin.crba(...)
self.pin.aba(...)
```

替换后的手写算法需要保持当前接口和约定不变：

- 固定基座 7 自由度 Franka 模型
- 关节顺序和当前 MuJoCo / Pinocchio 模型一致
- 重力方向和大小通过 `gravity_z` 保持一致
- 末端外力使用当前局部坐标系表达，格式为 `[fx, fy, fz, mx, my, mz]`
- 名义模型仍然表示无摩擦机器人动力学
- CRBA 输出 `7 x 7` 的关节空间质量矩阵 `M(q)`
- ABA 输出 `7` 维关节加速度 `ddq`

建议的验证流程：

1. 运行 `examples/franka_se3_test.py`，确认无摩擦情况下控制器稳定，并且稳态误差可以收敛到 0。
2. 运行 `examples/franka_se3_test_friction.py`，关闭 NRIC，确认有干摩擦时会出现无法完全抹平的稳态误差。
3. 运行 `examples/franka_se3_test_friction.py`，开启 NRIC，确认辅助力矩 `tau_aux` 能够减小或消除干摩擦造成的稳态误差。
4. 在开发手写 CRBA / ABA 时，先和 Pinocchio 的 `crba` / `aba` 输出做数值对比，直到误差足够小，再替换到 NRIC 控制器中。
