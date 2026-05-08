# SE3 Impedance

Self-contained SE(3) impedance control package and MuJoCo examples.

## Layout

- `src/se3_impedance/`: importable Python package.
- `src/se3_impedance/models/franka_panda/`: bundled MuJoCo XML and mesh assets.
- `examples/`: runnable experiment and Franka test scripts.
- `tests/`: minimal import and bundled model checks.

## Install

```bash
cd /Users/swk-pro/Documents/swk_python_files/iiwa-RNEA-program/SE3_impedance
python3 -m pip install -e .
```

For one-off runs from the checkout, the example scripts also add the local
`src/` directory to `sys.path`, so editable install is convenient but not
required.

## Run Examples

```bash
python3 examples/franka_se3_test.py --headless --max-steps 10
python3 examples/exp1_rendering_dynamics_rotation.py --no-render --duration 1
```

The SE(3) impedance implementation is pure Python (`geometry_impedance_control.py`
and `se3_controller.py`), so this package does not require pybind11-generated
extension modules.
