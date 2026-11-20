# Pendulum example — fixed `UnboundLocalError`

The `simulate.py` example imports both `mujoco` and `mujoco.viewer` at the module level.
Do not put `import mujoco.viewer` inside `main()`; that makes `mujoco` a local variable and causes `UnboundLocalError` at `MjModel.from_xml_path(...)`.

## Run from the extracted course root (macOS)

```bash
/path/to/your/mujoco/.venv/bin/mjpython examples/pendulum/simulate.py --viewer --controller pd
```

The `.venv` path should be replaced with your actual Python environment containing MuJoCo.

Headless test:

```bash
/path/to/your/mujoco/.venv/bin/python examples/pendulum/simulate.py --controller pd --seconds 2
```
