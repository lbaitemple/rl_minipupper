"""Pendulum experiment using MuJoCo directly.

Usage (Mac):  .venv/bin/mjpython simulate.py --viewer --controller pd
Usage (headless): .venv/bin/python simulate.py --controller pd
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_FILE = Path(__file__).with_name('pendulum.xml')


def compute_pd(data):
    x, theta = map(float, data.qpos[:2])
    vx, omega = map(float, data.qvel[:2])
    # One possible upright balance controller (linear around theta=0).
    force = 65.0 * theta + 14.0 * omega - 2.0 * x - 5.0 * vx
    return float(np.clip(force, -20.0, 20.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--controller', choices=['none', 'pd'], default='pd')
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--seconds', type=float, default=12.0)
    p.add_argument('--angle', type=float, default=0.14, help='initial pole lean (radians)')
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_FILE))
    data = mujoco.MjData(model)
    data.qpos[1] = args.angle
    mujoco.mj_forward(model, data)

    def run(viewer=None):
        count = 0
        while data.time < args.seconds and (viewer is None or viewer.is_running()):
            started = time.perf_counter()
            data.ctrl[0] = compute_pd(data) if args.controller == 'pd' else 0.0
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                delay = model.opt.timestep - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
            count += 1
            if count % 200 == 0:
                print(f't={data.time:.1f}s x={data.qpos[0]:+.3f}m '
                      f'theta={data.qpos[1]:+.3f}rad force={data.ctrl[0]:+.1f}N')
    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(viewer)
    else:
        run()


if __name__ == '__main__':
    main()
