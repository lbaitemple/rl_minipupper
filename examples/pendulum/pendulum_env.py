"""Small Gymnasium environment: 4 observations, 1 continuous force action.

This is an educational custom environment, not the built-in Gymnasium one.
"""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class PendulumEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('pendulum.xml')))
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self._steps = 0

    def _obs(self):
        # cart position, pole angle, cart speed, pole angular speed
        return np.array([self.data.qpos[0], self.data.qpos[1],
                         self.data.qvel[0], self.data.qvel[1]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[1] = self.np_random.uniform(-0.15, 0.15)
        self.data.qvel[:2] = self.np_random.uniform(-0.05, 0.05, size=2)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        value = float(np.clip(np.asarray(action).reshape(-1)[0], -1., 1.))
        force = 20.0 * value
        self.data.ctrl[0] = force
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        x, theta, vx, omega = self._obs()
        # Example dense reward, not the Mini Pupper reward function.
        reward = 1.0 - 2.0 * float(theta**2) - 0.05 * float(x**2) - 0.001 * force**2
        terminated = bool(abs(theta) > 0.7 or abs(x) > 2.3)
        truncated = bool(self._steps >= 500)
        return self._obs(), float(reward), terminated, truncated, {'force_newtons': force}

    def close(self):
        pass
