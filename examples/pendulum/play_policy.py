"""Load a PPO checkpoint and play it in MuJoCo (use mjpython on macOS)."""
import argparse
import time
from pathlib import Path
import mujoco.viewer
from stable_baselines3 import PPO
from pendulum_env import PendulumEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--episodes', type=int, default=3)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    agent = PPO.load(str(root / 'models' / 'pendulum_ppo.zip'))
    env = PendulumEnv()

    def play_one(viewer=None, ep=0):
        observation, info = env.reset(seed=ep+100)
        if viewer is not None:
            viewer.sync()
        total = 0.
        for _ in range(500):
            action, _ = agent.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if viewer is not None:
                viewer.sync()
                time.sleep(env.model.opt.timestep * 5)
            if terminated or truncated or (viewer is not None and not viewer.is_running()):
                break
        print(f'episode {ep+1}: return={total:.1f}, time={env.data.time:.2f}s')

    if args.headless:
        for ep in range(args.episodes):
            play_one(ep=ep)
    else:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for ep in range(args.episodes):
                if not viewer.is_running():
                    break
                play_one(viewer=viewer, ep=ep)
    env.close()


if __name__ == '__main__':
    main()
