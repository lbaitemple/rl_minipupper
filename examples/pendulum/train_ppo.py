"""Train an educational PPO model on the custom MuJoCo cart-pole model."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from torch import nn
from stable_baselines3.common.monitor import Monitor
from pendulum_env import PendulumEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60000)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    model_dir = root / 'models'
    model_dir.mkdir(exist_ok=True)
    env = Monitor(PendulumEnv())
    model = PPO('MlpPolicy', env, verbose=1, seed=42,
                n_steps=512, batch_size=64, learning_rate=3e-4,
                policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]},
                               'activation_fn': nn.Tanh})
    model.learn(total_timesteps=args.steps)
    model.save(str(model_dir / 'pendulum_ppo'))
    print('Saved:', model_dir / 'pendulum_ppo.zip')
    env.close()


if __name__ == '__main__':
    main()
