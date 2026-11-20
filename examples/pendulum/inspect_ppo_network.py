"""Inspect the actual SB3 PPO network instantiated for our MuJoCo cart-pole.

Run from extracted course root (after pendulum dependencies are installed):
    .venv-pendulum/bin/python examples/pendulum/inspect_ppo_network.py

This instantiates an UNTRAINED PPO actor and critic. To inspect a trained
network, add --model examples/pendulum/models/pendulum_ppo.zip
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from pendulum_env import PendulumEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=None, help='Optional trained PPO .zip checkpoint')
    args = ap.parse_args()
    env = PendulumEnv()
    if args.model:
        agent = PPO.load(str(args.model), env=env, device='cpu')
    else:
        agent = PPO('MlpPolicy', env, verbose=0, seed=42,
                    policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]},
                                   'activation_fn': nn.Tanh})
    obs, _ = env.reset(seed=42)
    print('observation:', obs.tolist(), 'shape:', obs.shape)
    print('observation_space:', env.observation_space)
    print('action_space:', env.action_space)
    print('\nActor hidden network:\n', agent.policy.mlp_extractor.policy_net)
    print('\nActor output layer:\n', agent.policy.action_net)
    print('\nAction log standard deviation:', agent.policy.log_std.detach().numpy())
    print('\nCritic hidden network:\n', agent.policy.mlp_extractor.value_net)
    print('\nCritic output layer:\n', agent.policy.value_net)
    print('\nTotal PPO policy parameters:', sum(p.numel() for p in agent.policy.parameters()))
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = agent.policy.get_distribution(t).distribution
        value = agent.policy.predict_values(t)
    action, _ = agent.predict(obs, deterministic=True)
    print('\nGaussian mean:', dist.mean.numpy().tolist(), 'shape:', tuple(dist.mean.shape))
    print('Gaussian std:', dist.stddev.numpy().tolist())
    print('Critic V(obs):', value.numpy().tolist(), 'shape:', tuple(value.shape))
    print('Deterministic action:', np.asarray(action).tolist())
    print('Force (N):', float(20.0 * np.clip(action[0], -1.0, 1.0)))
    print('NOTE: Without --model these values are from untrained weights.')
    env.close()


if __name__ == '__main__':
    main()
