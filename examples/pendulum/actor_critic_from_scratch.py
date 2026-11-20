"""See exactly how PPO's actor and critic map observations to outputs.

This is a PyTorch architecture demonstration, NOT a PPO trainer.
Run: python examples/pendulum/actor_critic_from_scratch.py
"""
import torch
from torch import nn
from torch.distributions import Normal


class PendulumActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor maps 4 observations to the Gaussian mean for 1 force action.
        self.actor = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Critic predicts the state-value V(s), one number.
        self.critic = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Learnable exploration standard deviation (in log scale).
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(obs)
        return dist, value


def main():
    torch.manual_seed(0)
    net = PendulumActorCritic()
    obs = torch.tensor([[0.02, 0.10, -0.03, 0.04]], dtype=torch.float32)
    dist, value = net(obs)
    unbounded_action = dist.sample()       # PPO stores action/log-prob for its update
    clipped_action = unbounded_action.clamp(-1.0, 1.0)
    force = 20.0 * clipped_action          # PendulumEnv action scaling (newtons)
    print('observation:', tuple(obs.shape))
    print('mean:', tuple(dist.mean.shape), dist.mean.detach().numpy())
    print('std:', tuple(dist.stddev.shape), dist.stddev.detach().numpy())
    print('unbounded action:', unbounded_action.detach().numpy())
    print('command after clipping:', clipped_action.detach().numpy())
    print('force (N):', force.detach().numpy())
    print('critic value:', tuple(value.shape), value.detach().numpy())
    print('actor parameters (+ log_std):', sum(p.numel() for p in net.actor.parameters()) + net.log_std.numel())
    print('critic parameters:', sum(p.numel() for p in net.critic.parameters()))
    print('TOTAL learnable parameters:', sum(p.numel() for p in net.parameters()))
    # Numbers printed above come from RANDOM initial weights, not a trained policy.


if __name__ == '__main__':
    main()
