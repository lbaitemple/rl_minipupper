# PPO neural-network architecture: from observations to actions

This is a companion reference for the interactive HTML course, **Lesson 4: Inside PPO**.

## 1. What determines the network dimensions?

The simulator / environment defines the observation vector and the action vector:

- **Pendulum**: observation = `[cart_position, pole_angle, cart_velocity, pole_angular_velocity]` (4 floats); action = one continuous normalized command (1 float).
- **Mini Pupper 4 flat**: 45 observation floats; 12 action floats.
- **Mini Pupper 4 history policies**: 450 observation floats (10 × 45); 12 action floats.

The network designer picks the **hidden widths, activation functions and whether to share features**. There is no universal layer count prescribed by PPO.

## 2. Actor and critic

Our teaching example uses **two independent multilayer perceptrons** (MLPs):

| Branch | Layer sequence | Last output | Purpose |
|---|---|---|---|
| Actor | `Linear(4,64) → Tanh → Linear(64,64) → Tanh → Linear(64,1)` | 1 mean | Propose a probability distribution over cart forces |
| Actor `log_std` | 1 learned parameter | 1 standard deviation | Exploration during training |
| Critic | `Linear(4,64) → Tanh → Linear(64,64) → Tanh → Linear(64,1)` | 1 value | Predict expected return `V(obs)` |

For a continuous Gaussian actor, the action during training is sampled from `Normal(mean, std)`. The environment clamps it to `[-1,1]` and applies `force = 20 × clipped_action` newtons. PPO uses action log probabilities and reward-derived advantages to update the actor. The critic is trained to fit estimated returns; it **does not** command the actuator.

**Parameter counts** for this exact from-scratch example:

- Actor: `(4×64+64) + (64×64+64) + (64×1+1) + 1 log_std = 4,546`.
- Critic: `(4×64+64) + (64×64+64) + (64×1+1) = 4,545`.
- Total: **9,091 trainable parameters**.

## 3. PPO is an optimizer, not a layer

The clipped policy objective is commonly written:

`L_clip = E[min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage)]`

where `ratio = pi_new(action|obs) / pi_old(action|obs)`. PPO also uses a value loss and usually an entropy term. A rollout buffer stores observations, actions, log probabilities, rewards, and estimated values. The actor and critic are updated from this collected experience.

## 4. Build and inspect a model

From the extracted course directory, with its `.venv-pendulum` environment installed:

```bash
.venv-pendulum/bin/python examples/pendulum/actor_critic_from_scratch.py
.venv-pendulum/bin/python examples/pendulum/inspect_ppo_network.py
.venv-pendulum/bin/python examples/pendulum/train_ppo.py --steps 30000
.venv-pendulum/bin/python examples/pendulum/inspect_ppo_network.py \
  --model examples/pendulum/models/pendulum_ppo.zip
```

You can use Stable-Baselines3 to choose your own architecture:

```python
from torch import nn
from stable_baselines3 import PPO
from pendulum_env import PendulumEnv

model = PPO(
    "MlpPolicy",
    PendulumEnv(),
    policy_kwargs={
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation_fn": nn.Tanh,
    },
    verbose=1,
)
print(model.policy)
```

The actor and critic hidden-layer widths are explicitly set in `net_arch`, rather than being silently assumed.

## 5. Move to Mini Pupper

An **illustrative** Mini Pupper actor could be `45 → 128 → 128 → 12`. If using 450 history inputs, you could choose `450 → 256 → 256 → 12` as another design experiment. These widths are *not* verified to match the repository's shipped policies.

The upstream policy's *real* tensor shapes can be read from `policies/*/policy.onnx` and its config / manifest. The upstream `src/mp4_sim/contract.py` defines observation ordering and the safe conversion from normalized actions to 12 robot actuator targets.

```bash
# From upstream mini_pupper_4_sim repository root:
.venv/bin/python /path/to/course/examples/pendulum/inspect_mini_pupper_onnx.py policies/flat_omni
.venv/bin/python /path/to/course/examples/pendulum/inspect_mini_pupper_onnx.py policies/stairs
```

The deployed ONNX inference graph may contain only the actor; the critic is generally only necessary during training.

## 6. ROS 2 integration

The bundled educational `pendulum_ros2_bridge.py` publishes `sensor_msgs/JointState` on `/pendulum/joint_states`. `pendulum_ppo_controller.py` subscribes to those observations, builds **the same exact 4-feature vector in the same order** used in training, calls `PPO.predict`, scales the action to newtons, and publishes `std_msgs/Float64` on `/pendulum/force`. The bridge applies the force to `MuJoCo.data.ctrl[0]` and steps physics.

This matches the custom cart-pole environment's `5 × 0.005 s = 0.025 s` policy interval (40 Hz). It is a teaching bridge for simulation, not a robot-safe v4 ROS 2 hardware interface.

## 7. Mini exercises

1. For 32 pendulum observations, what is the actor's input and output tensor shape? *(Answer: `(32,4)` to `(32,1)`.)*
2. Change actor hidden widths from `[64,64]` to `[32,32]`. How many parameters do you have? Run the code and find out.
3. Print the critic output before and after PPO training. Does it directly move the cart? *(No.)*
4. Move from 45 to 450 Mini Pupper observations, keeping 12 actions. Which layer must change? *(The input layer, and potentially other architectural choices.)*
5. Explain why an ROS 2 `JointState` message must be converted to the **same observation order and units** used for PPO training.

## References

- [Mini Pupper 4 simulator](https://github.com/mangdangroboticsclub/mini_pupper_4_sim)
- [Stable-Baselines3 custom policies](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- [MuJoCo Python reference](https://mujoco.readthedocs.io/en/latest/python.html)
