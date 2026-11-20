# Mini Pupper 4 Reinforcement Learning — Pendulum-first edition

Open `index.html` in Chrome/Safari/Firefox. Click any **native HTML** lesson link in the left navigation. JavaScript is only used for copy buttons and highlighting; the links work without it.

## Suggested order

1. Inverted pendulum and MuJoCo basics (`examples/pendulum/`)
2. PPO with Stable-Baselines3 (`examples/pendulum/train_ppo.py`)
3. Inspect actor/critic architecture and tensor shapes (`examples/pendulum/actor_critic_from_scratch.py`, `inspect_ppo_network.py`)
4. ROS 2 bridge and PPO actor node on Ubuntu (`ros2/`)
5. Mini Pupper scene, observations, PPO recipe, and ONNX deployment (remaining lessons)

## Two notebooks

- `notebooks/Pendulum_First_Colab.ipynb`: self-contained, CPU-friendly cart-pole tutorial.
- `notebooks/MiniPupper4_RL_Colab.ipynb`: original advanced repository training lab; NVIDIA GPU driver >= 580 required by upstream. Colab may not provide a compatible host.

## Important ROS 2 boundaries

The `ros2/` examples are **new educational bridges**, not part of the upstream Mini Pupper v4 repo. ROS 2 Jazzy is supported on Ubuntu 24.04 and ROS 2 Humble on Ubuntu 22.04. For v4 real-robot control, validate the robot's firmware, mapping, actuator limits, and ONNX policy contract; do not send raw MuJoCo controls to hardware.

## Sources
- Mini Pupper 4 simulator: https://github.com/mangdangroboticsclub/mini_pupper_4_sim
- Mini Pupper ROS 2 stack: https://github.com/mangdangroboticsclub/mini_pupper_ros/tree/ros2-dev
- Official MuJoCo Python: https://mujoco.readthedocs.io/en/latest/python.html
- Gymnasium MuJoCo InvertedPendulum: https://gymnasium.farama.org/main/environments/mujoco/inverted_pendulum/

The bundled PDF is the earlier six-section reference guide and has not been rewritten for the new pendulum/ROS 2 chapters.

For a standalone explanation of PPO tensor shapes, see `docs/PPO_NEURAL_NETWORK_EXPLAINED.md`.

## New neural-network lesson

- The tutorial now has **13 clickable lessons**, including **Inside PPO: observation → actor / critic → action**.
- Visuals: `assets/ppo_actor_critic_architecture.svg`, `assets/ppo_pendulum_to_minipupper.svg`, `assets/ppo_ros2_signal_chain.svg`.
- Run `examples/pendulum/actor_critic_from_scratch.py` to see the 4 → 64 → 64 → 1 actor and critic with PyTorch.
- Run `examples/pendulum/inspect_ppo_network.py` to inspect an SB3 policy, or `--model examples/pendulum/models/pendulum_ppo.zip` after training.
- Inspect upstream ONNX input/output with `examples/pendulum/inspect_mini_pupper_onnx.py`. The illustrated Mini Pupper hidden layers are examples, not asserted to match the shipped policy.
- The two Colab notebooks now include neural-network tensor/architecture inspection cells.
- To run the ROS 2 policy node, install `stable-baselines3` into the ROS 2 Python environment and train or copy the saved pendulum checkpoint first.
- The ROS 2 pendulum example now includes a PPO controller node (`ros2/pendulum_ppo_controller.py`) that builds the trained observation vector from JointState messages and sends force commands back to the MuJoCo bridge. Bridge updates run at 40 Hz, matching five 0.005-s physics steps per action.

**Simulation-only:** The custom ROS 2 bridges are educational examples, not official Mini Pupper ROS 2 packages or a validated hardware controller.

## Pendulum viewer fix (November 20, 2026)
`examples/pendulum/simulate.py` imports `mujoco.viewer` at module scope to avoid Python treating `mujoco` as an uninitialized local variable inside `main()`.
