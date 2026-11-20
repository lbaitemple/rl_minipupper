"""Educational ROS 2 node: feed MuJoCo joint states into a trained PPO actor.

Bridge publishes: /pendulum/joint_states (sensor_msgs/JointState)
This node outputs: /pendulum/force (std_msgs/Float64), in newtons.

Run in a ROS 2 Python environment containing stable-baselines3, torch and
rclpy. Controller and bridge have a ~40 Hz loop (5 * 0.005 s each update).
This example is simulation-only; never send this force command to a robot.
"""
import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class PendulumPPOController(Node):
    def __init__(self, checkpoint: Path):
        super().__init__('pendulum_ppo_controller')
        self.agent = PPO.load(str(checkpoint), device='cpu')
        self.pub = self.create_publisher(Float64, '/pendulum/force', 10)
        self.sub = self.create_subscription(
            JointState, '/pendulum/joint_states', self.on_joint_states, 10)
        self.counter = 0
        self.get_logger().info(f'Loaded PPO actor: {checkpoint}')

    def on_joint_states(self, msg):
        named = {name: i for i, name in enumerate(msg.name)}
        if 'cart_slide' not in named or 'pole_hinge' not in named:
            self.get_logger().warning('Waiting for cart_slide and pole_hinge')
            return
        c, p = named['cart_slide'], named['pole_hinge']
        if len(msg.velocity) <= max(c, p) or len(msg.position) <= max(c, p):
            return
        # Exact four observations and order used by PendulumEnv._obs().
        obs = np.array([msg.position[c], msg.position[p],
                        msg.velocity[c], msg.velocity[p]], dtype=np.float32)
        if not np.isfinite(obs).all():
            return
        normalized_action, _ = self.agent.predict(obs, deterministic=True)
        force_newtons = 20.0 * float(np.clip(normalized_action[0], -1.0, 1.0))
        cmd = Float64()
        cmd.data = force_newtons
        self.pub.publish(cmd)
        self.counter += 1
        if self.counter % 80 == 0:
            self.get_logger().info(f'obs={np.round(obs, 3).tolist()}, force={force_newtons:+.2f}N')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    args = ap.parse_args()
    if not args.policy.is_file():
        ap.error(f'No model file: {args.policy}. Train it first.')
    rclpy.init()
    node = PendulumPPOController(args.policy)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
