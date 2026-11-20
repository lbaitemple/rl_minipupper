"""Educational ROS 2 <-> MuJoCo cart-pole bridge.

Topics:
  publish   /pendulum/joint_states  sensor_msgs/msg/JointState
  subscribe /pendulum/force         std_msgs/msg/Float64 (newtons)
Run only in a ROS 2 Python environment with mujoco installed.
"""
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / 'examples' / 'pendulum' / 'pendulum.xml'


class PendulumBridge(Node):
    def __init__(self):
        super().__init__('pendulum_mujoco_bridge')
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[1] = 0.15
        mujoco.mj_forward(self.model, self.data)
        self.force = 0.0
        self.last_force_sec = -1e6
        self.pub = self.create_publisher(JointState, '/pendulum/joint_states', 10)
        self.sub = self.create_subscription(Float64, '/pendulum/force', self.on_force, 10)
        self.timer = self.create_timer(0.025, self.step)

    def on_force(self, msg):
        self.force = float(np.clip(msg.data, -20.0, 20.0))
        self.last_force_sec = self.get_clock().now().nanoseconds / 1e9

    def step(self):
        now = self.get_clock().now().nanoseconds / 1e9
        # Time out stale commands after 0.5s so force doesn't persist indefinitely.
        self.data.ctrl[0] = self.force if now - self.last_force_sec <= 0.5 else 0.0
        for _ in range(5):  # 5 * 0.005 s = 0.025 s, matching PendulumEnv.step()
            mujoco.mj_step(self.model, self.data)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['cart_slide', 'pole_hinge']
        msg.position = [float(v) for v in self.data.qpos[:2]]
        msg.velocity = [float(v) for v in self.data.qvel[:2]]
        msg.effort = []
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PendulumBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
