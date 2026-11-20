"""Education-only MuJoCo Mini Pupper state publisher for ROS 2.

This bridge is NOT the upstream mini_pupper_ros hardware stack and does not
implement the Mini Pupper v4 ONNX observation/action contract.

Topics:
  publish   /mini_pupper/sim_joint_states  sensor_msgs/msg/JointState
  subscribe /mini_pupper/sim_ctrl          std_msgs/msg/Float64MultiArray

Use --enable-control to apply raw MuJoCo actuator controls in SIMULATION ONLY.
These are NOT automatically servo angles or ONNX policy outputs.
"""
import argparse
from pathlib import Path
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class MiniPupperSimBridge(Node):
    def __init__(self, scene: Path, enable_control=False):
        super().__init__('mini_pupper_sim_bridge')
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        mujoco.mj_forward(self.model, self.data)
        self.enable_control = enable_control
        self.target_ctrl = np.array(self.data.ctrl, dtype=float)
        self.updated_at = -1e6
        self.pub = self.create_publisher(JointState, '/mini_pupper/sim_joint_states', 10)
        self.sub = self.create_subscription(Float64MultiArray, '/mini_pupper/sim_ctrl', self.on_ctrl, 10)
        self.timer = self.create_timer(0.02, self.on_timer)
        self.joints = []
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            self.joints.append((mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid),
                                int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])))
        self.get_logger().info(f'Scene: {scene} | joints={len(self.joints)}, actuators={self.model.nu} | raw control={enable_control}')

    def on_ctrl(self, msg):
        if not self.enable_control:
            return
        if len(msg.data) != self.model.nu or not np.all(np.isfinite(msg.data)):
            self.get_logger().warning(f'Expected {self.model.nu} finite raw MuJoCo controls')
            return
        c = np.array(msg.data, dtype=float)
        limited = self.model.actuator_ctrllimited.astype(bool)
        c[limited] = np.clip(c[limited], self.model.actuator_ctrlrange[limited, 0],
                                      self.model.actuator_ctrlrange[limited, 1])
        self.target_ctrl = c
        self.updated_at = self.get_clock().now().nanoseconds / 1e9

    def on_timer(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.enable_control and now - self.updated_at < 0.5:
            self.data.ctrl[:] = self.target_ctrl
        else:
            self.data.ctrl[:] = 0.0
        for _ in range(max(1, int(round(0.02 / self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = [n for n, _, _ in self.joints]
        out.position = [float(self.data.qpos[q]) for _, q, _ in self.joints]
        out.velocity = [float(self.data.qvel[v]) for _, _, v in self.joints]
        self.pub.publish(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--enable-control', action='store_true')
    args = ap.parse_args()
    if not args.scene.is_file():
        ap.error(f'Missing scene XML: {args.scene}')
    rclpy.init()
    node = MiniPupperSimBridge(args.scene, enable_control=args.enable_control)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
