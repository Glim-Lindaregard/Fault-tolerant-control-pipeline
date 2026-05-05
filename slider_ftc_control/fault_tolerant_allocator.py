import rclpy
import numpy as np

import pickle
from pathlib import Path

from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSPresetProfiles

from std_msgs.msg import UInt8MultiArray
from geometry_msgs.msg import Vector3

from slider_ftc_control.config import make_default_config
from slider_ftc_control.ams import build_ams_cache, allocate_wrench
from slider_ftc_control.ams.cache import thruster_state_to_index


class FaultTolerantAllocator(Node):
    def __init__(self):
        super().__init__("allocator_node")

        self.cfg = make_default_config()
        cache_path = Path.home() / "slider_ws" / "ams_cache.pkl"

        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        self.ams_cache = cache["ams_cache"]

        self.declare_parameter("pwm_frequency", 7)
        self.declare_parameter("pwm_resolution", 4)
        self.declare_parameter("max_force", 0.7)

        self.pwm_frequency = int(self.get_parameter("pwm_frequency").value)
        self.pwm_resolution = int(self.get_parameter("pwm_resolution").value)
        self.max_force = float(self.get_parameter("max_force").value)

        self.thruster_state = None
        self.signals = [
            self.create_pwm(0.0)
            for _ in range(self.cfg.phys.N_thrusters)
        ]
        self.i = 0

        self.cmd_pub = self.create_publisher(
            UInt8MultiArray,
            "/eight_thrust_pulse",
            QoSPresetProfiles.get_from_short_key("system_default"),
        )


        self.thruster_state_sub = self.create_subscription(
            UInt8MultiArray,
            "/thruster_state",
            self.thruster_state_callback,
            QoSPresetProfiles.get_from_short_key("system_default"),
        )

        self.thrust_sub = self.create_subscription(
            Vector3,
            "/thrust_cmd",
            self.thrust_cmd_callback,
            QoSPresetProfiles.get_from_short_key("system_default"),
        )

        self.pwm_timer = self.create_timer(
            1.0 / (self.pwm_frequency * self.pwm_resolution),
            self.send_signals,
        )

        self.get_logger().info("Allocator node started.")

    def thruster_state_callback(self, msg: UInt8MultiArray):
        self.thruster_state = np.array(msg.data, dtype=int)


    def thrust_cmd_callback(self, msg: Vector3):
        if self.thruster_state is None:
            self.get_logger().warning(
                "Skipping allocation: thruster_state not received yet."
            )
            return

        a_d = np.array([msg.x, msg.y, msg.z], dtype=float)

        idx = thruster_state_to_index(self.thruster_state)
        mode = self.ams_cache[idx]

        try:
            cmd = allocate_wrench(
                a_d=a_d,
                mode=mode,
                cfg=self.cfg,
            )
        except ValueError as e:
            self.get_logger().warn(
                f"Allocation failed: {e}. Setting all thrusters to 0."
            )
            cmd = np.zeros(self.cfg.phys.N_thrusters, dtype=float)

        for i, h in enumerate(self.thruster_state):
            if h == 0:
                cmd[i] = 0.0
            elif h == 2:
                cmd[i] = self.cfg.phys.u_max[i]

        self.signals = [
            self.create_pwm(float(cmd[i]))
            for i in range(self.cfg.phys.N_thrusters)
        ]

    def create_pwm(self, thrust: float) -> list[int]:
        duty = max(0.0, min(1.0, thrust / self.max_force))

        unit = 1.0 / self.pwm_resolution
        if duty < unit:
            number_of_pulses = 0
        else:
            number_of_pulses = int(duty * self.pwm_resolution)

        return [1] * number_of_pulses + [0] * (
            self.pwm_resolution - number_of_pulses
        )

    def send_signals(self):
        msg = UInt8MultiArray()
        msg.data = [
            int(self.signals[i][self.i])
            for i in range(self.cfg.phys.N_thrusters)
        ]

        self.i = (self.i + 1) % self.pwm_resolution
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaultTolerantAllocator()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()