import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray


class ThrusterStateNode(Node):
    """
    Publishes current thruster health state.

    Convention:
        0 = passive failure / forced OFF
        1 = healthy
        2 = active failure / stuck ON

    Topic:
        /thruster_state : std_msgs/UInt8MultiArray
    """

    def __init__(self):
        super().__init__("thruster_state_node")

        self.declare_parameter("publish_rate", 10.0)

        self.declare_parameter(
            "start_health",
            [1, 1, 1, 1, 1, 1, 1, 1],
        )

        # Use -1 for "never fails"
        self.declare_parameter(
            "failure_times",
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        )

        self.declare_parameter(
            "failure_states",
            [1, 1, 1, 1, 1, 1, 1, 1],
        )

        self.start_health = list(
            self.get_parameter("start_health").get_parameter_value().integer_array_value
        )
        self.failure_times = list(
            self.get_parameter("failure_times").get_parameter_value().double_array_value
        )
        self.failure_states = list(
            self.get_parameter("failure_states").get_parameter_value().integer_array_value
        )


        self.thruster_state = self.start_health.copy()
        self.failure_logged = [False] * 8
        self.t0 = self.get_clock().now()

        self.pub = self.create_publisher(
            UInt8MultiArray,
            "/thruster_state",
            10,
        )

        publish_rate = float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

        self.get_logger().info(
            f"ThrusterStateNode started with initial state: {self.thruster_state}"
        )

    def timer_callback(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        for i in range(8):
            t_fail = self.failure_times[i]

            if t_fail < 0.0:
                continue

            if t >= t_fail:
                self.thruster_state[i] = int(self.failure_states[i])

                if not self.failure_logged[i]:
                    self.get_logger().error(
                        f"THRUSTER {i + 1} FAILED at t={t:.2f}s "
                        f"-> state {self.thruster_state[i]}"
                    )
                    self.failure_logged[i] = True

        msg = UInt8MultiArray()
        msg.data = [int(x) for x in self.thruster_state]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ThrusterStateNode.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()