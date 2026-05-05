import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
import pickle
from pathlib import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import UInt8MultiArray
from tf_transformations import euler_from_quaternion
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from slider_ftc_control.ams.cache import thruster_state_to_index
from slider_ftc_control.config import make_default_config
from slider_ftc_control.ams import build_ams_cache
from slider_ftc_control.mpc import MPCController,compute_wrench_bounds
from slider_ftc_control.ams.simplified_cache import build_simplified_ams_cache



class FaultTolerantController(Node):
    def __init__(self):
        super().__init__('control_node')
  
        self.cfg = make_default_config()
        cache_path = Path.home() / "slider_ws" / "ams_cache.pkl"

        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        self.ams_cache = cache["ams_cache"]
        self.simplified_ams_cache = cache["simplified_ams_cache"]
        #self.ams_cache = build_ams_cache(self.cfg)
        #self.simplified_ams_cache = build_simplified_ams_cache(self.cfg, self.ams_cache)

        self.declare_parameter("bounds_mode", "box")  # box | ellipsoid | ams
        self.bounds_mode = self.get_parameter("bounds_mode").value

        
        if self.bounds_mode not in {"box", "ellipsoid", "ams"}:
            self.get_logger().warn(f"Invalid bounds_mode='{self.bounds_mode}', defaulting to 'box'")
            self.bounds_mode = "box"

        self.get_logger().info(f"Using bounds_mode='{self.bounds_mode}'")
        self.mpc = MPCController(self.cfg, bounds_mode=self.bounds_mode)


        
        self.state = None
        self.reference = None
        self.thruster_state = None

        self.ad_print_counter = 0


        # --- control refresh time ---
        self.refresh_rate = self.cfg.mpc.Ts
        self.timer = self.create_timer(self.refresh_rate, self.control_step)
        
        # --- Define subscribers and publishers ---

        # - \odom sub -
        self.state_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            QoSPresetProfiles.get_from_short_key('system_default')
        )

        # - \target_point sub - 
        self.ref_sub = self.create_subscription(
            Odometry,
            '/target_point',
            self.target_point_callback,
            QoSPresetProfiles.get_from_short_key('system_default')
        )

        self.ad_pub = self.create_publisher(   #[fx fy tau] from MPC
            Vector3,
            "/thrust_cmd",
            QoSPresetProfiles.get_from_short_key('system_default')
        )

        self.thruster_state_sub = self.create_subscription(
            UInt8MultiArray,
            "/thruster_state",
            self.thruster_state_callback,
            QoSPresetProfiles.get_from_short_key('system_default')
        )

        

        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.node_started_pub = self.create_publisher(
            Bool,
            "/ftc_node_started",
            qos
        )


        self.get_logger().info('FTCNode started, waiting for /odom and /target_point...')
        

        
    def odom_callback(self, msg: Odometry):

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        #theta = msg.pose.pose.orientation.z

        self.x_log = x
        self.y_log = y
        

        omega = float(msg.twist.twist.angular.z)
        vx_b = float(msg.twist.twist.linear.x)
        vy_b = float(msg.twist.twist.linear.y)

        q_msg = msg.pose.pose.orientation
        q = np.array([q_msg.x, q_msg.y, q_msg.z, q_msg.w], dtype=float)
        q = self._normalize_quat(q,eps=1e-12) #q / (np.linalg.norm(q) + 1e-12)
        q = self._hemisphere_w_positive(q)

        _, _, theta = euler_from_quaternion([q[0], q[1], q[2], q[3]])
        self.theta_log = theta
        self.state = np.array([x,y,theta,vx_b,vy_b,omega], dtype=float)



    def target_point_callback(self, msg: Odometry):

        self.x_ref = float(msg.pose.pose.position.x)
        self.y_ref = float(msg.pose.pose.position.y)

        q_msg = msg.pose.pose.orientation
        q = np.array([q_msg.x, q_msg.y, q_msg.z, q_msg.w], dtype=float)
        q = self._hemisphere_w_positive(q)
        q = self._normalize_quat(q, eps=1e-12)
        _, _, self.theta_ref = euler_from_quaternion([q[0], q[1], q[2], q[3]])


        vx_ref = float(msg.twist.twist.linear.x)
        vy_ref = float(msg.twist.twist.linear.y)
        omega_ref = float(msg.twist.twist.angular.z)

        self.reference = np.array(
            [self.x_ref, self.y_ref, self.theta_ref, vx_ref, vy_ref, omega_ref],
            dtype=float
        )


    def thruster_state_callback(self, msg: UInt8MultiArray):
        data = msg.data
        self.thruster_state = np.array(data, dtype=int)
     



    def control_step(self):
        if not hasattr(self, "ready_sent"):
            self.node_started_pub.publish(Bool(data=True))
            self.get_logger().info("FTC node ready.")
            self.ready_sent = True


        if not self._telemetry_available():
            self.get_logger().warning("Skipping control step: telemetry not yet available.")
            return

        
        # --- MPC bounds for current failure case
        idx = thruster_state_to_index(self.thruster_state)
        mode = self.ams_cache[idx]


        bounds = compute_wrench_bounds(
            self.cfg,
            self.thruster_state
            )

        #self.get_logger().info(
        #    f"Bounds | "
        #    f"Fx:[{bounds.Fx_min:.2f}, {bounds.Fx_max:.2f}] "
        #    f"Fy:[{bounds.Fy_min:.2f}, {bounds.Fy_max:.2f}] "
        #    f"Tau:[{bounds.Tau_min:.2f}, {bounds.Tau_max:.2f}]"
        #)

        # --- Main control step ---
        if self.bounds_mode == "ams":
            planes = np.array([[p.n[0], p.n[1], p.n[2], p.b] for p in self.simplified_ams_cache[idx]], dtype=float)
            a_d = self.mpc.step(self.state, self.reference, bounds=bounds, planes=planes)
        else:
            a_d = self.mpc.step(self.state, self.reference, bounds=bounds) 



        ad_msg = Vector3()
        ad_msg.x = float(a_d[0])
        ad_msg.y = float(a_d[1])
        ad_msg.z = float(a_d[2])
        self.ad_pub.publish(ad_msg)


        # LOG TO TERMINAL
        self.ad_print_counter += 1
        deg = 180/np.pi
        if self.ad_print_counter % 10 == 0:
            self.get_logger().info(
            f"a_d=[{a_d[0]:6.2f}, {a_d[1]:6.2f}, {a_d[2]:6.2f}] | "
            f"idx: {idx:4d} | "
            f"State: [{self.state[0]:6.2f}, {self.state[1]:6.2f}, {deg*self.state[2]:6.2f}] | "
            #f"Ref:   [{self.reference[0]:6.2f}, {self.reference[1]:6.2f}, {self.reference[2]:6.2f}] | "
            f"Odom: [{self.x_log:6.2f}, {self.y_log:6.2f}, {deg*self.theta_log:6.2f}]"
            #f"bias: [{a_bias[0]:6.2f}, {a_bias[1]:6.2f}, {a_bias[2]:6.2f}]"
            f"wrench: [{a_d[0]:6.2f}, {a_d[1]:6.2f}, {a_d[2]:6.2f}]"
            #f"cmd: [{cmd[0]:6.2f}, {cmd[1]:6.2f}, {cmd[2]:6.2f}, {cmd[3]:6.2f}, {cmd[4]:6.2f}, {cmd[5]:6.2f}, {cmd[6]:6.2f}, {cmd[7]:6.2f}]"
        )



  
    def _hemisphere_w_positive(self,q):
        return -q if q[3] < 0 else q
    
    def _normalize_quat(self, q, eps: float = 1e-12):
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < eps:
            # fallback: identity quaternion (no rotation)
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n


    def _telemetry_available(self):
        if self.state is None:
            self.get_logger().warning("Skipping control step: state not yet available.")
            return False

        if self.reference is None:
            self.get_logger().warning("Skipping control step: reference not yet available.")
            return False

        if self.thruster_state is None:
            self.get_logger().warning("Skipping control step: thruster_state not yet available.")
            return False

        return True

def main(args=None):
    rclpy.init(args=args)
    node = FaultTolerantController()
    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        # Clean, no scary traceback on Ctrl-C or external shutdown
        pass

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
