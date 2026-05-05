import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import UInt8MultiArray
from tf_transformations import euler_from_quaternion
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from slider_ftc_control.ams.cache import thruster_state_to_index
from slider_ftc_control.config import make_default_config
from slider_ftc_control.ams import build_ams_cache, allocate_wrench
from slider_ftc_control.mpc import MPCController,compute_wrench_bounds
from slider_ftc_control.ams.simplified_cache import build_simplified_ams_cache


# ------------------- SET FAILED THRUSTERS HERE (Outside FTCNode so that the plotting function can reach it) --------------------
START_HEALTH = [1,1,1,1,1,1,1,1]  # [T11, T12, T21, T22, T31, T32, T41, T42]   You can put 0 here to fail a thruster from t=0
#FAILURE_TIMES  = [7.0,    None,    None,  5.0,  None,  None,  None, 10.0]   # What time to inject failure (None to never fail)
#FAILURE_STATES = [2,       1,       1,     0,    1,    1,    1,    0] # 0 -> Passive (turned off), 1-> healthy, 2-> active (always on)

FAILURE_TIMES  = [None,    None,    None,  None,  None,  None,  None, None]   # What time to inject failure (None to never fail)
FAILURE_STATES = [1,       1,       1,     1,    1,    1,    1,    1] # 0 -> Passive (turned off), 1-> healthy, 2-> active (always on)
# -------------------------------------------------------------------------------------------------------------------------------


class FTCNode(Node):
    def __init__(self):
        super().__init__('ftc_node')
        self.health = START_HEALTH.copy()
        self.failure_times  = FAILURE_TIMES.copy()
        self.failure_states = FAILURE_STATES.copy()


        self.cfg = make_default_config()
        self.ams_cache = build_ams_cache(self.cfg)
        self.simplified_ams_cache = build_simplified_ams_cache(self.cfg, self.ams_cache)

        self.declare_parameter("bounds_mode", "box")  # box | ellipsoid | ams
        self.bounds_mode = self.get_parameter("bounds_mode").value

        
        if self.bounds_mode not in {"box", "ellipsoid", "ams"}:
            self.get_logger().warn(f"Invalid bounds_mode='{self.bounds_mode}', defaulting to 'box'")
            self.bounds_mode = "box"

        self.get_logger().info(f"Using bounds_mode='{self.bounds_mode}'")
        self.mpc = MPCController(self.cfg, bounds_mode=self.bounds_mode)


        
        self.state = 6*[None]
        self.reference = 6*[None]

        self.signals = [self.create_pwm(0.0) for _ in range(self.cfg.phys.N_thrusters)]
        self.i = 0
        self.t = 0.0 # For failure activation 
        self.ad_print_counter = 0
        self.prev_idx = None
        self.failure_already_logged = [False]*self.cfg.phys.N_thrusters


        # -PWM output timer -
        self.pwm_timer = self.create_timer(
            1.0 / (self.cfg.pwm.pwm_frequency*self.cfg.pwm.pwm_resolution),
            self.send_signals,
        )

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

        # - /eight_thrust_pulse pub -
        self.cmd_pub = self.create_publisher(    ## UInt8MultiArray of PWM signals to arduino, on off
            UInt8MultiArray,
            '/eight_thrust_pulse',
            QoSPresetProfiles.get_from_short_key('system_default')
        )

        self.ad_pub = self.create_publisher(   #[fx fy tau] from MPC
            Vector3,
            "/wrench_cmd",
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
        self.ready_sent = False


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

    def control_step(self):
        if not hasattr(self, "_ready_sent"):
            self.node_started_pub.publish(Bool(data=True))
            self.get_logger().info("FTC node ready.")
            self._ready_sent = True


        if not self._telemetry_available():
            return

        
        self.t += self.refresh_rate

        thruster_state = np.array(self.health, dtype=int)

        thruster_state = self._update_thruster_state(thruster_state)
        


        # --- MPC bounds for current failure case
        idx = thruster_state_to_index(thruster_state)
        mode = self.ams_cache[idx]

        self._log_state_on_failure(idx, thruster_state)

        testing_thruster_state = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=int)
        bounds = compute_wrench_bounds(
            self.cfg,
            thruster_state
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

        self._log_ad_for_plotting(a_d)
        self._log_state_on_failure(idx, thruster_state)
        

        # --- Allocation ---
        try:
            cmd = allocate_wrench(
            a_d=a_d,
            mode=mode,
            cfg=self.cfg,
           )
        except ValueError as e:
            self.get_logger().warn(f"Allocation failed: {e}. Setting all thrusters to 0.")
            cmd = np.zeros(self.cfg.phys.N_thrusters, dtype=float)


        # --- Set passive to 0 and active to u_max
        for i, h in enumerate(thruster_state):
            if h == 0:
                cmd[i] = 0.0
            elif h == 2:
                cmd[i] = self.cfg.phys.u_max[i]

        # --- Convert continous 'cmd' to PWM'ed 'signals'
        self.signals = [self.create_pwm(float(cmd[i])) for i in range(self.cfg.phys.N_thrusters)]


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
            f"allocated: [{a_d[0]:6.2f}, {a_d[1]:6.2f}, {a_d[2]:6.2f}]"
            #f"cmd: [{cmd[0]:6.2f}, {cmd[1]:6.2f}, {cmd[2]:6.2f}, {cmd[3]:6.2f}, {cmd[4]:6.2f}, {cmd[5]:6.2f}, {cmd[6]:6.2f}, {cmd[7]:6.2f}]"
        )








    def create_pwm(self, thrust: float) -> list[int]:
        duty = max(0.0, min(1.0, thrust / self.cfg.phys.max_thrust))  # Normalize thrust to [0, 1]
        res = self.cfg.pwm.pwm_resolution
        unit = 1.0 / res
        if duty < unit:
            number_of_pulses = 0
        else:
            number_of_pulses = int(duty * res)
        
        signals = [1] * number_of_pulses + [0] * (res - number_of_pulses)
        return signals


    def send_signals(self):

        msg = UInt8MultiArray()
        msg.data = [int(self.signals[i][self.i]) for i in range(self.cfg.phys.N_thrusters)]

        self.i = (self.i + 1) % self.cfg.pwm.pwm_resolution
        self.cmd_pub.publish(msg)


  
    def _hemisphere_w_positive(self,q):
        return -q if q[3] < 0 else q
    
    def _normalize_quat(self, q, eps: float = 1e-12):
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < eps:
            # fallback: identity quaternion (no rotation)
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n

    def _update_thruster_state(self, thruster_state):
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        for i in range(self.cfg.phys.N_thrusters):
            t_fail = self.failure_times[i]
            if t_fail is not None and self.t >= t_fail:
                thruster_state[i] = int(self.failure_states[i])
                if not self.failure_already_logged[i]:
                    msg = (
                        f"{RED}{BOLD}"
                        f"*** THRUSTER {i+1} FAILED (code {self.failure_states[i]}) ***"
                        f"{RESET}"
                    )
                    self.get_logger().error(msg)
                    self.failure_already_logged[i] = True
        return thruster_state
            
    def _log_state_on_failure(self, idx, thruster_state):
        if idx != self.prev_idx:
            state_str = "[" + ", ".join(str(int(s)) for s in thruster_state) + "]"

            self.get_logger().warn(
                "\n"
                "========================================\n"
                "  FAILURE MODE CHANGED\n"
                f"  Thruster state: {state_str}\n"
                "  (0=passive, 1=healthy, 2=active)\n"
                "========================================"
            )
            self.get_logger().warn(f"idx = {idx}")

            self.prev_idx = idx

    def _log_ad_for_plotting(self, a_d):
        ad_msg = Vector3()
        ad_msg.x = float(a_d[0])
        ad_msg.y = float(a_d[1])
        ad_msg.z = float(a_d[2])
        self.ad_pub.publish(ad_msg)

    def _telemetry_available(self):
        if self.state is None:
            self.get_logger().warning(
                "Skipping control step: state not yet available."
            )
            return False
        else:
                return True

        if self.reference is None:
            self.get_logger().warning(
                "Skipping control step: reference not yet available."
            )
            return False
        else:
            return True

def main(args=None):
    rclpy.init(args=args)
    node = FTCNode()
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
