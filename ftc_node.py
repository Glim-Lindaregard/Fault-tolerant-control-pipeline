import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3 #251
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import UInt8MultiArray
from tf_transformations import euler_from_quaternion
from rclpy.executors import ExternalShutdownException
from slider_ftc_control.ams.cache import mask_to_index
from slider_ftc_control.config import make_default_config
from slider_ftc_control.ams import build_ams_cache, allocate_wrench
from slider_ftc_control.mpc import MPCController,compute_wrench_bounds
from slider_ftc_control.ams.simplified_cache import build_simplified_ams_cache


# ------------------- SET FAILED THRUSTERS HERE (Outside FTCNode so that the plotting function can reach it) --------------------
START_HEALTH = [1,1,1,1,1,1,1,1]  # [T11, T12, T21, T22, T31, T32, T41, T42]   You can put 0 here to fail a thruster from t=0
FAILURE_TIMES  = [3.0,    None,    None,  None,  None,  None,  None, 6.0]   # What time to inject failure (None to never fail)
FAILURE_STATES = [2,       1,       1,     1,    1,    1,    1,    0] # 0 -> Passive (turned off), 1-> healthy, 2-> active (always on)
# -------------------------------------------------------------------------------------------------------------------------------


class FTCNode(Node):
    def __init__(self):
        super().__init__('ftc_node')
        self.health = START_HEALTH.copy()
        self.failure_times  = FAILURE_TIMES.copy()
        self.failure_states = FAILURE_STATES.copy()



        self.cfg = make_default_config()
        self.ams_cache = build_ams_cache(self.cfg)
        self.simplified_ams_cache = build_simplified_ams_cache(self.ams_cache)

        self.declare_parameter("bounds_mode", "box")  # box | ellipsoid | ams
        self.bounds_mode = self.get_parameter("bounds_mode").value

        valid = {"box", "ellipsoid", "ams"}
        if self.bounds_mode not in valid:
            self.get_logger().warn(f"Invalid bounds_mode='{self.bounds_mode}', defaulting to 'box'")
            self.bounds_mode = "box"

        self.get_logger().info(f"Using bounds_mode='{self.bounds_mode}'")
        self.mpc = MPCController(self.cfg, bounds_mode=self.bounds_mode)  # will be set in _rebuild_mpc()


        #Rebuilds MPC from tuned parameters, sets to mpc0 first time
        #self._rebuild_mpc()

        self.state = None
        self.reference = None

        # --- Hardcoded parameters ---
        self.pwm_frequency = 8
        self.pwm_resolution = 10
        self.max_force = 0.7
        
        #Initialazion
        self.signals = [self.create_pwm(0.0) for _ in range(self.cfg.phys.N_thrusters)]
        self.i = 0
        self.t = 0.0 # For failure activation 
        self.ad_print_counter = 0
        #self.prev_msg = np.zeros((8,1), dtype=int) 
        self.prev_idx = None
        self.failure_detection_timing_buffer = 0.0 #seconds, time between failure injection and when system detects it, can be adjusted for testing purposes but should be set to realistic estimate of detection time for final testing





        # -PWM output timer -
        self.pwm_timer = self.create_timer(
            1.0 / (self.pwm_frequency*self.pwm_resolution),
            self.send_signals,
        )


        # --- control refresh time ---
        
        self.refresh_rate = self.cfg.mpc.Ts
        self.timer = self.create_timer(self.refresh_rate, self.control_step)


        self.failure_already_logged = [False]*self.cfg.phys.N_thrusters

        
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


        if self.state is None and self.reference is None:
            self.get_logger().warning(
                "Skipping control step: state AND reference not yet available."
            )
            return

        if self.state is None:
            self.get_logger().warning(
                "Skipping control step: state not yet available."
            )
            return

        if self.reference is None:
            self.get_logger().warning(
                "Skipping control step: reference not yet available."
            )
            return



        self.t += self.refresh_rate

        health_arr = np.array(self.health, dtype=int)

        # --- Print message if thruster failes ---
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        for i in range(self.cfg.phys.N_thrusters):
            t_fail = self.failure_times[i]
            if t_fail is not None and self.t >= t_fail:
                health_arr[i] = int(self.failure_states[i])
                if not self.failure_already_logged[i]:
                    msg = (
                        f"{RED}{BOLD}"
                        f"*** THRUSTER {i+1} FAILED (code {self.failure_states[i]}) ***"
                        f"{RESET}"
                    )
                    self.get_logger().error(msg)
                    self.failure_already_logged[i] = True


        ams_mask = (health_arr != 0).astype(int)


        # --- MPC bounds for current failure case
        idx = mask_to_index(ams_mask)
        mode = self.ams_cache[idx]


        A_curr = mode.A_mode
        if idx != self.prev_idx:
            A_str = "\n".join(
                ["  [" + "  ".join(f"{val:7.3f}" for val in row) + "]"
                for row in A_curr]
            )

            self.get_logger().warn(
                "\n"
                "========================================\n"
                "  FAILURE MODE CHANGED\n"
                "  New Allocation Matrix A (3x8):\n"
                f"{A_str}\n"
                "========================================"
            )
            self.get_logger().warn("idx = " + str(idx))
            self.prev_idx = idx

  

        u_min = self.cfg.phys.u_min.copy()
        u_max = self.cfg.phys.u_max.copy()

        mask_active = (health_arr == 2).astype(float)          # 1 for stuck-on thrusters
        a_bias = A_curr @ (mask_active * u_max)                     # (3,) bias wrench from stuck thrusters

         

        bounds = compute_wrench_bounds(
            A=A_curr,
            #A = self.cfg.phys.A,
            u_min=u_min,
            u_max=u_max,
            )


        caps = np.array([
            bounds.Fx_max, abs(bounds.Fx_min),
            bounds.Fy_max, abs(bounds.Fy_min),
            bounds.Tau_max, abs(bounds.Tau_min),
        ], dtype=float)

        # avoid divide-by-zero in MPC constraint
        caps = np.maximum(caps, 1e-6)


                # --- Main control step ---
        if self.bounds_mode == "ams":
            planes = np.array([[p.n[0], p.n[1], p.n[2], p.b] for p in self.simplified_ams_cache[idx]], dtype=float)
            a_d = self.mpc.step(self.state, self.reference, bounds=bounds, caps=caps, planes=planes)
        else:
            a_d = self.mpc.step(self.state, self.reference, bounds=bounds, caps=caps) 

        ad_msg = Vector3()
        ad_msg.x = float(a_d[0])
        ad_msg.y = float(a_d[1])
        ad_msg.z = float(a_d[2])
        self.ad_pub.publish(ad_msg) #FOR PLOTTING PURPOSES ONLY, NOT USED FOR CONTROL

        # --- Allocation ---

        a_d_for_alloc = a_d - a_bias  
        try:
            cmd = allocate_wrench(
            a_d=a_d_for_alloc,
            mode=mode,
            cfg=self.cfg,
           )
        except ValueError as e:
            self.get_logger().warn(f"Allocation failed: {e}. Setting all thrusters to 0.")
            cmd = np.zeros(self.cfg.phys.N_thrusters, dtype=float)


        # --- After step, set passive to 0 and active to u_max
        for i, h in enumerate(health_arr):
            if h == 0:
                cmd[i] = 0.0
            elif h == 2:
                cmd[i] = self.cfg.phys.u_max[i]

        # --- Convert continous 'cmd' to PWM'ed 'signals'
        self.signals = [self.create_pwm(float(cmd[i])) for i in range(self.cfg.phys.N_thrusters)]   ## pwm_resx8


        # increment first
        self.ad_print_counter += 1

        deg = 180/np.pi
        self.get_logger().info(
            f"a_d=[{a_d[0]:6.2f}, {a_d[1]:6.2f}, {a_d[2]:6.2f}] | "
            f"idx: {idx:4d} | "
            f"State: [{self.state[0]:6.2f}, {self.state[1]:6.2f}, {deg*self.state[2]:6.2f}] | "
            f"Ref:   [{self.reference[0]:6.2f}, {self.reference[1]:6.2f}, {self.reference[2]:6.2f}] | "
            f"Odom: [{self.x_log:6.2f}, {self.y_log:6.2f}, {deg*self.theta_log:6.2f}]"
            f"bias: [{a_bias[0]:6.2f}, {a_bias[1]:6.2f}, {a_bias[2]:6.2f}]"
            f"allocated: [{a_d_for_alloc[0]:6.2f}, {a_d_for_alloc[1]:6.2f}, {a_d_for_alloc[2]:6.2f}]"
            f"cmd: [{cmd[0]:6.2f}, {cmd[1]:6.2f}, {cmd[2]:6.2f}, {cmd[3]:6.2f}, {cmd[4]:6.2f}, {cmd[5]:6.2f}, {cmd[6]:6.2f}, {cmd[7]:6.2f}]"
        )





    def create_pwm(self, thrust: float) -> list[int]:
        duty = max(0.0, min(1.0, thrust / self.max_force))
        unit = 1.0 / self.pwm_resolution
        if duty < unit:
            number_of_pulses = 0
        else:
            number_of_pulses = int(duty * self.pwm_resolution)
        

        signals = [1] * number_of_pulses + [0] * (self.pwm_resolution - number_of_pulses)
        return signals


    def send_signals(self):

        msg = UInt8MultiArray()
        msg.data = [int(self.signals[i][self.i]) for i in range(self.cfg.phys.N_thrusters)]

        #current_msg= np.array([int(self.signals[i][self.i]) for i in range(self.cfg.phys.N_thrusters)],
        #dtype=int)
        
        #self.prev_msg = current_msg

        self.i = (self.i + 1) % self.pwm_resolution

        self.cmd_pub.publish(msg)


  
    def _hemisphere_w_positive(self,q):
        return -q if q[3] < 0 else q
    
    def _normalize_quat(self, q, eps: float = 1e-12):
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < eps:
            # fallback: identity quaternion (no rotation)
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n
    
            

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
