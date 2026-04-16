
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import UInt8MultiArray
from tf_transformations import euler_from_quaternion
from rclpy.executors import ExternalShutdownException
#from slider_ftc_control.ams.cache import mask_to_index
from slider_ftc_control.config import make_default_config
from slider_ftc_control.ams import build_ams_cache, allocate_wrench
from slider_ftc_control.mpc import MPCController,compute_wrench_bounds
from slider_ftc_control.ams.simplified_cache import build_simplified_ams_cache



# ------------------- SET FAILED THRUSTERS HERE (Outside FTCNode so that the plotting function can reach it) --------------------
FAILURE_STATE  = [2,       1,       1,     1,    1,    1,    1,    0]        # 0 -> Passive (turned off), 1-> healthy, 2-> active (always on)
FAILURE_TIMES  = [0.0,    None,    None,  None,  None,  None,  None, 0.0]   # [T11, T12, T21, T22, T31, T32, T41, T42] What time to inject failure (None to never fail)


class FTCNode(Node):
    def __init__(self):
        super().__init__('ftc_node')
        self.failure_times  = FAILURE_TIMES.copy()
        self.failure_states = FAILURE_STATE.copy()
        self.health_arr = np.ones(len(FAILURE_STATE), dtype=int)


        self.S = np.array([
                [2,1,1,1,1,1,1,0], [1,1,0,1,1,1,1,0], [1,1,1,1,1,1,1,1]
            ], dtype=int)

        self.lookup = self._build_lookup()

        self.cfg = make_default_config()
        self.ams_cache = build_ams_cache(self.cfg, self.S)
        self.simplified_ams_cache = build_simplified_ams_cache(self.ams_cache, tol_n=1e-6, tol_b=1e-6)


        #Set up parameter for bounds mode (box, ellipsoid, or ams)
        self.declare_parameter("bounds_mode", "box")  # box | ellipsoid | ams
        self.bounds_mode = self.get_parameter("bounds_mode").value


        #Check if bounds_mode is valid, if not default to box and print warning
        valid = {"box", "ellipsoid", "ams"}
        if self.bounds_mode not in valid:
            self.get_logger().warn(f"Invalid bounds_mode='{self.bounds_mode}', defaulting to 'box'")
            self.bounds_mode = "box"

        self.get_logger().info(f"Using bounds_mode='{self.bounds_mode}'")
        self.mpc = MPCController(self.cfg, bounds_mode=self.bounds_mode)  # will be set in _rebuild_mpc()


        self.state = None
        self.reference = None

        self.pwm_frequency = self.cfg.pwm.pwm_frequency
        self.pwm_resolution = self.cfg.pwm.pwm_resolution

        self.max_force = self.cfg.phys.max_thrust

        
        #Initialazion
        self.signals = [self.create_pwm(0.0) for _ in range(self.cfg.phys.N_thrusters)]
        self.i = 0
        self.t = 0.0 # For failure activation 
        self.prev_idx = -1 # For logging when failure mode changes


        # -PWM output timer -
        self.pwm_timer = self.create_timer(
            1.0 / (self.pwm_frequency*self.pwm_resolution),
            self.send_signals,
        )


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

        # --- Print message if thruster failes ---
        self.health_arr = self._update_failure_states()

        #self.U_curr = self._get_current_control_space_from_lookup()


        #ams_mask = (self.health_arr != 0).astype(int)


        #-------------------------------------------------------------------#

        # --- MPC bounds for current failure case
        idx = 2 #mask_to_index(ams_mask)
        mode = self.ams_cache[idx]
        A_curr = mode.A_mode

        #self.U_curr = mode.facets

        if idx != self.prev_idx:
            self._print_failure_mode_changed(A_curr, idx)

  

        u_min = self.cfg.phys.u_min.copy()
        u_max = self.cfg.phys.u_max.copy()

        mask_active = (self.health_arr == 2).astype(float)          # 1 for stuck-on thrusters
        a_bias = A_curr @ (mask_active * u_max)                     # (3,) bias wrench from stuck thrusters

         

        bounds = compute_wrench_bounds(
            A=A_curr,
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
            a_d=a_d, #_for_alloc,
            mode=mode,
            cfg=self.cfg,
           )
        except ValueError as e:
            self.get_logger().warn(f"Allocation failed: {e}. Setting all thrusters to 0.")
            cmd = np.zeros(self.cfg.phys.N_thrusters, dtype=float)


        #Enforce failed thrusters: set passive (0) to 0, active (2) to max, healthy (1) unchanged
        for i, h in enumerate(self.health_arr):
            if h == 0:
                cmd[i] = 0.0
            elif h == 2:
                cmd[i] = self.cfg.phys.u_max[i]



        #Create PWM signals for each thruster based on cmd and max_force, store in self.signals as list of lists, where each inner list is the PWM signal for that thruster
        self.signals = [self.create_pwm(float(cmd[i])) for i in range(self.cfg.phys.N_thrusters)]   ## pwm_resx8


        #Log telemetry to terminal
        self._log_telemetry(a_d, a_bias, a_d_for_alloc, cmd, idx)





    ############### HELPERS ###############
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

        current_msg= np.array([int(self.signals[i][self.i]) for i in range(self.cfg.phys.N_thrusters)],
        dtype=int)
        
        self.prev_msg = current_msg

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
    

    def _update_failure_states(self):
        #self.health_arr = np.array(self.failure_states, dtype=int)

        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        for i in range(self.cfg.phys.N_thrusters):
            t_fail = self.failure_times[i]

            if t_fail is not None and self.t >= t_fail:
                self.health_arr[i] = int(self.failure_states[i])

                if not self.failure_already_logged[i]:
                    self.get_logger().error(
                        f"{RED}{BOLD}"
                        f"*** THRUSTER {i+1} FAILED (code {self.failure_states[i]}) ***"
                        f"{RESET}"
                    )
                    self.failure_already_logged[i] = True
        return self.health_arr

    def _print_failure_mode_changed(self, A_curr, idx):
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


    def _log_telemetry(self, a_d, a_bias, a_d_for_alloc, cmd, idx):
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


    def _build_lookup(self):
        return np.array([
            [2952,  328, 2952,  328,  328, 2952,  328, 2952],
            [3195, 1057, 2955,  337, 2515, 3033,  355, 2953],
            [5382, 1058, 2982,  418, 2758, 3042,  358, 3682],
            [3276, 3244, 2956,  364, 3244, 3276,  364, 2956],
            [5463, 3245, 2983,  445, 3487, 3285,  367, 3685],
            [5472, 3488, 3712,  448, 3488, 5472,  448, 3712],
            [3198, 1066, 3198, 1066, 2542, 3034, 2542, 3034],
            [5385, 1067, 3225, 1147, 2785, 3043, 2545, 3763],
            [3036, 2524, 3196, 1084, 1084, 3196, 2524, 3036],
            [2550, 3250, 3190,  850, 3190, 3250, 2550,  850],
            [3279, 3253, 3199, 1093, 3271, 3277, 2551, 3037],
            [5466, 3254, 3226, 1174, 3514, 3286, 2554, 3766],
            [5412, 1148, 5412, 1148, 2788, 3772, 2788, 3772],
            [3063, 2605, 5383, 1085, 1087, 3925, 2767, 3045],
            [3306, 3334, 5386, 1094, 3274, 4006, 2794, 3046],
            [5493, 3335, 5413, 1175, 3517, 4015, 2797, 3775],
            [3288, 3496, 3928, 1096, 3272, 5464, 2632, 3064],
            [5475, 3497, 3955, 1177, 3515, 5473, 2635, 3793],
            [3072, 2848, 6112, 1088, 1088, 6112, 2848, 3072],
            [3315, 3577, 6115, 1097, 3275, 6193, 2875, 3073],
            [5502, 3578, 6142, 1178, 3518, 6202, 2878, 3802],
            [3280, 3280, 3280, 3280, 3280, 3280, 3280, 3280],
            [5467, 3281, 3307, 3361, 3523, 3289, 3283, 4009],
            [6196, 3284, 3316, 3604, 3604, 3316, 3284, 6196],
            [5710, 4010, 3310, 3370, 5710, 3370, 3310, 4010],
            [5494, 3362, 5494, 3362, 3526, 4018, 3526, 4018],
            [4036, 3364, 5476, 3524, 3364, 4036, 3524, 5476],
            [6223, 3365, 5503, 3605, 3607, 4045, 3527, 6205],
            [6232, 3608, 6232, 3608, 3608, 6232, 3608, 6232],
        ], dtype=int)
            


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
