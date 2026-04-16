import os
import csv
import rclpy
import numpy as np
from rclpy.node import Node
from dataclasses import replace
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3 #251
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import UInt8MultiArray
from rcl_interfaces.msg import SetParametersResult
from tf_transformations import euler_from_quaternion
from rclpy.executors import ExternalShutdownException
from slider_ftc_control.ams.cache import mask_to_index
from slider_ftc_control.config import make_default_config
from slider_ftc_control.ams import build_ams_cache, allocate_wrench
from slider_ftc_control.mpc import MPCController,compute_wrench_bounds
from slider_ftc_control.ams.simplified_cache import Plane, build_simplified_ams_cache


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

        self.S = np.array([
                [2,1,1,1,1,1,1,1],[2,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1], [1,1,0,1,1,1,1,1], [1,1,0,1,1,1,1,0]
            ], dtype=int)
            
        self.t_failure = np.array([3.0, 6.0], dtype=float)  # for switching AMS modes in MPC, should match FAILURE_TIMES for consistency

        self.cfg = make_default_config()
        self.ams_cache = build_ams_cache(self.cfg, self.S)
        self.simplified_ams_cache = build_simplified_ams_cache(self.ams_cache, tol_n=1e-6, tol_b=1e-6)

        self.declare_parameter("bounds_mode", "box")  # box | ellipsoid | ams
        self.bounds_mode = self.get_parameter("bounds_mode").value

        valid = {"box", "ellipsoid", "ams"}
        if self.bounds_mode not in valid:
            self.get_logger().warn(f"Invalid bounds_mode='{self.bounds_mode}', defaulting to 'box'")
            self.bounds_mode = "box"

        self.get_logger().info(f"Using bounds_mode='{self.bounds_mode}'")
        self.mpc = MPCController(self.cfg, bounds_mode=self.bounds_mode)  # will be set in _rebuild_mpc()

        """self.declare_parameter("plot_ams_planes", False)
        if self.get_parameter("plot_ams_planes").get_parameter_value().bool_value:
            self.get_logger().warn("PLOTTING AMS PLANES NOW")
            self._save_planes_to_file()"""


        #Live tunable parameters
        self.declare_parameter('q_x',     float(self.cfg.mpc.Q[0, 0]))
        self.declare_parameter('q_y',     float(self.cfg.mpc.Q[1, 1]))
        self.declare_parameter('q_theta', float(self.cfg.mpc.Q[2, 2]))
        self.declare_parameter('q_vx',    float(self.cfg.mpc.Q[3, 3]))
        self.declare_parameter('q_vy',    float(self.cfg.mpc.Q[4, 4]))
        self.declare_parameter('q_r',     float(self.cfg.mpc.Q[5, 5]))
        self.declare_parameter('r_force', float(self.cfg.mpc.R[0, 0]))
        self.declare_parameter('r_tau',   float(self.cfg.mpc.R[2, 2]))
        self.declare_parameter('N',   int(self.cfg.mpc.N))   # horizon length
        self.declare_parameter('pwm_frequency',8)   #1 2 3 4 6 7 8 9 10 12
        self.declare_parameter('pwm_resolution',10)  #1 2 3 4 5 10 20
        self.declare_parameter('max_force', 0.7)

        self.add_on_set_parameters_callback(self._on_params)

        #Rebuilds MPC from tuned parameters, sets to mpc0 first time
        self._rebuild_mpc()

        self.state = None
        self.reference = None
        
        self.pwm_frequency = self.get_parameter('pwm_frequency').get_parameter_value().integer_value
        self.pwm_resolution = self.get_parameter('pwm_resolution').get_parameter_value().integer_value
        self.max_force = self.get_parameter('max_force').get_parameter_value().double_value
        
        #Initialazion
        self.signals = [self.create_pwm(0.0) for _ in range(self.cfg.phys.N_thrusters)]
        self.i = 0
        self.t = 0.0 # For failure activation 
        self.ad_print_counter = 0
        self.click_counter = 0
        self.stop = False
        self.prev_msg = np.zeros((8,1), dtype=int) 
        self.sum_err_x = 0.0
        self.sum_err_theta = 0.0
        self.err_samples = 0
        self.eval_logged = False
        self.sim_stop_time = 5000
        self.prev_idx = None
        self.failure_detection_timing_buffer = 0.0 #seconds, time between failure injection and when system detects it, can be adjusted for testing purposes but should be set to realistic estimate of detection time for final testing


        # --- Error metrics accumulators ---
        self.sum_sq_pos_err = 0.0     # for RMSE position
        self.itae_pos = 0.0           # for ITAE position (time-weighted absolute error)
        self.err_samples = 0          # you already have this; keep one copy only
        self.last_t = None            # for dt

        

        self.results_file = "sweep_results_V2.csv"

        # Create file with header if it doesn't exist
        if not os.path.exists(self.results_file):
            with open(self.results_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frequency", "resolution", "clicks", "RMSE_pos", "ITAE_pos"])


        # -PWM output timer -
        self.pwm_timer = self.create_timer(
            1.0 / (self.pwm_frequency*self.pwm_resolution),
            self.send_signals,
        )


        # --- control refresh time ---
        Ts = 1.0/float(self.pwm_frequency)
        self.cfg = replace(
            self.cfg,
            mpc=replace(self.cfg.mpc, Ts=Ts)
        )
        self.refresh_rate = Ts
        self.timer = self.create_timer(self.refresh_rate, self.control_step)


        # --- This is just to slow down the printing of states and stuff in the terminal ---
        self.printer_timing = 1


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

            #if t_fail is not None and self.t >= t_fail + self.failure_detection_timing_buffer:
            #    health_arr[i] = int(self.failure_states[i])
            #    if not self.failure_already_logged[i]:
            #        msg = (
            #                f"{RED}{BOLD}"
            #                f"*** THRUSTER {i+1} FAILURE DETECTED BY SYSTEM. AMS SWITCHED TO:***"
            #                f"{RESET}"
            #            )
            #        self.get_logger().error(msg)

        #--------------- Check if too many failed thrusters------------------#
        REP_health_arr = np.array([0,0,1,1,1,1,1,1], dtype=int) 
        rep_ams_mask = (health_arr != 0).astype(int)
        ams_mask = (health_arr != 0).astype(int)

        n_thrusters = self.cfg.phys.N_thrusters
        n_failed = n_thrusters - int(ams_mask.sum())
        if n_failed > self.cfg.phys.max_failed_thr:
            self.get_logger().error(
                f"Too many failed thrusters: {n_failed} > max_failed_thr={self.cfg.phys.max_failed_thr}. "
                "Controller is not designed for this failure level. Shutting down."
            )
            rclpy.shutdown()
            raise SystemExit(1)
        #-------------------------------------------------------------------#

        # --- MPC bounds for current failure case
        idx = mask_to_index(ams_mask)
        REP_idx = mask_to_index(REP_health_arr)

        if self.t >= self.t_failure[1]:
            mode = self.ams_cache[1]
        elif self.t >= self.t_failure[0]:
            mode = self.ams_cache[0]
        else:
            mode = self.ams_cache[2]

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
            #A=A_curr,
            A = self.cfg.phys.A,
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

        # --- dt (seconds) ---
        if self.last_t is None:
            dt = 0.0
        else:
            dt = float(self.t - self.last_t)
            if dt < 0.0:
                dt = 0.0
        self.last_t = float(self.t)

        # --- Instantaneous 2D position error ---
        ex = float(self.state[0] - self.reference[0])
        ey = float(self.state[1] - self.reference[1])
        pos_err = float(np.hypot(ex, ey))  # >= 0

        # --- Metric 1: RMSE position (accumulate squared error) ---
        self.sum_sq_pos_err += pos_err * pos_err
        self.err_samples += 1

        # --- Metric 2: ITAE position (time-weighted absolute error integral) ---
        self.itae_pos += float(self.t) * pos_err * dt


                # --- Main control step ---
        if self.bounds_mode == "ams":
            if self.t >= self.t_failure[1]:
                planes = np.array([[p.n[0], p.n[1], p.n[2], p.b] for p in self.simplified_ams_cache[4]], dtype=float)
                a_d = self.mpc.step(self.state, self.reference, bounds=bounds, caps=caps, planes=planes)
            elif self.t >= self.t_failure[0]:
                planes = np.array([[p.n[0], p.n[1], p.n[2], p.b] for p in self.simplified_ams_cache[3]], dtype=float)
                a_d = self.mpc.step(self.state, self.reference, bounds=bounds, caps=caps, planes=planes)
            else:
                planes = np.array([[p.n[0], p.n[1], p.n[2], p.b] for p in self.simplified_ams_cache[2]], dtype=float)
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
            a_d=a_d,
            mode=mode,
            cfg=self.cfg,
           )
        except ValueError as e:
            self.get_logger().warn(f"Allocation failed: {e}. Setting all thrusters to 0.")
            cmd = np.zeros(self.cfg.phys.N_thrusters, dtype=float)


        #self.alloc_pub.publish(cmd) #FOR PLOTTING PURPOSES ONLY, NOT USED FOR CONTROL


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

        current_msg= np.array([int(self.signals[i][self.i]) for i in range(self.cfg.phys.N_thrusters)],
        dtype=int)
        
        self.click_counter += int(np.sum(current_msg != self.prev_msg))
        if self.t >= self.sim_stop_time and not self.eval_logged:

            rmse_pos = np.sqrt(self.sum_sq_pos_err / max(self.err_samples, 1))
            itae_pos = self.itae_pos

            self.get_logger().info(
                f"=== Evaluation ===\n"
                f"RMSE position = {rmse_pos:.4f} m\n"
                f"ITAE position = {itae_pos:.4f} (m*s)\n"
                f"Total clicks  = {self.click_counter}"
            )

            # ---- Write to CSV ----
            with open(self.results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.pwm_frequency,     # adjust to your variable name
                    self.pwm_resolution,
                    self.click_counter,
                    rmse_pos,
                    itae_pos
                ])

            self.eval_logged = True
        self.prev_msg = current_msg

        self.i = (self.i + 1) % self.pwm_resolution

        self.cmd_pub.publish(msg)


    #This is a trigger for when we change a tuning parameter, it can be removed when tuning is done if desired. 
    def _on_params(self, params):
        watch = {
    'q_x','q_y','q_theta','q_vx','q_vy','q_omega','r_force','r_tau','pwm_frequency','pwm_resolution','max_force','N'}
    

        if any(p.name in watch for p in params):
            try:
                self._rebuild_mpc()
            except Exception as e:
                return SetParametersResult(successful=False, reason=str(e))

        return SetParametersResult(successful=True)

    #Rebuild the mpc with new tuned parameters live
    def _rebuild_mpc(self):
        cfg0 = make_default_config()
        Q = np.zeros((6,6), dtype=float)
        Q[0,0] = float(self.get_parameter('q_x').value)
        Q[1,1] = float(self.get_parameter('q_y').value)
        Q[2,2] = float(self.get_parameter('q_theta').value)
        Q[3,3] = float(self.get_parameter('q_vx').value)
        Q[4,4] = float(self.get_parameter('q_vy').value)
        Q[5,5] = float(self.get_parameter('q_r').value)
        r_force = self.get_parameter('r_force').value
        r_tau   = self.get_parameter('r_tau').value
        N = self.get_parameter('N').value

        R = cfg0.mpc.R.copy()
        R[0, 0] = r_force
        R[1, 1] = r_force
        R[2, 2] = r_tau

        mpc0 = cfg0.mpc
        self.cfg = cfg0.__class__(
            phys=cfg0.phys,
            mpc=mpc0.__class__(
                Fx_max=mpc0.Fx_max,
                Fy_max=mpc0.Fy_max,
                Tau_max=mpc0.Tau_max,
                N=N,
                Q=Q,
                R=R,
                Ts=self.cfg.mpc.Ts,
                max_planes=self.cfg.mpc.max_planes,
            ),
            world=cfg0.world,
        )
    
        self.mpc = MPCController(self.cfg, bounds_mode=self.bounds_mode)



        self.get_logger().info(
            f"Q updated: x={Q[0,0]:.1f} y={Q[1,1]:.1f} "
            f"theta={Q[2,2]:.1f} vx={Q[3,3]:.1f} vy={Q[4,4]:.1f} ω={Q[5,5]:.1f}"
        )


    def _hemisphere_w_positive(self,q):
        return -q if q[3] < 0 else q
    
    def _normalize_quat(self, q, eps: float = 1e-12):
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < eps:
            # fallback: identity quaternion (no rotation)
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n
    
    def _save_planes_to_file(self):
        from pathlib import Path
        from scipy.io import savemat
        health_arr = np.array(self.failure_states, dtype=int)
        mode_idx = mask_to_index(health_arr)  # change as needed

        planes = self.simplified_ams_cache[mode_idx]
        P = len(planes)

        n = np.zeros((P, 3), dtype=float)
        b = np.zeros((P,), dtype=float)

        for i, p in enumerate(planes):
            n[i, :] = np.asarray(p.n, dtype=float).reshape(3,)
            b[i] = float(p.b)

        out = Path("~/slider_ws/test1/planes_mode0.mat").expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        savemat(str(out), {
            "n": n,
            "b": b,
            "mode_idx": np.array([mode_idx], dtype=int),
        })

        self.get_logger().warn(f"Saved AMS planes to {out} (n:{n.shape}, b:{b.shape})")


            

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
