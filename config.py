from dataclasses import dataclass
import numpy as np

### --- This file sets up a lot of parameters used throughout the package --- ###

# --- Dataclasses ---

@dataclass(frozen=True)
class MPCParams:
    Fx_max: float
    Fy_max: float
    Tau_max: float
    N: int
    Q: np.ndarray
    R: np.ndarray
    Ts: float
    max_planes: int


@dataclass(frozen=True)
class PhysicalParams:
    m: float
    Izz: float
    N_thrusters: int
    max_failed_thr: int
    u_min: np.ndarray
    u_max: np.ndarray
    A: np.ndarray
    #max_thrust: float


@dataclass(frozen=True)
class WorldLimits:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    wall_buffer: float

#@dataclass(frozen=True)
#class PWMparams:
 #   pwm_frequency: int
 #   pwm_resolution: int


@dataclass(frozen=True)
class SliderConfig:
    phys: PhysicalParams
    mpc: MPCParams
    world: WorldLimits
    #pwm: PWMparams



def make_default_config() -> SliderConfig:


    # --- Physical parameters ---
    m = 4.436
    Izz = 1.092
    N_thrusters = 8
    max_failed_thr = 4 #Changes how many AMS's are pre computed
    slider_radius = 0.2
    slider_table_length = 4
    moment_arm = 0.14
    max_thrust = 0.7
    max_planes = 12 #For simpleified AMS cache


    A = np.array([
             [-1,     0,     1,     0,     1,     0,    -1,     0   ],
             [ 0,     1,     0,     1,     0,    -1,     0,    -1   ],
             [-moment_arm,  moment_arm,  moment_arm, -moment_arm, -moment_arm,  moment_arm,  moment_arm, -moment_arm],
         ], dtype=np.float64)


    

    #PWM parameters
    pwm_frequency = 10  # Hz
    pwm_resolution = 10   # bits


    # --- MPC parameters ---
    Ts = 1.0/(pwm_frequency)  #WAS 1.0 / 7.0
    Fx_Max  = 2 * max_thrust
    Fy_Max  = 2 * max_thrust
    Tau_Max = 4* max_thrust * moment_arm

    N = 15

    #circle
    #Q = np.diag([15.0, 15.0, 0.0, 0.0, 5.0, 0.0,  3.0, 3.0, 3.5])  # [x,y,qx,qy,qz,qs,vx,vy,omega]
    #R = np.diag([0.1, 0.1, 0.5])

    #HEALTHY SET POINT    ALSO FOR PLOTTING DONT CHANGE!!!!!!
    Q = np.diag([5.0, 5.0, 5.0, 10.0, 10.0, 5.0])  # [x,y,theta,vx,vy,omega]
    R = np.diag([0.0, 0.0, 0.0])

    #HW?
    #Q = np.diag([6.0, 4.0, 8.0, 2.0, 2.0, 10.0])  # [x,y,theta,vx,vy,omega]  4 4 8 2 2 10
    #R = np.diag([0.0, 0.0, 0.0])

    #Q = np.diag([5.0, 5.0, 5.0, 2.0, 2.0, 10.0])  # [x,y,theta,vx,vy,omega]  4 4 8 2 2 10
    #R = np.diag([0.1, 0.1, 0.2])
    #HEALTHY SET POINT FREQ 1 RES HIGH   ALSO FOR PLOTTING DONT CHANGE!!!!!!
    #Q = np.diag([7.0, 7.0, 0.0, 0.0, 4.0, 0.0,  15.0, 15.0, 2.0])  # [x,y,qx,qy,qz,qs,vx,vy,omega]
    #R = np.diag([0.3, 0.3, 3.0])


    #TWO PASSIVE
    #Q = np.diag([5.0, 5.0, 0.0, 0.0, 10.0, 0.0,  25.0, 25.0, 10.0])  # [x,y,qx,qy,qz,qs,vx,vy,omega]
    #R = np.diag([0.3, 0.3, 1.0])

    #ONE ACTIVE
    #Q = np.diag([15.0, 15.0, 0.0, 0.0, 10.0, 0.0,  30.0, 30.0, 12.0])  # [x,y,qx,qy,qz,qs,vx,vy,omega]
    #R = np.diag([0.0, 0.0, 0.0])

    #FOR ACTIVE PUT LIKE 150 FOR x,y,theta. 




    mpc = MPCParams(
        Fx_max=Fx_Max,
        Fy_max=Fy_Max,
        Tau_max=Tau_Max,
        N=N,
        Q=Q,
        R=R,
        Ts=Ts,
        max_planes=max_planes,
    )

    # thrust limits
    u_max = max_thrust * np.ones(N_thrusters, dtype=float)
    u_min = np.zeros(N_thrusters, dtype=float)


    phys = PhysicalParams(
        m=m,
        Izz=Izz,
        N_thrusters=N_thrusters,
        max_failed_thr = max_failed_thr,
        u_min=u_min,
        u_max=u_max,
        A=A,
        #max_thrust = max_thrust,
    )

    # --- World limits ---
    wall_buffer = slider_radius
    x_min = -slider_table_length / 2 + wall_buffer
    x_max = slider_table_length / 2 - wall_buffer
    y_min = -slider_table_length / 2 + wall_buffer
    y_max = slider_table_length / 2 - wall_buffer

    world = WorldLimits(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        wall_buffer=wall_buffer,
    )

    #pwm = PWMparams(
        #pwm_frequency=pwm_frequency,
        #pwm_resolution=pwm_resolution,
    #)

    return SliderConfig(
        phys=phys,
        mpc=mpc,
        world=world,
        #pwm=pwm,
    )
