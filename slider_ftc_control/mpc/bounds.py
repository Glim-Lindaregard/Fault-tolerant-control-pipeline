from dataclasses import dataclass
import numpy as np
from slider_ftc_control.config import SliderConfig

@dataclass
class WrenchBounds:
    Fx_min: float
    Fx_max: float
    Fy_min: float
    Fy_max: float
    Tau_min: float
    Tau_max: float

# --- Looks at available forces and moments and finds bounds for mpc --- 
def compute_wrench_bounds(
    cfg: SliderConfig,
    state: np.ndarray,
) -> WrenchBounds:

    A = cfg.phys.A
    u_min = cfg.phys.u_min.copy()
    u_max = cfg.phys.u_max.copy()

    passive = (state == 0)
    u_min[passive] = 0.0
    u_max[passive] = 0.0

    active = (state == 2)
    u_min[active] = cfg.phys.u_max[active]
    u_max[active] = cfg.phys.u_max[active]

    def one_row_bounds(a_row: np.ndarray):
        amin = 0.0
        amax = 0.0
        for aj, umin_j, umax_j in zip(a_row, u_min, u_max):
            if aj > 0:
                amax += aj * umax_j
                amin += aj * umin_j
            elif aj < 0:
                amax += aj * umin_j
                amin += aj * umax_j
        return amin, amax

    Fx_min, Fx_max = one_row_bounds(A[0, :])
    Fy_min, Fy_max = one_row_bounds(A[1, :])
    Tau_min, Tau_max = one_row_bounds(A[2, :])

    return WrenchBounds(
        Fx_min=Fx_min,
        Fx_max=Fx_max,
        Fy_min=Fy_min,
        Fy_max=Fy_max,
        Tau_min=Tau_min,
        Tau_max=Tau_max,
    )
