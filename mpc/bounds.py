from dataclasses import dataclass
import numpy as np


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
    A: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
) -> WrenchBounds:

    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)


    def one_row_bounds(a_row: np.ndarray, u_min: np.ndarray, u_max: np.ndarray):
        a = a_row.reshape(-1)
        amin = 0.0
        amax = 0.0
        for aj, umin_j, umax_j in zip(a, u_min, u_max):
            if aj > 0:
                amax += aj * umax_j
                amin += aj * umin_j
            elif aj < 0:
                amax += aj * umin_j
                amin += aj * umax_j
            # aj == 0 contributes nothing
        return amin, amax

    rowFx = A[0, :]
    rowFy = A[1, :]
    rowTau = A[2, :]

    Fx_min, Fx_max = one_row_bounds(rowFx, u_min, u_max)
    Fy_min, Fy_max = one_row_bounds(rowFy, u_min, u_max)
    Tau_min, Tau_max = one_row_bounds(rowTau, u_min, u_max)

    return WrenchBounds(
        Fx_min=Fx_min,
        Fx_max=Fx_max,
        Fy_min=Fy_min,
        Fy_max=Fy_max,
        Tau_min=Tau_min,
        Tau_max=Tau_max,
    ) 
