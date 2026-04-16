from typing import List
import numpy as np
from slider_ftc_control.config import SliderConfig
from .cache import AMSMode

# --- This function takes desired moment a_d and current failure mode and find U (8x1 float, not pulse width)
def allocate_wrench(a_d: List[float],
                    mode: AMSMode,
                    cfg: SliderConfig) -> np.ndarray:



    ad = np.asarray(a_d, dtype=float).reshape(3,)
    N = cfg.phys.N_thrusters
    tol = 1e-15


    A_current = mode.A_mode
    facets = mode.facets

    if not facets:
        return np.zeros(N, dtype=float)

    found = False
    # --- Search method --- 
    for Uk in facets:

        Vk = A_current @ Uk
        adi = Vk[:,0]

        if np.all(np.abs(adi) < tol):
            adi = Vk[:,2]

        adj = Vk[:,1]
        adk = Vk[:,3]

        M = np.column_stack((ad,adi-adj,adi-adk))

        try:
            x = np.linalg.solve(M, adi)
        except np.linalg.LinAlgError:
            continue

        a,b,c = x

        if a > 0 and 0 <= b <= 1 and 0 <= c <= 1:

            ui = Uk[:,0]
            uj = Uk[:,1]
            uk = Uk[:,3]
            uStar = ui + b*(uj - ui) + c*(uk - ui)  

            if a >= 1:
                uOut = uStar / a
            else:
                uOut = uStar

            found = True
            break


    if not found:
        raise ValueError("No feasible thruster moment found")

    return uOut