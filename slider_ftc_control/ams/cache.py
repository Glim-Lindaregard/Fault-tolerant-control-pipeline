import numpy as np
from typing import List
from slider_ftc_control.config import SliderConfig
from dataclasses import dataclass


@dataclass
class AMSMode:
    idx: int
    thruster_state: np.ndarray
    facets: list[np.ndarray]
    

# --- Thruster int index to bit index for failure injection ---
def _index_to_thruster_state(idx: int, N_thrusters: int) -> np.ndarray:
    thruster_state = np.zeros(N_thrusters, dtype=int)
    for k in range(N_thrusters):
        thruster_state[k] = idx % 3
        idx //= 3
    return thruster_state


# --- Inverse of above ---
def thruster_state_to_index(thruster_state: np.ndarray) -> int:
    idx = 0
    mult = 1
    for s in thruster_state:
        idx += int(s) * mult
        mult *= 3
    return idx


# --- Main AMS builder function --- 
def _build_ams_from_state(cfg: SliderConfig, thruster_state: np.ndarray) -> List[np.ndarray]:
    A = cfg.phys.A
    umin = cfg.phys.u_min.copy()
    umax = cfg.phys.u_max.copy()

    passive = (thruster_state == 0)
    umin[passive] = 0.0
    umax[passive] = 0.0

    active = (thruster_state == 2)
    umin[active] = cfg.phys.u_max[active]
    umax[active] = cfg.phys.u_max[active]

    N = cfg.phys.N_thrusters
    tol = 1e-15
    tol_fix = 1e-6

    facets = []

    fixed = np.abs(umin - umax) < tol_fix

    # --- Loop over thruster pairs, ignoring paires when one columns is zero ---
    for i in range(N - 1):
        if fixed[i]:
            continue
        for j in range(i + 1, N):
            if fixed[j]:
                continue

            Asub = A[:, [i, j]]

            A1, dropRow = best_submatrix(Asub)

            rows = [0, 1, 2]
            keepRows = [r for r in rows if r != dropRow]
            A2 = Asub[dropRow, :].reshape(1, 2)

            n = np.zeros(3)
            try:
                x = np.linalg.solve(A1, A2.T).flatten()
            except np.linalg.LinAlgError:
                continue
            n[keepRows] = -x
            n[dropRow] = 1.0

            n[np.abs(n) < tol] = 0.0
            if np.linalg.norm(n) < tol:
                continue

            ss = n @ A
            s = np.zeros(N)
            s[ss < -tol] = -1
            s[ss > tol] = 1
            s[i] = 0
            s[j] = 0


            for which in [1, 2]:

                u = np.zeros(N)
                if which == 1:
                    u[s == 1] = umax[s == 1]
                    u[s == -1] = umin[s == -1]
                else:
                    u[s == 1] = umin[s == 1]
                    u[s == -1] = umax[s == -1]

                u1 = u.copy()
                u2 = u.copy()
                u3 = u.copy()
                u4 = u.copy()

                u1[i] = umin[i]; u1[j] = umin[j]
                u2[i] = umax[i]; u2[j] = umin[j]
                u3[i] = umax[i]; u3[j] = umax[j]
                u4[i] = umin[i]; u4[j] = umax[j]

                Uk = np.column_stack((u1, u2, u3, u4))
                facets.append(Uk)

    return facets


# --- Build all AMS for cases with failed thrusters <= max_thrusters_failed ---
def build_ams_cache(cfg: SliderConfig) -> List[AMSMode]:
    N_thrusters = cfg.phys.N_thrusters
    N_modes = 3 ** N_thrusters

    cache: List[AMSMode] = [None] * N_modes

    for idx in range(N_modes):
        thruster_state = _index_to_thruster_state(idx, N_thrusters)

        facets = _build_ams_from_state(cfg, thruster_state)

        cache[idx] = AMSMode(
            idx=idx,
            thruster_state=thruster_state,
            facets=facets
        )
    return cache


# --- Helper to find best conditioned 2x2 from a 3x2 ---
def best_submatrix(Asub: np.ndarray):
    conds = []
    mats = []
    for drop in range(3):
        keep = [r for r in range(3) if r != drop]
        Atemp = Asub[keep, :]
        conds.append(1.0 / np.linalg.cond(Atemp))
        mats.append(Atemp.T)

    dropRow = int(np.argmax(conds))
    A1 = mats[dropRow]
    return A1, dropRow

