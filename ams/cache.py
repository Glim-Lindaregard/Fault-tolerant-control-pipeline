import numpy as np
from typing import List
from slider_ftc_control.config import SliderConfig
from dataclasses import dataclass


@dataclass
class AMSMode:
    A_mode: np.ndarray
    facets: list[np.ndarray]
    mask: np.ndarray

# --- Thruster int index to bit index for failure injection ---
def index_to_mask(idx: int, N_thrusters: int) -> np.ndarray:

    bits = [(idx >> k) & 1 for k in range(N_thrusters)]
    return np.array(bits, dtype=int)


# --- Inverse of above ---
def mask_to_index(mask: np.ndarray) -> int:

    idx = 0
    for k, bit in enumerate(mask):
        if int(bit) != 0:
            idx |= (1 << k)
    return idx


# --- Set failed columns of A to zero ---
def mask_columns(A: np.ndarray, mask: np.ndarray) -> np.ndarray:

    A_mode = A.copy()
    for i, h in enumerate(mask):
        if h == 0:
            A_mode[:, i] = 0.0
    return A_mode

# --- Main AMS builder function --- 
def build_ams_for_mask(cfg: SliderConfig, A_mode: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    A = A_mode
    umin = cfg.phys.u_min.copy()
    umax = cfg.phys.u_max.copy()

    failed = (mask == 0)
    umin[failed] = 0.0
    umax[failed] = 0.0

    N = cfg.phys.N_thrusters
    tol = 1e-15

    facets = []

    zero_cols = np.where(np.all(np.abs(A) < tol, axis=0))[0]

    if len(zero_cols) >= N - 1:
        return []

    # --- Loop over thruster pairs, ignoring paires when one columns is zero ---
    for i in range(N - 1):
        if i in zero_cols:
            continue
        for j in range(i + 1, N):
            if j in zero_cols:
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
    N_modes = 2 ** N_thrusters

    cache: List[AMSMode] = [None] * N_modes
    max_failed = cfg.phys.max_failed_thr

    for idx in range(N_modes):
        mask = index_to_mask(idx, N_thrusters)
        n_failed = int((mask == 0).sum())
        A_mode = mask_columns(cfg.phys.A, mask)

        # --- Dont compute AMS when more then max_failed_thr are broken ---
        if n_failed > max_failed:
            cache[idx] = AMSMode(
                A_mode = A_mode,
                facets = [],
                mask = mask
            )

        # --- Normal AMS build
        facets = build_ams_for_mask(cfg, A_mode,mask)


        cache[idx] = AMSMode(
            A_mode=A_mode,
            facets = facets, 
            mask=mask
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

