from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from slider_ftc_control.config import SliderConfig


@dataclass
class AMSMode:
    state: np.ndarray          # (N,) values in {0,1,2}
    A_mode: np.ndarray         # (3,N)
    facets: List[np.ndarray]   # each facet is (N,4)


def state_to_A_mode(A: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Build the allocation matrix for a given thruster state.

    Convention:
        0 = passive failed  -> column set to zero
        1 = healthy         -> original column
        2 = active failed   -> keep column in A_mode

    Note:
        State=2 means the thruster is geometrically still available in the AMS
        representation. Any stuck-on bias handling must be done consistently
        elsewhere if that is part of your chosen formulation.
    """
    A_mode = A.copy()
    passive = (state == 0)
    A_mode[:, passive] = 0.0
    return A_mode


def build_ams_for_state(
    cfg: SliderConfig,
    A_mode: np.ndarray,
    state: np.ndarray,
) -> List[np.ndarray]:
    """
    Build AMS facets for a state in {0,1,2}^N.

    Assumed convention for bounds:
        0 -> u_i fixed to 0
        1 -> u_i in [u_min, u_max]
        2 -> u_i fixed to u_max

    Returns:
        facets: list of arrays Uk with shape (N,4)
    """
    N = cfg.phys.N_thrusters
    tol = 1e-15

    umin = cfg.phys.u_min.copy()
    umax = cfg.phys.u_max.copy()

    # Apply state constraints directly
    passive = (state == 0)
    active = (state == 2)

    umin[passive] = 0.0
    umax[passive] = 0.0

    umin[active] = cfg.phys.u_max[active]
    umax[active] = cfg.phys.u_max[active]

    A = A_mode
    facets: List[np.ndarray] = []

    zero_cols = np.where(np.all(np.abs(A) < tol, axis=0))[0]

    # Not enough nonzero directions left to define 3D geometry
    if len(zero_cols) >= N - 1:
        return []

    for i in range(N - 1):
        if i in zero_cols:
            continue

        for j in range(i + 1, N):
            if j in zero_cols:
                continue

            Asub = A[:, [i, j]]
            A1, drop_row = best_submatrix(Asub)

            rows = [0, 1, 2]
            keep_rows = [r for r in rows if r != drop_row]
            A2 = Asub[drop_row, :].reshape(1, 2)

            n = np.zeros(3, dtype=float)
            try:
                x = np.linalg.solve(A1, A2.T).flatten()
            except np.linalg.LinAlgError:
                continue

            n[keep_rows] = -x
            n[drop_row] = 1.0

            n[np.abs(n) < tol] = 0.0
            if np.linalg.norm(n) < tol:
                continue

            ss = n @ A
            s = np.zeros(N, dtype=int)
            s[ss < -tol] = -1
            s[ss > tol] = 1
            s[i] = 0
            s[j] = 0

            for which in (1, 2):
                u = np.zeros(N, dtype=float)

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


def build_ams_cache(
    cfg: SliderConfig,
    S: np.ndarray,
) -> List[AMSMode]:
    """
    Build AMS cache only for the states listed in S.

    Args:
        cfg: slider config
        S: array of shape (n_states, N_thrusters), values in {0,1,2}

    Returns:
        cache: list of AMSMode, same order as rows of S
    """
    S = np.asarray(S, dtype=int)

    if S.ndim != 2:
        raise ValueError("S must be a 2D array of shape (n_states, N_thrusters).")

    N = cfg.phys.N_thrusters
    if S.shape[1] != N:
        raise ValueError(f"S must have {N} columns, got {S.shape[1]}.")

    if not np.all(np.isin(S, [0, 1, 2])):
        raise ValueError("S must contain only values in {0,1,2}.")

    cache: List[AMSMode] = []

    for state in S:
        state = state.copy()
        A_mode = state_to_A_mode(cfg.phys.A, state)
        facets = build_ams_for_state(cfg, A_mode, state)

        cache.append(
            AMSMode(
                state=state,
                A_mode=A_mode,
                facets=facets,
            )
        )

    return cache


def best_submatrix(Asub: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pick the best-conditioned 2x2 submatrix from a 3x2 matrix Asub.
    Returns:
        A1: chosen 2x2 matrix
        drop_row: index of removed row in original 3x2 matrix
    """
    scores = []
    mats = []

    for drop in range(3):
        keep = [r for r in range(3) if r != drop]
        Atemp = Asub[keep, :]
        cond = np.linalg.cond(Atemp)
        score = 0.0 if not np.isfinite(cond) else 1.0 / cond
        scores.append(score)
        mats.append(Atemp.T)

    drop_row = int(np.argmax(scores))
    A1 = mats[drop_row]
    return A1, drop_row