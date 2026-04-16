from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from slider_ftc_control.ams.cache import AMSMode


@dataclass(frozen=True)
class Plane:
    n: np.ndarray  # (3,) unit normal
    b: float       # halfspace: n^T x <= b


# ----------------------- Helpers ------------------------

def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3,)
    nv = np.linalg.norm(v)
    if nv < eps:
        raise ValueError("Degenerate vector (near zero norm).")
    return v / nv


def _normal_from_quad(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """V: (3,4) quad vertices. Returns a unit normal (sign not canonicalized)."""
    v = np.asarray(V, dtype=float)
    if v.shape != (3, 4):
        raise ValueError("Expected V shape (3,4).")

    pts = v.T  # (4,3)
    tri = [(0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)]
    for i, j, k in tri:
        n_raw = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        if np.linalg.norm(n_raw) > eps:
            return _unit(n_raw, eps)
    raise ValueError("Degenerate facet normal.")


def _is_same_direction(n1: np.ndarray, n2: np.ndarray, tol_dir: float) -> bool:
    """Signed match: dot ~ +1 (does not treat n and -n as same)."""
    return (1.0 - float(np.dot(n1, n2))) <= tol_dir


def _orient_outward(n: np.ndarray, b: float, c: np.ndarray, tol: float = 1e-12):
    """Ensure interior point c satisfies n^T c <= b; if not, flip (n,b)."""
    if float(np.dot(n, c)) > b + tol:
        return -n, -b
    return n, b


# ------------------------ Main cache builder --------------------------

def build_simplified_ams_cache(
    ams_cache: List[AMSMode],
    tol_n: float = 1e-6,
    tol_b: float = 1e-6,
) -> List[List[Plane]]:

    simplified_cache: List[List[Plane]] = []

    for mode in ams_cache:
        A = np.asarray(mode.A_mode, dtype=float)

        V_list = []
        normals = []

        for Uk in mode.facets:
            Uk = np.asarray(Uk, dtype=float)
            Vk = A @ Uk  # (3,4)
            V_list.append(Vk.T)  # (4,3)
            try:
                normals.append(_normal_from_quad(Vk))
            except ValueError:
                pass

        if not V_list or not normals:
            simplified_cache.append([])
            continue

        V_all = np.vstack(V_list)   # (Nv,3)
        c = V_all.mean(axis=0)      # guaranteed inside conv(V_all)

        cluster_n: List[np.ndarray] = []
        cluster_b: List[float] = []

        for n in normals:
            vals = V_all @ n
            b = float(np.max(vals))          # support in direction n
            n, b = _orient_outward(n, b, c)  # orient so c is inside

            assigned = False
            for i in range(len(cluster_n)):
                if _is_same_direction(cluster_n[i], n, tol_n):
                    if b > cluster_b[i] + tol_b:
                        cluster_b[i] = b
                        cluster_n[i] = n
                    assigned = True
                    break

            if not assigned:
                cluster_n.append(n)
                cluster_b.append(b)

        simplified_cache.append([Plane(n=cluster_n[i], b=float(cluster_b[i]))
                                 for i in range(len(cluster_n))])

    return simplified_cache