#!/usr/bin/env python3
"""
3D visualization for simplified AMS planes.

Each plane is represented as:
    n · x = b
where n is a (3,) unit normal and b is a scalar.

Designed to be called from ftc_node after build_simplified_ams_cache(...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# If you import Plane from your module, delete this local definition.
@dataclass(frozen=True)
class Plane:
    n: np.ndarray  # (3,)
    b: float


def _plane_patch_in_box(n: np.ndarray, b: float,
                        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                        tol: float = 1e-12) -> Optional[np.ndarray]:
    """
    Compute a convex polygon patch (as vertices) given by the intersection of the plane
    n·x = b with an axis-aligned box.

    Returns:
        verts: (K,3) array of vertices ordered around centroid, or None if empty.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    n = np.asarray(n, dtype=float).reshape(3,)
    b = float(b)

    # 12 edges of the box as (p0, p1)
    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ], dtype=float)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    pts: List[np.ndarray] = []

    for i0, i1 in edges:
        p0 = corners[i0]
        p1 = corners[i1]
        d0 = np.dot(n, p0) - b
        d1 = np.dot(n, p1) - b

        # If endpoints lie (nearly) on plane, include them
        if abs(d0) <= tol:
            pts.append(p0)
        if abs(d1) <= tol:
            pts.append(p1)

        # Proper segment intersection if signs differ
        if d0 * d1 < -tol**2:
            t = d0 / (d0 - d1)  # in (0,1)
            p = p0 + t * (p1 - p0)
            pts.append(p)

    if len(pts) < 3:
        return None

    # Deduplicate points (box intersections can repeat)
    P = np.unique(np.round(np.vstack(pts), decimals=12), axis=0)
    if P.shape[0] < 3:
        return None

    # Order vertices around centroid in the plane
    c = P.mean(axis=0)

    # Build an orthonormal basis (u,v) spanning the plane
    # Choose a vector not parallel to n
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    nu = np.linalg.norm(u)
    if nu < tol:
        return None
    u /= nu
    v = np.cross(n, u)

    # Project to 2D and sort by angle
    Q = P - c
    x2 = Q @ u
    y2 = Q @ v
    ang = np.arctan2(y2, x2)
    order = np.argsort(ang)
    return P[order]


def plot_planes_3d(
    planes: Sequence[Plane],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)
    ),
    title: str = "Planes: n·x = b",
    show_normals: bool = True,
    normal_scale: float = 0.25,
    alpha: float = 0.20,
    max_planes: Optional[int] = None,
    ax=None,
):
    """
    Render plane patches (clipped to bounds) into a 3D matplotlib axis.

    bounds: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) in *wrench space* coordinates.
    """
    if ax is None:
        fig = plt.figure(figsize=(7.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Axis bounds
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.set_xlabel("Fx")
    ax.set_ylabel("Fy")
    ax.set_zlabel("Tau")
    ax.set_title(title)

    # Plot origin for reference
    ax.scatter([0], [0], [0], s=20)

    count = 0
    for p in planes:
        if max_planes is not None and count >= max_planes:
            break

        n = np.asarray(p.n, dtype=float).reshape(3,)
        b = float(p.b)

        verts = _plane_patch_in_box(n, b, bounds)
        if verts is None:
            continue

        poly = Poly3DCollection([verts], alpha=alpha, linewidths=0.5)
        ax.add_collection3d(poly)

        if show_normals:
            # closest point on plane to origin (since n is unit): p0 = b*n
            p0 = b * n
            ax.quiver(
                p0[0], p0[1], p0[2],
                n[0], n[1], n[2],
                length=normal_scale,
                normalize=True,
            )

        count += 1

    # Make axes roughly equal scale (matplotlib doesn't do this nicely by default)
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    midx = 0.5 * (xmin + xmax)
    midy = 0.5 * (ymin + ymax)
    midz = 0.5 * (zmin + zmax)
    ax.set_xlim(midx - max_range / 2, midx + max_range / 2)
    ax.set_ylim(midy - max_range / 2, midy + max_range / 2)
    ax.set_zlim(midz - max_range / 2, midz + max_range / 2)

    return fig, ax


def plot_mode_planes(
    simplified_cache: Sequence[Sequence[Plane]],
    mode_idx: int,
    bounds=((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
    **kwargs,
):
    """
    Convenience wrapper: pick a mode and plot its planes.
    """
    planes = simplified_cache[mode_idx]
    title = kwargs.pop("title", f"Mode {mode_idx}: {len(planes)} unique planes")
    return plot_planes_3d(planes, bounds=bounds, title=title, **kwargs)


if __name__ == "__main__":
    # Tiny self-test (one plane): x + y + z = 1
    cache = [[Plane(n=np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), b=1.0)]]
    plot_mode_planes(cache, 0, bounds=((-1, 1), (-1, 1), (-1, 1)))
    plt.show()
