#!/usr/bin/env python3
"""
plot_IROS_V2.py

Compare a healthy run against 1 or 2 failure runs from rosbag2 (SQLite .db3 in a bag folder).

Outputs (same look as your reference script):
  1) states: x, y, theta(deg) vs time + dashed reference (x_ref, y_ref, theta_ref)
  2) velocities: xdot, ydot, thetadot(deg/s) vs time (no ref)

Legend + color rules (as you specified earlier):
  - 1 fail  : Healthy (green), Failure (red)
  - 2 fails : Healthy (green), With FWSm (blue), Without FWS (red)

Saving:
  Saves PNG + PDF into the directory where you run the script (cwd).
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ---------------- Style (match your look) ----------------
def apply_matlab_ieee_look():
    plt.style.use(["science"])
    mpl.rcParams.update({
        "text.usetex": False,

        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8,

        "axes.labelweight": "bold",
        "axes.linewidth": 1.0,
        "axes.grid": False,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.fontsize": 7,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.6,

        "lines.linewidth": 1.2,

        # keep identical canvas size (no tight bbox)
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "savefig.dpi": 300,
    })


# ---------------- ROS helpers ----------------
def yaw_from_quat(q):
    return np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )

def add_red_crosses(ax, t, y, times=(5.0, 15.0)):
    for te in times:
        if te < t[0] or te > t[-1]:
            continue
        ye = float(np.interp(te, t, y))
        ax.plot(te, ye, marker="x", color="red", markersize=6, mew=1.5,
                linestyle="None", zorder=10)

def _forward_fill_nan(y):
    y = y.copy()
    if len(y) == 0:
        return y
    mask = np.isfinite(y)
    if not np.any(mask):
        return y
    last = y[mask][0]
    for i in range(len(y)):
        if np.isfinite(y[i]):
            last = y[i]
        else:
            y[i] = last
    return y


def load_run(bag_dir, odom_topic="/odom", target_topic="/target_point", storage_id="sqlite3"):
    """
    Reads:
      - odom_topic: nav_msgs/msg/Odometry
      - target_topic (optional): typically geometry_msgs/msg/PoseStamped or nav_msgs/Odometry-ish,
        but you used msg.pose.pose... in your reference code, so we keep that access pattern.

    Returns arrays (time normalized to start at 0):
      t, x, y, th, vx, vy, om, xref, yref, thref
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if odom_topic not in types:
        raise RuntimeError(f"{odom_topic} not found in {bag_dir}. Topics: {list(types)}")

    Odom = get_message(types[odom_topic])
    Target = get_message(types[target_topic]) if target_topic in types else None

    t0 = None
    t, x, y, th, vx, vy, om = [], [], [], [], [], [], []
    xref, yref, thref = [], [], []
    latest_ref = None  # (x,y,theta)

    while reader.has_next():
        topic, data, t_nsec = reader.read_next()
        if t0 is None:
            t0 = t_nsec
        ts = (t_nsec - t0) / 1e9

        if Target and topic == target_topic:
            msg = deserialize_message(data, Target)
            # keep same access pattern as your reference script:
            q = msg.pose.pose.orientation
            latest_ref = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                float(yaw_from_quat(q)),
            )

        if topic == odom_topic:
            msg = deserialize_message(data, Odom)

            t.append(ts)
            x.append(float(msg.pose.pose.position.x))
            y.append(float(msg.pose.pose.position.y))
            q = msg.pose.pose.orientation
            th.append(float(yaw_from_quat(q)))

            vx.append(float(msg.twist.twist.linear.x))
            vy.append(float(msg.twist.twist.linear.y))
            om.append(float(msg.twist.twist.angular.z))

            if latest_ref is not None:
                xref.append(latest_ref[0])
                yref.append(latest_ref[1])
                thref.append(latest_ref[2])
            else:
                xref.append(np.nan)
                yref.append(np.nan)
                thref.append(np.nan)

    return {
        "t": np.asarray(t, float),
        "x": np.asarray(x, float),
        "y": np.asarray(y, float),
        "th": np.asarray(th, float),
        "vx": np.asarray(vx, float),
        "vy": np.asarray(vy, float),
        "om": np.asarray(om, float),
        "xref": np.asarray(xref, float),
        "yref": np.asarray(yref, float),
        "thref": np.asarray(thref, float),
    }


def common_time(runs):
    t_end = min(r["t"][-1] for r in runs if len(r["t"]) > 0)
    n = min(len(r["t"]) for r in runs if len(r["t"]) > 0)
    return np.linspace(0.0, t_end, num=max(n, 2))


def resample(run, t_new):
    t = run["t"]
    out = {"t": t_new}
    if len(t) < 2:
        for k in ("x", "y", "th", "vx", "vy", "om", "xref", "yref", "thref"):
            out[k] = np.full_like(t_new, np.nan, float)
        return out

    for k in ("x", "y", "th", "vx", "vy", "om"):
        out[k] = np.interp(t_new, t, run[k])

    for k in ("xref", "yref", "thref"):
        y = run[k]
        out[k] = (
            np.interp(t_new, t, _forward_fill_nan(y))
            if np.any(np.isfinite(y))
            else np.full_like(t_new, np.nan, float)
        )

    return out


def save(fig, out_dir, stem, prefix=""):
    fig.savefig(os.path.join(out_dir, f"{prefix}{stem}.png"))
    fig.savefig(os.path.join(out_dir, f"{prefix}{stem}.pdf"))


# ---------------- Main plotting ----------------
def main():
    FIGSIZE = (4.8, 3.6)
    DEG = 180.0 / np.pi

    p = argparse.ArgumentParser()
    p.add_argument("--healthy", required=True, help="Bag folder (contains metadata.yaml + *.db3)")
    p.add_argument("--fail1", required=True, help="Failure bag folder")
    p.add_argument("--fail2", default=None, help="Optional second failure bag folder")
    p.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"])
    p.add_argument("--odom-topic", default="/odom")
    p.add_argument("--target-topic", default="/target_point")
    p.add_argument("--name", default="", help="Filename prefix")
    args = p.parse_args()

    apply_matlab_ieee_look()

    # Labels/colors/styles per your rules
    if args.fail2 is None:
        labels = ["Healthy", "Failure"]
        colors = ["#2ca02c", "#d62728"]   # green, red
        styles = ["-", "--"]
        bags = [args.healthy, args.fail1]
    else:
        labels = ["Healthy", "With FWSm", "Without FWS"]
        colors = ["#2ca02c", "#1f77b4", "#d62728"]  # green, blue, red
        styles = ["-", "--", "-."]
        bags = [args.healthy, args.fail1, args.fail2]

    REF_STYLE = dict(color="k", linewidth=1.2, linestyle=(0, (6, 3)))  # dashed black

    runs = [
        load_run(b, odom_topic=args.odom_topic, target_topic=args.target_topic, storage_id=args.storage_id)
        for b in bags
    ]

    t = common_time(runs)
    rs = [resample(r, t) for r in runs]

    # Save to where the script is run from
    out_dir = os.getcwd()
    prefix = f"{args.name}_" if args.name else ""

    # ---- Figure: states (x, y, theta) + ref dashed ----
    fig, ax = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
    fig.set_constrained_layout(True)

    # x
    for r, lab, c, ls in zip(rs, labels, colors, styles):
        ax[0].plot(t, r["x"], color=c, linestyle=ls, label=lab)
    if np.any(np.isfinite(rs[0]["xref"])):
        ax[0].plot(t, rs[0]["xref"], label="x_ref", **REF_STYLE)
    add_red_crosses(ax[0], t, rs[1]["x"],  times=(5.0, 15.0))
    ax[0].set_ylabel("x (m)")
    ax[0].legend(loc="best")

    # y
    for r, lab, c, ls in zip(rs, labels, colors, styles):
        ax[1].plot(t, r["y"], color=c, linestyle=ls, label=lab)
    if np.any(np.isfinite(rs[0]["yref"])):
        ax[1].plot(t, rs[0]["yref"], label="y_ref", **REF_STYLE)
    add_red_crosses(ax[1], t, rs[1]["y"],  times=(5.0, 15.0))    
    ax[1].set_ylabel("y (m)")

    # theta (deg)
    for r, lab, c, ls in zip(rs, labels, colors, styles):
        ax[2].plot(t, DEG * r["th"], color=c, linestyle=ls, label=lab)
    if np.any(np.isfinite(rs[0]["thref"])):
        ax[2].plot(t, DEG * rs[0]["thref"], label=r"$\theta_\mathrm{ref}$", **REF_STYLE)
    add_red_crosses(ax[2], t, (180/np.pi)*rs[1]["th"], times=(5.0, 15.0))    
    ax[2].set_ylabel(r"$\theta$ (deg)")
    ax[2].set_xlabel("Time (s)")

    save(fig, out_dir, "states", prefix=prefix)

    # ---- Figure: velocities (xdot, ydot, thetadot) ----
    fig, ax = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
    fig.set_constrained_layout(True)

    for r, lab, c, ls in zip(rs, labels, colors, styles):
        ax[0].plot(t, r["vx"], color=c, linestyle=ls, label=lab)
    add_red_crosses(ax[0], t, rs[1]["vx"], times=(5.0, 15.0))    
    ax[0].set_ylabel(r"$\dot{x}$ (m/s)")
    ax[0].legend(loc="best")

    for r, lab, c, ls in zip(rs, labels, colors, styles):
        ax[1].plot(t, r["vy"], color=c, linestyle=ls, label=lab)
    add_red_crosses(ax[1], t, rs[1]["vy"], times=(5.0, 15.0))    
    ax[1].set_ylabel(r"$\dot{y}$ (m/s)")

    for r, lab, c, ls in zip(rs, labels, colors, styles):
        ax[2].plot(t, DEG * r["om"], color=c, linestyle=ls, label=lab)
    add_red_crosses(ax[2], t, (180/np.pi)*rs[1]["om"], times=(5.0, 15.0))    
    ax[2].set_ylabel(r"$\dot{\theta}$ (deg/s)")
    ax[2].set_xlabel("Time (s)")

    save(fig, out_dir, "velocities", prefix=prefix)

    plt.close("all")
    print(f"=== Done ===\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()