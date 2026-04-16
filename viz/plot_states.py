import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

# import failure config from FTC node
from slider_ftc_control.ftc_node import FAILURE_TIMES, FAILURE_STATES

# ---------- small helper: quaternion -> euler (roll, pitch, yaw) ----------

def euler_from_quaternion(q):
    """
    q: [x, y, z, w]
    returns (roll, pitch, yaw) in radians
    """
    x, y, z, w = q

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


# ---------- config ----------

DEFAULT_BAG = "/root/slider_ws/test1/test1_0.db3"
odom_topic = "/odom"
target_topic = "/target_point"
action_bounds_topic = "/action_bounds"  # Float32MultiArray (3x2 or 2x2)
thrust_cmd_topic = "/thrust_cmd"        # geometry_msgs/Vector3: [fx, fy, tau]
DEFAULT_STORAGE_ID = "sqlite3"

SCALE = 0          # ellipse scale (0 = disabled)
DEFAULT_PM = 0.7
TOL = 1e-3

# failure info (shared with FTCNode)
failure_times = list(FAILURE_TIMES)
failure_labels = [f"T{i+1}" for i in range(len(failure_times))]

parser = argparse.ArgumentParser(description="Plot odometry/target/commands from a rosbag2.")
parser.add_argument(
    "bag",
    nargs="?",
    default=DEFAULT_BAG,
    help="Path to rosbag2 file (.db3/.mcap). Defaults to example path.",
)
parser.add_argument(
    "--storage-id",
    default=DEFAULT_STORAGE_ID,
    choices=["sqlite3", "mcap"],
    help="rosbag2 storage id (default: sqlite3).",
)

args = parser.parse_args()
bag = args.bag
storage_id = args.storage_id

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id),
    rosbag2_py.ConverterOptions("cdr", "cdr"),
)

types = {t.name: t.type for t in reader.get_all_topics_and_types()}
if odom_topic not in types:
    raise SystemExit(f"Missing {odom_topic}. Found topics: {list(types)}")

Odom = get_message(types[odom_topic])
Target = get_message(types[target_topic]) if target_topic in types else None
ActionBounds = get_message(types[action_bounds_topic]) if action_bounds_topic in types else None
Thrust = get_message(types[thrust_cmd_topic]) if thrust_cmd_topic in types else None

# ---------- data containers ----------

# Odom series
times = []
xs, ys = [], []
vxs, vys = [], []
thetas, omegas = [], []    # orientation and yaw-rate
ell_bounds = []            # per-odom bounds snapshot

# Target trajectory (ZOH at odom timestamps)
txs, tys, tts = [], [], []
latest_target = None
target = None

# Action bounds
latest_bounds = None       # last seen (xmin,xmax,ymin,ymax)

# Thrust commands (ZOH) + bounds at each thrust sample
thr_times, fxs, fys = [], [], []
thr_bounds = []            # per-thrust bounds snapshot

t0 = None  # time zero (first message timestamp)


# ---------- helpers ----------

def _extract_xy_bounds(ab_msg):
    data = getattr(ab_msg, "data", None)
    if data is None:
        if hasattr(ab_msg, "__iter__"):
            data = list(ab_msg)
        else:
            return None
    data = list(data)
    if len(data) == 6:   # 3x2 -> use first 2 rows
        arr = np.array(data, dtype=float).reshape(3, 2)
    elif len(data) == 4: # 2x2
        arr = np.array(data, dtype=float).reshape(2, 2)
    else:
        return None
    xmin, xmax = float(arr[0, 0]), float(arr[0, 1])
    ymin, ymax = float(arr[1, 0]), float(arr[1, 1])
    return (xmin, xmax, ymin, ymax)


def _is_default_bounds(b, pm=DEFAULT_PM, tol=TOL):
    xmin, xmax, ymin, ymax = b
    x_ok = abs(xmin + pm) <= tol and abs(xmax - pm) <= tol
    y_ok = abs(ymin + pm) <= tol and abs(ymax - pm) <= tol
    return x_ok and y_ok


def _extract_fx_fy_tau(msg):
    if hasattr(msg, "x") and hasattr(msg, "y") and hasattr(msg, "z"):
        return float(msg.x), float(msg.y), float(msg.z)
    for attr in ("vector", "force"):
        sub = getattr(msg, attr, None)
        if sub is not None and all(hasattr(sub, k) for k in ("x", "y", "z")):
            return float(sub.x), float(sub.y), float(sub.z)
    try:
        return float(msg["x"]), float(msg["y"]), float(msg["z"])
    except Exception:
        return None


# ---------- read loop ----------

while reader.has_next():
    topic, data, t_nsec = reader.read_next()

    if topic == odom_topic:
        msg = deserialize_message(data, Odom)

        if t0 is None:
            t0 = t_nsec
        t = (t_nsec - t0) / 1e9

        # Position
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        # Linear velocities
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)

        # Orientation → yaw theta
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Angular velocity (yaw rate)
        omega = float(msg.twist.twist.angular.z)

        times.append(t)
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        thetas.append(theta)
        omegas.append(omega)
        ell_bounds.append(latest_bounds)  # snapshot bounds for this odom

        # ZOH target at odom timestamps
        if latest_target is not None:
            txs.append(latest_target[0])
            tys.append(latest_target[1])
            tts.append(latest_target[2])
        else:
            txs.append(np.nan)
            tys.append(np.nan)
            tts.append(np.nan)

    elif Target and topic == target_topic:
        tmsg = deserialize_message(data, Target)
        q = tmsg.pose.pose.orientation
        _, _, theta_ref = euler_from_quaternion([q.x, q.y, q.z, q.w])

        latest_target = (tmsg.pose.pose.position.x, tmsg.pose.pose.position.y, theta_ref)
        target = latest_target  # for latest marker

    elif ActionBounds and topic == action_bounds_topic:
        ab_msg = deserialize_message(data, ActionBounds)
        b = _extract_xy_bounds(ab_msg)
        if b is not None:
            latest_bounds = b

    elif Thrust and topic == thrust_cmd_topic:
        tmsg = deserialize_message(data, Thrust)
        if t0 is None:
            t0 = t_nsec
        t = (t_nsec - t0) / 1e9

        fx_fy_tau = _extract_fx_fy_tau(tmsg)
        if fx_fy_tau is not None:
            fx, fy, _tau = fx_fy_tau
            thr_times.append(t)
            fxs.append(fx)
            fys.append(fy)
            thr_bounds.append(latest_bounds)  # snapshot bounds for this thrust sample

if not xs:
    raise SystemExit("No odometry data")


# ---------- common arrays + failure-event mapping ----------

t_arr = np.array(times, dtype=float)
x_arr = np.array(xs, dtype=float)
y_arr = np.array(ys, dtype=float)

# map failure_times (from FTCNode) to nearest odom sample
fail_events = []  # list of (t_fail, x_fail, y_fail, label)
for i, t_fail in enumerate(failure_times):
    if t_fail is None:
        continue
    # if bag is shorter than t_fail, this will give the last index
    j = int(np.argmin(np.abs(t_arr - float(t_fail))))
    fail_events.append((
        t_arr[j],
        x_arr[j],
        y_arr[j],
        failure_labels[i] if i < len(failure_labels) else f"T{i+1}",
    ))


# ---------- Plot 1: XY trajectory ----------

plt.figure(figsize=(8, 4))
sc = plt.scatter(xs, ys, c=times, cmap="viridis")

# Continuous target trajectory (sampled ZOH at odom timestamps)
if txs and tys and tts:
    tmx = np.array(txs, dtype=float)
    tmy = np.array(tys, dtype=float)
    tmt = np.array(tts, dtype=float)
    mask = np.isfinite(tmx) & np.isfinite(tmy)
    if np.any(mask):
        plt.plot(tmx[mask], tmy[mask], label="target traj", linewidth=1)
        # latest target marker
        li = np.where(mask)[0][-1]
        plt.scatter(tmx[li], tmy[li], label="target (latest)", marker="x")

# square [-2,2]
sqx = [-2, 2, 2, -2, -2]
sqy = [-2, -2, 2, 2, -2]
plt.plot(sqx, sqy, linestyle="-.")

# Draw bounds ellipses centered on odom points
ax = plt.gca()
for x, y, b in zip(xs, ys, ell_bounds):
    if b is None or _is_default_bounds(b):
        continue
    xmin, xmax, ymin, ymax = b
    xmin = np.minimum(0, xmin)
    xmax = np.maximum(0, xmax)
    ymin = np.minimum(0, ymin)
    ymax = np.maximum(0, ymax)
    w = SCALE * abs(xmax - xmin)
    h = SCALE * abs(ymax - ymin)
    e = Ellipse((x, y), width=w, height=h)
    ax.add_patch(e)

# failure markers in XY
for t_fail, x_fail, y_fail, lab in fail_events:
    plt.scatter(x_fail, y_fail, color="red", marker="x", s=60)
    plt.text(x_fail, y_fail, f" {lab}", color="red", fontsize=8)

plt.axis("equal")
plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")
plt.colorbar(sc, label="Time (s)")
plt.legend()


# ---------- Plot 2: 3x2 grid of [x,y,theta] vs [vx,vy,omega] ----------

th_arr = np.array(thetas, dtype=float)
vx_arr = np.array(vxs, dtype=float)
vy_arr = np.array(vys, dtype=float)
om_arr = np.array(omegas, dtype=float)

tx_arr = np.array(txs, dtype=float)
ty_arr = np.array(tys, dtype=float)
tt_arr = np.array(tts, dtype=float)
ref_mask = np.isfinite(tx_arr) & np.isfinite(ty_arr) & np.isfinite(tt_arr)

fig_states, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
ax_x, ax_vx = axes[0, 0], axes[0, 1]
ax_y, ax_vy = axes[1, 0], axes[1, 1]
ax_th, ax_om = axes[2, 0], axes[2, 1]

# x(t) and x_ref
ax_x.plot(t_arr, x_arr, label="x")
if np.any(ref_mask):
    ax_x.plot(t_arr[ref_mask], tx_arr[ref_mask], ":", label="x_ref")
ax_x.set_ylabel("x [m]")
ax_x.legend()

# y(t) and y_ref
ax_y.plot(t_arr, y_arr, label="y")
if np.any(ref_mask):
    ax_y.plot(t_arr[ref_mask], ty_arr[ref_mask], ":", label="y_ref")
ax_y.set_ylabel("y [m]")
ax_y.legend()

# theta(t)
ax_th.plot(t_arr, (180/3.14)*th_arr, label=r"$\theta$")
if np.any(ref_mask):
    ax_th.plot(t_arr[ref_mask], (180/3.14)*tt_arr[ref_mask], ":", label="theta_ref")
ax_th.set_ylabel(r"$\theta$ [rad]")
ax_th.set_xlabel("Time [s]")
ax_th.legend()

# vx(t), ref = 0
ax_vx.plot(t_arr, vx_arr, label=r"$v_x$")
ax_vx.axhline(0.0, linestyle=":", label=r"$v_{x,\mathrm{ref}}$")
ax_vx.set_ylabel(r"$v_x$ [m/s]")
ax_vx.legend()

# vy(t), ref = 0
ax_vy.plot(t_arr, vy_arr, label=r"$v_y$")
ax_vy.axhline(0.0, linestyle=":", label=r"$v_{y,\mathrm{ref}}$")
ax_vy.set_ylabel(r"$v_y$ [m/s]")
ax_vy.legend()

# omega(t), ref = 0
ax_om.plot(t_arr, om_arr, label=r"$\dot{\theta}$")
ax_om.axhline(0.0, linestyle=":", label=r"$\dot{\theta}_\mathrm{ref}$")
ax_om.set_ylabel(r"$\dot{\theta}$ [rad/s]")
ax_om.set_xlabel("Time [s]")
ax_om.legend()

# vertical failure lines on all state plots
for t_fail, *_ in fail_events:
    for ax_state in (ax_x, ax_y, ax_th, ax_vx, ax_vy, ax_om):
        ax_state.axvline(t_fail, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

fig_states.tight_layout()


# ---------- Plot 3: Thrust cmds (2-row) with bounds (ZOH) ----------

if thr_times:
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for b in thr_bounds:
        if b is None:
            xmins.append(np.nan); xmaxs.append(np.nan)
            ymins.append(np.nan); ymaxs.append(np.nan)
        else:
            xmin, xmax, ymin, ymax = b
            xmins.append(xmin); xmaxs.append(xmax)
            ymins.append(ymin); ymaxs.append(ymax)

    fig, (ax_fx, ax_fy) = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    ax_fx.step(thr_times, fxs, where="post", label="$f_x$", linewidth=1)
    ax_fx.step(thr_times, xmins, where="post", label=r"$f_{x,\mathrm{min}}$", linewidth=0.9, linestyle=":")
    ax_fx.step(thr_times, xmaxs, where="post", label=r"$f_{x,\mathrm{max}}$", linewidth=0.9, linestyle=":")
    ax_fx.set_ylabel("$f_x$ (N)")
    ax_fx.legend()

    ax_fy.step(thr_times, fys, where="post", label="$f_y$", linewidth=1)
    ax_fy.step(thr_times, ymins, where="post", label=r"$f_{y,\mathrm{min}}$", linewidth=0.9, linestyle=":")
    ax_fy.step(thr_times, ymaxs, where="post", label=r"$f_{y,\mathrm{max}}$", linewidth=0.9, linestyle=":")
    ax_fy.set_xlabel("Time (s)")
    ax_fy.set_ylabel("$f_y$ (N)")
    ax_fy.legend()

    # vertical failure lines on thrust plots
    for t_fail, *_ in fail_events:
        ax_fx.axvline(t_fail, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_fy.axvline(t_fail, color="red", linestyle="--", linewidth=0.8, alpha=0.7)


# ---------- Save figs to same folder as bag ----------

out_dir = os.path.dirname(bag) or "."
for i, num in enumerate(sorted(plt.get_fignums()), start=1):
    fig = plt.figure(num)
    fig.savefig(os.path.join(out_dir, f"fig_{i}.png"))
plt.close("all")
