#!/usr/bin/env bash
###############################################################################
# Slider FTC run script — clean terminal (FTC logs visible, Gazebo spam to logs)
###############################################################################
set -Eeuo pipefail

# ------------------------- Configuration ------------------------------------
WS="$HOME/slider_ws"
BAG_DIR="$WS/rosbags"
BAG_NAME="ftc_test"
ros2 daemon stop
sleep 1
ros2 daemon start
sleep 1

export ROS_DOMAIN_ID=0

export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{name}] [{severity}] {message}"

LOG_DIR="/tmp/slider_run_logs"
mkdir -p "$LOG_DIR"
# -----------------------------------------------------------------------------

# ------------------------- Build phase ---------------------------------------
echo "Building workspace..."
cd "$WS"
colcon build

# ROS setup scripts are not -u safe
set +u
source "$WS/install/setup.bash"
set -u
# -----------------------------------------------------------------------------

# ------------------------- Bag setup -----------------------------------------
mkdir -p "$BAG_DIR"
rm -rf "$BAG_DIR/$BAG_NAME"
# -----------------------------------------------------------------------------

# ------------------------- Runtime bookkeeping -------------------------------
PID_GAZEBO=""
PID_TARGET=""
PID_FTC=""
PID_BAG=""
PID_VICON=""

PGID_GAZEBO=""
PGID_TARGET=""
PGID_FTC=""
PGID_BAG=""
PGID_VICON=""

CLEANED_UP=0
# -----------------------------------------------------------------------------

get_pgid() {
  local pid="$1"
  [[ -z "${pid:-}" ]] && return 0
  ps -o pgid= "$pid" 2>/dev/null | tr -d '[:space:]' || true
}

kill_group_int() {
  local pgid="$1"
  [[ -z "${pgid:-}" ]] && return 0
  kill -INT -- "-$pgid" 2>/dev/null || true
}

kill_group_hard() {
  local pgid="$1"
  [[ -z "${pgid:-}" ]] && return 0
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep 0.3
  kill -KILL -- "-$pgid" 2>/dev/null || true
}

cleanup() {
  [[ $CLEANED_UP -eq 1 ]] && return
  CLEANED_UP=1

  trap '' SIGINT SIGTERM EXIT

  echo
  echo "=== Shutting down cleanly ==="

  echo "Thrusters OFF..."
  ros2 topic pub --once /eight_thrust_pulse std_msgs/msg/UInt8MultiArray \
    "{data: [0,0,0,0,0,0,0,0]}" >/dev/null 2>&1 || true

  # Resolve PGIDs
  PGID_FTC="${PGID_FTC:-$(get_pgid "${PID_FTC:-}")}"
  PGID_TARGET="${PGID_TARGET:-$(get_pgid "${PID_TARGET:-}")}"
  PGID_VICON="${PGID_VICON:-$(get_pgid "${PID_VICON:-}")}"
  PGID_BAG="${PGID_BAG:-$(get_pgid "${PID_BAG:-}")}"
  PGID_GAZEBO="${PGID_GAZEBO:-$(get_pgid "${PID_GAZEBO:-}")}"

  # 1) Stop controllers/publishers first
  echo "Stopping FTC/target/vicon..."
  kill_group_int "$PGID_FTC"
  kill_group_int "$PGID_TARGET"
  kill_group_int "$PGID_VICON"
  sleep 0.7

  # 2) Stop rosbag last and wait for flush
  echo "Stopping rosbag (flush)..."
  if [[ -n "${PID_BAG:-}" ]]; then
    kill_group_int "$PGID_BAG"
    wait "$PID_BAG" 2>/dev/null || true
  fi

  # 3) Stop gazebo after bag is finalized
  echo "Stopping gazebo..."
  kill_group_int "$PGID_GAZEBO"
  sleep 0.7

  # 4) Hard kill leftovers (just in case)
  echo "Ensuring processes are down..."
  kill_group_hard "$PGID_FTC"
  kill_group_hard "$PGID_TARGET"
  kill_group_hard "$PGID_VICON"
  kill_group_hard "$PGID_GAZEBO"

  # 5) Plot only if bag exists and is valid
  echo "Plotting latest bag..."
  if ros2 bag info "$BAG_DIR/$BAG_NAME" >/dev/null 2>&1; then
    DB3="$(ls -t "$BAG_DIR/$BAG_NAME"/*.db3 2>/dev/null | head -n 1 || true)"
    if [[ -n "$DB3" ]]; then
      python3 "$WS/src/slider/slider_ftc_control/slider_ftc_control/viz/plot_states.py" "$DB3"
    else
      echo "Bag folder exists but no .db3 file found."
    fi
  else
    echo "No valid bag to plot (not finalized / empty / missing)."
  fi

  # 6) Keep your copy behavior
  if [[ -f "$BAG_DIR/$BAG_NAME/${BAG_NAME}_0.db3" ]]; then
    cp "$BAG_DIR/$BAG_NAME/${BAG_NAME}_0.db3" "$WS/test1/TEMP.db3" || true
  fi

  echo "Logs saved in: $LOG_DIR"
  echo "=== Done ==="
}

trap cleanup SIGINT SIGTERM EXIT
# -----------------------------------------------------------------------------

# ------------------------- Helper: update target via params ------------------


set_target () {
  local X=$1
  local Y=$2
  local TH=$3
  echo "Target → ($X, $Y, $TH)"
  ros2 param set /target_point target_x "$X" >/dev/null 2>&1 || true
  ros2 param set /target_point target_y "$Y" >/dev/null 2>&1 || true
  ros2 param set /target_point target_theta "$TH" >/dev/null 2>&1 || true
}
# -----------------------------------------------------------------------------

# ------------------------- Gazebo --------------------------------------------
echo "Starting Gazebo (logs -> $LOG_DIR/gazebo.log)..."
setsid ros2 launch slider_gazebo slider_launch.py gui:=true \
  >"$LOG_DIR/gazebo.log" 2>&1 &
PID_GAZEBO=$!
PGID_GAZEBO="$(get_pgid "$PID_GAZEBO")"
sleep 5
# -------------------------- Vicon --------------------------------------------
#echo "Starting Vicon (mocap → odom)..."
#ros2 launch mocap_pose_to_odom get_odom_vicon.launch.py \
# > /tmp/vicon.log 2>&1 &
#PID_VICON=$!

# Give Vicon time to connect and start publishing
#sleep 3
# -----------------------------------------------------------------------------

pkill -f "/target_trajectories/target_point" || true
sleep 1
# ------------------------- Target node ---------------------------------------    #FOR TESTING target_x:=1.0 target_y:=1.0 target_theta:=2.356 \
echo "Starting target node (logs -> $LOG_DIR/target.log)..."
setsid ros2 launch target_trajectories target_point.launch.py \
  target_x:=0.0 target_y:=0.0 target_theta:=0.0 \
  >"$LOG_DIR/target.log" 2>&1 &
PID_TARGET=$!
PGID_TARGET="$(get_pgid "$PID_TARGET")"
sleep 2
# -----------------------------------------------------------------------------

# ------------------------- FTC node ------------------------------------------
# THIS is the one you want visible in terminal (your get_logger().info prints)
echo "Starting FTC node"
BOUNDS_MODE="ams"  # box | ellipsoid | ams

# Keep it on terminal, but ALSO log to file
setsid bash -lc "
  stdbuf -oL ros2 run slider_ftc_control ftc_node \
    --ros-args -p bounds_mode:='$BOUNDS_MODE' \
    --ros-args -p plot_ams_planes:=false \
  2>&1 | tee -a '$LOG_DIR/ftc.log'
" &
PID_FTC=$!
PGID_FTC="$(get_pgid "$PID_FTC")"
sleep 1
# -----------------------------------------------------------------------------

# ------------------------- Rosbag --------------------------------------------
echo "Recording rosbag (logs -> $LOG_DIR/rosbag.log)..."
setsid ros2 bag record -o "$BAG_DIR/$BAG_NAME" \
  /odom /target_point /eight_thrust_pulse /wrench_cmd \
  >"$LOG_DIR/rosbag.log" 2>&1 &
PID_BAG=$!
PGID_BAG="$(get_pgid "$PID_BAG")"
# -----------------------------------------------------------------------------

echo
echo "=== Running stepped square (parameter updates) ==="
echo "Ctrl+C to stop"

#sleep 8
#set_target 0.0 0.5 1.56
#sleep 6
#set_target 0.0 0.0 1.56
#sleep 5
#set_target 0.0 0.0 -1.56


wait || true


#python3 plot_states.py ~/slider_ws/rosbags/ftc_test/ftc_test_0.db3

#python3 plot_3runs.py \
#  --healthy /path/to/healthy_bag \
#  --fail1   /path/to/fail1_bag \
#  --fail2   /path/to/fail2_bag \
#  --name paper