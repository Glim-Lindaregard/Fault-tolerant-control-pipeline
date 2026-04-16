#!/usr/bin/env bash
###############################################################################
# Slider FTC run script — Odometry target, updated via parameters (robust)
###############################################################################

set -Eeuo pipefail

# ------------------------- Configuration ------------------------------------
WS="$HOME/slider_ws"
BAG_DIR="$WS/rosbags"
BAG_NAME="ftc_test"
export ROS_DOMAIN_ID=0
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
PID_FTC=""      # will be the PID of the subshell running the FTC pipe
PID_BAG=""
PID_VICON=""

CLEANED_UP=0
# -----------------------------------------------------------------------------

cleanup() {
  [[ $CLEANED_UP -eq 1 ]] && return
  CLEANED_UP=1

  # Prevent re-entrancy while we clean up
  trap '' SIGINT SIGTERM

  echo
  echo "=== Shutting down cleanly ==="

  echo "Thrusters OFF..."
  ros2 topic pub --once /eight_thrust_pulse std_msgs/msg/UInt8MultiArray \
    "{data: [0,0,0,0,0,0,0,0]}" >/dev/null 2>&1 || true

  # 1) Stop publishers/controllers first (so bag can flush and not get spammed)
  echo "Stopping FTC/target/vicon..."
  kill -SIGINT ${PID_FTC:-} ${PID_TARGET:-} ${PID_VICON:-} 2>/dev/null || true
  sleep 0.5

  # 2) Stop rosbag LAST with SIGINT and wait for flush/finalize
  echo "Stopping rosbag (flush)..."
  if [[ -n "${PID_BAG:-}" ]]; then
    kill -SIGINT "$PID_BAG" 2>/dev/null || true
    wait "$PID_BAG" 2>/dev/null || true
  fi

  # 3) Stop gazebo after bag is finalized (Gazebo can exit non-zero on shutdown)
  echo "Stopping gazebo..."
  if [[ -n "${PID_GAZEBO:-}" ]]; then
    kill -SIGINT "$PID_GAZEBO" 2>/dev/null || true
    wait "$PID_GAZEBO" 2>/dev/null || true
  fi

  # 4) Plot only if the bag exists and is valid
  echo "Plotting latest bag..."
  if ros2 bag info "$BAG_DIR/$BAG_NAME" >/dev/null 2>&1; then
    DB3="$(ls -t "$BAG_DIR/$BAG_NAME"/*.db3 2>/dev/null | head -n 1)"
    if [[ -n "$DB3" ]]; then
      python3 \
        "$WS/src/slider/slider_ftc_control/slider_ftc_control/viz/plot_states.py" \
        "$DB3" || true
    else
      echo "Bag folder exists but no .db3 file found."
    fi
  else
    echo "No valid bag to plot (not finalized / empty / missing)."
  fi

  # 5) Keep your copy behavior, but don’t hard-crash if file is missing
  if [[ -f "$BAG_DIR/$BAG_NAME/${BAG_NAME}_0.db3" ]]; then
    cp "$BAG_DIR/$BAG_NAME/${BAG_NAME}_0.db3" "$WS/test1/TEMP.db3" || true
  fi

  echo "=== Done ==="
}

trap cleanup SIGINT SIGTERM EXIT
# -----------------------------------------------------------------------------

# ------------------------- Gazebo --------------------------------------------
#echo "Starting Gazebo..."
#ros2 launch slider_gazebo slider_launch.py gui:=true &
#PID_GAZEBO=$!
#sleep 5
# -----------------------------------------------------------------------------

#echo "Starting Vicon (mocap → odom)..."
ros2 launch mocap_pose_to_odom get_odom_vicon.launch.py \
 > /tmp/vicon.log 2>&1 &
PID_VICON=$!
#
## Give Vicon time to connect and start publishing
sleep 3

# ------------------------- Target node (Odometry) ----------------------------
#echo "Starting target trajectory node..."
#ros2 launch target_trajectories lissajous_trajectory.launch.py \
#  lissa_A:=0.4 lissa_B:=0.4 lissa_a:=1 lissa_b:=1 lissa_delta:=1.57 lissa_omega:=0.09 &

ros2 launch target_trajectories target_point.launch.py \
  target_x:=0.0 target_y:=0.0 target_theta:=0.0 &
PID_TARGET=$!
sleep 1
# -----------------------------------------------------------------------------

# ------------------------- FTC node (live telemetry) -------------------------
echo "Starting FTC node..."

# NOTE: Don't put comments after a line-continuation backslash in a pipeline.
# Choose one: box | ellipsoid | ams
BOUNDS_MODE="ams"

(
  stdbuf -oL ros2 run slider_ftc_control ftc_node --ros-args -p bounds_mode:="$BOUNDS_MODE" --ros-args -p plot_ams_planes:=true | \
  awk '
  /STATE:/ { state=$0 }
  /REF:/   { ref=$0 }
  {
    if (state && ref) {
      printf "\r%-120s\n%-120s", state, ref
      fflush()
    }
  }
  '
) &
PID_FTC=$!
sleep 1
# -----------------------------------------------------------------------------

# ------------------------- Rosbag --------------------------------------------
echo "Recording rosbag..."
ros2 bag record -o "$BAG_DIR/$BAG_NAME" \
  /odom /target_point /eight_thrust_pulse /thruster_state /wrench_cmd_pre_pwm &
PID_BAG=$!
# -----------------------------------------------------------------------------

# ------------------------- Helper: update target via params ------------------
set_target () {
  local X=$1
  local Y=$2
  local TH=$3

  echo "Target → ($X, $Y, $TH)"
  ros2 param set /target_point target_x "$X" >/dev/null || true
  ros2 param set /target_point target_y "$Y" >/dev/null || true
  ros2 param set /target_point target_theta "$TH" >/dev/null || true
}
# -----------------------------------------------------------------------------

echo
echo "=== Running stepped square (parameter updates) ==="
echo "Ctrl+C to stop"

# Keep the script alive; don't die on non-zero child exit
wait || true
