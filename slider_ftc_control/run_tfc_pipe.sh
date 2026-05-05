#!/usr/bin/env bash
set -Eeuo pipefail

WS="$HOME/slider_ws"
BAG_DIR="$WS/rosbags"
BAG_NAME="ftc_test"
BAG_PATH="$BAG_DIR/$BAG_NAME"

export ROS_DOMAIN_ID=0
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{name}] [{severity}] {message}"

PID_GAZEBO=""
PID_BAG=""
PID_TARGET=""
PID_FTC_PIPE=""

cleanup() {
  trap - SIGINT SIGTERM EXIT
  echo
  echo "=== CLEANUP ==="

  [[ -n "${PID_FTC_PIPE:-}" ]] && kill -INT "$PID_FTC_PIPE" 2>/dev/null || true
  [[ -n "${PID_TARGET:-}" ]] && kill -INT "$PID_TARGET" 2>/dev/null || true
  sleep 0.5

  if [[ -n "${PID_BAG:-}" ]]; then
    echo "Stopping rosbag..."
    kill -INT "$PID_BAG" 2>/dev/null || true
    wait "$PID_BAG" 2>/dev/null || true
  fi

  [[ -n "${PID_GAZEBO:-}" ]] && kill -INT "$PID_GAZEBO" 2>/dev/null || true
  sleep 0.5

  [[ -n "${PID_FTC_PIPE:-}" ]] && kill -TERM "$PID_FTC_PIPE" 2>/dev/null || true
  [[ -n "${PID_TARGET:-}" ]] && kill -TERM "$PID_TARGET" 2>/dev/null || true
  [[ -n "${PID_GAZEBO:-}" ]] && kill -TERM "$PID_GAZEBO" 2>/dev/null || true

  echo "Bag check:"
  ros2 bag info "$BAG_PATH" || true

  echo "=== Plotting bag ==="
  python3 "$WS/src/slider/utils/plot_rosbags.py" "$BAG_PATH" || true

  echo "Done."
  exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "=== Building workspace ==="
cd "$WS"
colcon build

set +u
source "$WS/install/setup.bash"
set -u

mkdir -p "$BAG_DIR"
rm -rf "$BAG_PATH"

echo "=== Starting Gazebo ==="
ros2 launch slider_gazebo slider_launch.py gui:=true &
PID_GAZEBO=$!


sleep 3

echo "=== Starting target node ==="
ros2 launch target_trajectories target_point.launch.py \
  target_x:=0.0 target_y:=0.0 target_theta:=0.0 &
PID_TARGET=$!

sleep 1

echo "=== Building AMS cache... ==="
ros2 run slider_ftc_control build_ams_cache

sleep 5
echo "=== Finished building AMS cache... ==="

echo "=== Starting rosbag ==="
ros2 bag record -o "$BAG_PATH" \
  /odom \
  /target_point \
  /thruster_state \
  /thrust_cmd \
  /eight_thrust_pulse &
PID_BAG=$!



echo "=== Starting FTC pipeline ==="
ros2 launch slider_ftc_control launch_ftc_pipe.launch.py &
PID_FTC_PIPE=$!




echo
echo "=== Running ==="
echo "Ctrl+C to stop."

while true; do
  sleep 1
done