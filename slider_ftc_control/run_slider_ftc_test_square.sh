#!/usr/bin/env bash
set -Eeuo pipefail

source ~/slider_ws/install/setup.bash

echo "Starting Gazebo..."
ros2 launch slider_gazebo slider_launch.py &
GAZEBO_PID=$!

sleep 3

echo "Starting target point..."
ros2 launch target_trajectories target_point.launch.py &
TARGET_PID=$!

sleep 2

echo "Starting FTC pipeline..."
ros2 launch slider_ftc_control ftc_pipeline.launch.py &
FTC_PID=$!

sleep 5

echo "Injecting thruster_state..."
ros2 topic pub --once /thruster_state std_msgs/msg/UInt8MultiArray \
"data: [0, 1, 1, 1, 1, 1, 1, 1]"

echo "All running. Press Ctrl+C to stop."

trap 'echo "Stopping..."; kill $GAZEBO_PID $TARGET_PID $FTC_PID 2>/dev/null || true' INT TERM EXIT

wait