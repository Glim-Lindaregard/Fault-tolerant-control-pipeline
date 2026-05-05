# Slider Fault-Tolerant Control / Allocation (FTC)

This package implements a modular fault-tolerant control pipeline for the planar slider spacecraft emulator.

It consists of:
- A **controller** (MPC → desired wrench)
- An **allocator** (wrench → thruster commands)
- A **thruster state publisher** (failure injection)

Each component can be used **independently or together**.

---

# System Architecture

```
/odom + /target_point + /thruster_state
                    ↓
              control_node
                    ↓  /thrust_cmd
              allocator_node
                    ↓  /eight_thrust_pulse
               Slider (Gazebo)
```

---

# Nodes

## 1. Controller (`control_node`)

Computes the desired wrench:

```
a_d = [Fx, Fy, τ]
```

### Subscribes to
- `/odom` → system state  
  `[x, y, θ, vx, vy, ω]`
- `/target_point` → reference trajectory (Odometry)
- `/thruster_state` → current failure configuration

### Publishes
- `/thrust_cmd` (`geometry_msgs/Vector3`)

### Key parameter

```yaml
bounds_mode: "box" | "ellipsoid" | "ams"
```

- `box` → fast, conservative
- `ellipsoid` → smooth approximation
- `ams` → exact feasible wrench set (slowest, most accurate)

---

## 2. Allocator (`allocator_node`)

Converts desired wrench into individual thruster commands.

### Subscribes to
- `/thrust_cmd`
- `/thruster_state`

### Publishes
- `/eight_thrust_pulse` (`std_msgs/UInt8MultiArray`)

### Behavior

- Uses precomputed AMS geometry
- Enforces failure constraints:
  - `0` → thruster OFF
  - `2` → thruster forced to max

---

## 3. Thruster State (`thruster_state`)

Publishes the current health state of each thruster.

### Convention

```
0 = passive failure (OFF)
1 = healthy
2 = active failure (stuck ON)
```

### Publishes
- `/thruster_state`

---

# Changing Failure Behavior

Failure behavior is configured in the **launch file**:

```
launch/launch_ftc_pipe.launch.py
```

Example:

```python
parameters=[{
    "failure_times":  [15.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "failure_states": [0,    1,    1,    1,    1,    1,    1,    1],
}]
```

### Meaning

- At `t = 15s`, thruster 1 → state `0` (OFF)
- `-1.0` → never fails

### Example cases

**Single failure**
```yaml
failure_times:  [10, -1, -1, -1, -1, -1, -1, -1]
failure_states: [0,   1,  1,  1,  1,  1,  1,  1]
```

**Multiple failures**
```yaml
failure_times:  [10, 20, -1, -1, -1, -1, -1, -1]
failure_states: [0,  2,  1,  1,  1,  1,  1,  1]
```

---

# Using Nodes Independently

You can run only parts of the pipeline by editing the launch file.

## Controller only

Comment out allocator:

```python
return LaunchDescription([
    controller_node,
])
```

Useful for:
- debugging MPC
- logging desired wrench

---

## Allocator only

Provide your own `/thrust_cmd`:

```bash
ros2 topic pub /thrust_cmd geometry_msgs/msg/Vector3 "{x: 0.5, y: 0.0, z: 0.0}"
```

Useful for:
- testing allocation
- validating AMS geometry

---

## Full pipeline

```bash
ros2 launch slider_ftc_control launch_ftc_pipe.launch.py
```

---

# AMS Cache (Required)

Both controller and allocator depend on a precomputed AMS cache.

## Build once before running

```bash
ros2 run slider_ftc_control build_ams_cache
```

This creates:

```
~/slider_ws/ams_cache.pkl
```

### Important

- Must be rebuilt if:
  - thruster configuration changes
  - system parameters change
- Both nodes load this file at startup

---

# Run Script (`run_ftc.sh`)

The `.sh` script automates the full pipeline:

### What it does

1. Builds workspace
2. Starts Gazebo
3. Starts rosbag recording
4. Starts target trajectory node
5. Launches FTC pipeline
6. Handles clean shutdown

### Run it

```bash
./run_tfc_pipe.sh
```

### Notes

- Bag is saved to:
  ```
  ~/slider_ws/rosbags/
  ```
- Use Ctrl+C to stop cleanly
- Plotting is optional and handled at shutdown

---

# Topics Summary

| Topic | Type | Description |
|------|------|------------|
| `/odom` | Odometry | Current state |
| `/target_point` | Odometry | Reference |
| `/thruster_state` | UInt8MultiArray | Failure configuration |
| `/thrust_cmd` | Vector3 | Desired wrench |
| `/eight_thrust_pulse` | UInt8MultiArray | PWM signals |

---

# Dependencies

- ROS2 Humble
- NumPy
- CasADi
- slider ecosystem:
  - `slider_gazebo`
  - `slider_description`
  - `target_trajectories`

---

# Notes

- Designed for **on/off thruster systems**
- Supports **passive and active failures**
- AMS ensures physically feasible control
- Modular design allows easy testing and extension

---

# Author

Glim Lindaregard  
MSc Space Engineering, LTU
