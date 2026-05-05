# Slider Fault-Tolerant Control (FTC)

This package implements a fault-tolerant control pipeline for the planar slider spacecraft emulator.  
It is designed for use within a ROS2-based simulation and experimental setup.

---

## Overview

The pipeline is split into three modular nodes:

```
/odom + /target_point
        ↓
   Controller (MPC)
        ↓  /thrust_cmd
   Allocator (AMS-based)
        ↓  /eight_thrust_pulse
   Slider (Gazebo / hardware)
```

Additionally, a dedicated node publishes the current thruster health state.

---

## Nodes

### 1. Controller (`control_node`)

- Computes desired wrench:
  ```
  a_d = [Fx, Fy, τ]
  ```
- Uses MPC with selectable constraints:
  - `box`
  - `ellipsoid`
  - `ams`

**Subscribes to:**
- `/odom`
- `/target_point`
- `/thruster_state`

**Publishes:**
- `/thrust_cmd` (`geometry_msgs/Vector3`)

---

### 2. Allocator (`allocator_node`)

- Maps desired wrench → individual thruster commands
- Uses Attainable Moment Set (AMS) geometry
- Applies failure handling:
  - Passive failure → force = 0
  - Active failure → force = max

**Subscribes to:**
- `/thrust_cmd`
- `/thruster_state`

**Publishes:**
- `/eight_thrust_pulse` (`std_msgs/UInt8MultiArray`)

---

### 3. Thruster State (`thruster_state`)

Publishes health state of each thruster.

**Convention:**
```
0 = passive failure (OFF)
1 = healthy
2 = active failure (stuck ON)
```

**Publishes:**
- `/thruster_state`

Supports:
- configurable failure times
- configurable failure modes

---

## AMS Cache

AMS geometry is precomputed and stored to disk to avoid recomputation at runtime.

### Build cache

```bash
ros2 run slider_ftc_control build_ams_cache
```

This generates:

```
~/slider_ws/ams_cache.pkl
```

Both controller and allocator load this file at startup.

---

## Launch

Run the full FTC pipeline:

```bash
ros2 launch slider_ftc_control launch_ftc_pipe.launch.py
```

---

## Parameters

### Controller

```yaml
bounds_mode: "box" | "ellipsoid" | "ams"
```

### Thruster State

```yaml
failure_times:  [t1, t2, ..., t8]
failure_states: [s1, s2, ..., s8]
```

---

## Dependencies

- ROS2 Humble
- NumPy
- CasADi (for MPC)
- Custom slider packages:
  - `slider_gazebo`
  - `slider_description`
  - `target_trajectories`

---

## Notes

- AMS-based allocation assumes precomputed geometry
- Designed for on/off thruster systems
- Focus is on passive fault-tolerant control

---

## Author

Glim Lindaregard  
MSc Space Engineering, LTU
