# Autonomous RC Car Navigation

A ROS-based navigation stack for a 1/10-scale autonomous RC car. The project implements three independent navigation strategies — **line/wall following**, **gap-barrier navigation**, and **probabilistic occupancy grid mapping** — all driven by a 2-D LiDAR and wheel odometry. Vehicle dynamics follow the standard bicycle model with Ackermann steering.

---

## Repository Structure

```
.
├── navigation.py               # Wall / line-following controller (PD)
├── navigation_gap_barrier.py   # Gap-following + virtual-barrier controller
├── occupancygridmap.py         # Bayesian log-odds occupancy grid mapper
├── params.yaml                 # All vehicle and algorithm parameters
```

---

## Algorithms

### 1. Wall / Line Following (`navigation.py`)

The `WallFollow` node centres the car between two detected walls using a feedback-linearising PD controller.

- Two pairs of LiDAR beams sample each wall surface; a Cartesian fit yields the perpendicular distance and surface tangent direction for both walls.
- The signed lateral error `d_lr = d_l − d_r` and its derivative (estimated from wall geometry and current speed) drive a PD law that outputs a desired steering angle `δ`.
- Forward speed is regulated with an exponential profile that slows the car as it approaches a frontal obstacle.

Key parameters: `kp_linefollow`, `kd_linefollow`, `vs_des`, `d_stop`, `d_tau`, `bl_angle_deg` / `al_angle_deg` / `br_angle_deg` / `ar_angle_deg`.

---

### 2. Gap-Barrier Navigation (`navigation_gap_barrier.py`)

The `GapBarrier` node finds the widest free corridor in the LiDAR scan and drives towards it while respecting virtual side barriers.

**Pipeline per scan:**
1. **Preprocess** — clamp invalid ranges, extract the configurable forward FOV, apply a moving-average smoothing window, and inflate each obstacle by the vehicle half-width + clearance margin so that the resulting gaps are guaranteed to be physically drivable.
2. **Find max gap** — scan the processed ranges for the longest contiguous run of non-zero values; reject gaps narrower than `min_gap_beams`.
3. **Select best point** — score each beam in the gap by its range weighted by a forward-bias penalty on lateral angle; pick the centroid of the top-5 candidates.
4. **Virtual barriers** — project nearby obstacle points onto the heading axis and laterally to estimate left/right corridor offsets (with exponential smoothing); compute a centre-line error.
5. **Control** — blended heading + centre-line PD with a first-order steer filter; speed scaled by corridor clearance and steering magnitude.

Key parameters: `fov_deg`, `safe_dist`, `k_heading`, `k_center`, `barrier_lookahead`, `smoothing_window`, `angle_bias`, `steer_filter`, `creep_speed`.

---

### 3. Occupancy Grid Mapping (`occupancygridmap.py`)

The `OccupancyGridMap` node maintains a probabilistic 2-D map of the environment in the odometry frame.

- Each cell is tracked in **log-odds** space to avoid numerical overflow and allow efficient additive updates.
- **Inverse sensor model** per scan:
  - *Occupied* — beam endpoint lies within the cell bounding box → `lo_hit`
  - *Free* — beam passes beyond the cell → `lo_miss`
  - *Unknown* — beam falls short of the cell → no update
- Log-odds are clamped to `±50` and converted to the standard ROS `[0, 100]` occupancy scale for publishing.
- The grid is centred on the robot's starting position; size and resolution are configurable.

Key parameters: `map_width`, `map_height`, `map_res`, `p_occ`, `p_free`.

---

## Dependencies

| Dependency | Version |
|---|---|
| ROS (Robot Operating System) | Melodic / Noetic |
| Python | 2.7 / 3.x |
| `numpy` | ≥ 1.16 |
| `opencv-python` (`cv2`) | ≥ 3.4 |
| ROS packages | `sensor_msgs`, `nav_msgs`, `ackermann_msgs`, `tf`, `tf2_ros` |

Install Python dependencies:
```bash
pip install numpy opencv-python
```

Install ROS message packages (example for Noetic):
```bash
sudo apt install ros-noetic-ackermann-msgs ros-noetic-tf2-ros
```

---

## Configuration

All tuneable parameters live in `params.yaml`. Load it via a ROS launch file using the `rosparam` tag, or push it onto the parameter server directly:

```bash
rosparam load params.yaml
```

Key sections in `params.yaml`:

| Section | Description |
|---|---|
| Vehicle model | `wheelbase`, `width`, `mass`, speed and steering limits |
| LiDAR | `scan_beams`, `scan_field_of_view`, `scan_range` |
| Wall following | `kp_linefollow`, `kd_linefollow`, beam angles |
| Gap / barrier | `fov_deg`, `k_heading`, `k_center`, `barrier_lookahead` |
| Occupancy map | `map_width`, `map_height`, `map_res`, `p_occ`, `p_free` |
| ROS topics | All topic names for drive, scan, odometry, map |

---

## Running the Nodes

Each node is a standalone ROS node. Launch with your preferred method; an example using `rosrun`:

```bash
# Start ROS core
roscore

# Load parameters
rosparam load params.yaml

# Wall / line following
rosrun <your_package> navigation.py

# OR gap-barrier navigation
rosrun <your_package> navigation_gap_barrier.py

# Occupancy grid mapping (can run alongside either navigation node)
rosrun <your_package> occupancygridmap.py
```

The navigation nodes publish `AckermannDriveStamped` messages to the topic configured by `nav_drive_topic` (default: `/nav`). This is fed into the multiplexer (`mux`) that selects the active drive source.

---

## Vehicle Parameters

The codebase targets a 1/10-scale RC car with the following physical characteristics:

| Parameter | Value |
|---|---|
| Wheelbase | 0.287 m |
| Track width | 0.342 m |
| Mass | 3.47 kg |
| Max speed | 1.5 m/s |
| Max steering angle | 0.419 rad (~24°) |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
