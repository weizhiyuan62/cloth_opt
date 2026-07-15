# ClothOpt

A demo for cloth simulation with intelligent control.

## Python optimization interface
- use `pybind` to build the interface to the simulator

The original `Visualization`, `Simulation`, and `Optimization` executables are
unchanged and can still be built with the original CMake project.

### Install

Python 3.10+ and a C++17 compiler are required. The build dependencies and the
C++ extension are handled by pip:

```bash
cd cloth_opt
conda create -n clothopt python=3.10.16 -y
conda activate clothopt
pip install -e .
```

- run pip install -e . again if c++ source code modified.


### rollout

`scripts/demo.py` is the position-control-only rollout entry point. 

```bash
conda activate clothopt
python scripts/demo.py
```

Rendering is disabled by default for the generic demo. Select the native
Polyscope renderer or the Matplotlib debug renderer explicitly:

```bash
python scripts/demo.py render=native
python scripts/demo.py render=debug
```

The render choices used by every script are:

```text
render=disabled  no frames or video
render=native    Polyscope/OpenGL, matching the original simulator renderer
render=debug     Matplotlib with controlled and target markers
```

`native` uses the same Polyscope rendering stack as the original simulator
executables. Its defaults reproduce `cpp/apps/optimization.cpp`: cloth color
`[0.3, 0.7, 0.9]`, wax material, edge width `1.0`, gray ground
`[0.5, 0.5, 0.5]`, red control points, and the same camera offsets. The green
target marker is an additional Python-rollout diagnostic. All of these values
can be adjusted under `configs/render/native.yaml`; for example:

```bash
python scripts/symmetric_fold.py render=native \
  render.appearance.background_color='[0.95,0.95,0.95]' \
  render.appearance.cloth_color='[0.8,0.2,0.2]' \
  render.camera.height=1.0
```

Leaving `background_color: null` preserves Polyscope's original default
background. On Linux without a display, Polyscope is allowed to select its EGL
backend automatically. To force the cloud renderer, use
`render=native render.engine=egl`; an EGL-capable driver must be installed on
that server. Both video backends save one fixed camera to `trajectory.mp4`
through ffmpeg.


### Symmetric-fold state machine

The symmetric policy folds one complete half of the cloth onto the other. Its
fixed high-level state machine is:

```text
initial_settle -> lift -> transfer -> place -> hold -> release -> final_settle
```

Only the 10 vertices on the outer moving edge receive staged rotational
position targets. The intervening moving-half rows remain physical, allowing
gravity and spring forces to create curvature. The default cloth is the
original **10 x 10 vertices**. Because an even grid has no row exactly on the
center, the geometric fold axis lies between rows 4 and 5, while stationary
row 5 is pinned as the discrete crease line. At release, the edge controller is
cleared and the moving surface remains unpinned during final settling.
Run the hand-initialized baseline with:

```bash
conda activate clothopt
python scripts/symmetric_fold.py
```

Native rendering is enabled by default for this script. Select the diagnostic
Matplotlib view with:

```bash
python scripts/symmetric_fold.py render=debug
```

The baseline parameters and objective weights are defined in
`configs/policy/symmetric_fold.yaml`. Individual values can be overridden from
the command line:

```bash
python scripts/symmetric_fold.py \
  policy.params.lift_angle=50 \
  policy.params.transfer_angle=170 \
  policy.params.transfer_frames=160
```

Use the cross-entropy method (CEM) to optimize the continuous parameters of
the same state machine:

```bash
python scripts/optimize.py
```

For a small cloud smoke experiment before launching the full search:

```bash
python scripts/optimize.py \
  optimizer.population_size=4 \
  optimizer.iterations=1
```

The default search optimizes the staged rotation angles, layer gap, phase
durations, position gain, and maximum force. Bounds and CEM settings are in
`configs/optimizer/cem.yaml`. The output
contains:

```text
outputs/optimize/<date>/<time>/
├── .hydra/
├── optimization_history.json
├── optimization_summary.json
└── best/
    ├── parameters.json
    ├── metrics.json
    ├── actions.json
    ├── trajectory.npz
    └── final.obj
```

The optimization objective and success checks are aligned with the folding
criteria:

- final vertex-pair alignment measures target-fold accuracy;
- commanded-speed effort is used as an energy proxy because the C++ interface
  does not expose actuator power;
- trajectory-wide mean and maximum grid-edge stretch measure structural
  distortion;
- target acceleration, terminal speed, and layer penetration proxies penalize
  abrupt motion, residual motion, and implausible overlap;
- `aesthetic_quality` is a deterministic 0--100 composite of alignment,
  distortion, smoothness, and penetration, with success requiring at least 70.

The aesthetic score is an optimization proxy rather than a learned human
perceptual rating. `structural_integrity` additionally requires the maximum
relative structural-edge stretch to stay below the configured threshold.

Important limitation: the current C++ engine does not yet implement cloth
self-collision. The penetration term can discourage invalid final-layer
placement, but cannot detect all triangle-triangle intersections throughout a
rollout. The benchmark is therefore still a geometry/trajectory proxy rather
than a complete contact benchmark.

### Surface-constrained diagonal fold

The diagonal-fold policy uses `controlled_corner` to select the triangular side
that moves. It position-controls every movable vertex on the two outer boundary
edges adjacent to that corner. On the default 10 x 10 grid this gives 17
controlled vertices (`2N - 3` on an `N x N` grid): the shared corner is counted
once, while the two endpoints on the fold diagonal are excluded because they
belong to the pinned crease. The interior of the moving triangle remains free
and follows through gravity, springs, and constraints.

Only the full corner-to-corner diagonal is pinned. The default rule moves the
top and right edges around the main diagonal (top left to bottom right):

```text
initial_settle -> grasp_hold -> early_lift -> rotate -> place
               -> hold -> release -> final_settle
```

Every controlled boundary vertex follows its own circular path generated by a
3-D rotation of its measured initial position around the diagonal crease. This
keeps path distances proportional to the actual cloth geometry instead of a
fixed displacement. Run the baseline and save its single-camera video with the
default native renderer:

```bash
python scripts/diagonal_fold.py
```

Disable rendering for rollout-only experiments:

```bash
python scripts/diagonal_fold.py render=disabled
```

Trajectory rules are in `configs/policy/diagonal_fold.yaml` and can be overridden
through Hydra:

```bash
python scripts/diagonal_fold.py \
  policy.params.rotate_frames=160 \
  policy.params.place_frames=60 \
  policy.params.max_force=200
```

Supported diagonal/corner combinations are:

```text
main diagonal: top_right or bottom_left
anti diagonal: top_left or bottom_right
```

For example, fold the bottom-right corner around the anti-diagonal with:

```bash
python scripts/diagonal_fold.py \
  policy.setup.fold_diagonal=anti \
  policy.setup.controlled_corner=bottom_right
```

At release, all boundary controllers are cleared without pinning the moving
triangle, allowing final settling. The result includes `trajectory.npz`,
`actions.json`, `metrics.json`, `final.obj`, and, when rendering is enabled,
`trajectory.mp4`.

### Gravity after the integrator fix

The active C++ integration path now divides accumulated force by per-vertex
mass. This matches gravity construction (`vertex_mass * gravity`), so the Python
wrapper passes the configured physical gravity directly to C++ without an
additional multiplier. All rollout configs inherit `[0, -9.81, 0]` from
`configs/env/default.yaml`. A different experimental gravity can still be set
explicitly, for example `env.scene.gravity=[0,-15,0]`.

Because the C++ extension changed, rebuild the editable package in the active
Conda environment before running Python:

```bash
pip install -e .
```

Both rule policies derive their controlled/pinned indices from `width` and
`height`, and derive rotation axes and path radii from the measured initial
vertex positions. Objective terms with length units are normalized by the cloth
extent; `success_alignment` is therefore a fraction of cloth extent and
`success_max_speed` is measured in cloth extents per second. Absolute height
parameters such as `layer_gap` intentionally remain in simulator length units.
Symmetric fold currently requires an even grid so the virtual center axis
remains unambiguous; diagonal fold supports square grids. These constraints make
runs reproducible across different supported cloth resolutions and sizes.

## System Requirements (Tested)

- **OS**: Ubuntu 24.04 LTS
- **Compiler**: GCC 11+ or Clang 14+
- **CMake**: 3.16+
- **Graphics**: OpenGL 3.3+ support

## Installation on Ubuntu 24.04

### 1. Install System Dependencies

```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install build-essential cmake git

# Install graphics and windowing libraries
sudo apt install libgl1-mesa-dev libglu1-mesa-dev libglfw3-dev

# Install linear algebra library
sudo apt install libeigen3-dev

# Install Intel TBB for parallel processing
sudo apt install libtbb-dev

# Install additional dependencies
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

### 2. Build

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project (use -j for parallel compilation)
make -j$(nproc)

# Verify build success
ls ./bin/
```

You should see these executables:
- `Visualization` - Basic waving cloth visualization
- `Simulation` - Cloth simulation with collide
- `Optimization` - Edge pulling demo with control system

### 3. Run Examples

```bash
# Run the cloth edge pulling demo
./bin/Optimization

# Run basic simulation collide with sphere
./bin/Simulation

# Run visualization example
./bin/Visualization
```

#### Demo Screenshots

<div align="center">

**Basic Visualization (`./bin/Visualization`)**
![Basic Visualization](image/visual.png)
*Fundamental cloth waving animation with material properties showcase*

**Collision Simulation (`./bin/Simulation`)**
![Edge Pulling Demo](image/sim.png)
*Cloth physics simulation with sphere collision detection and realistic draping*

**Edge Pulling Demo (`./bin/Optimization`)**
![Collision Simulation](image/control.png)
*Interactive cloth control with edge manipulation and real-time parameter adjustment*


</div>

## Project Structure

```
cloth_opt/
├── cpp/
│   ├── include/cloth_opt/   # Public C++ headers
│   ├── src/                 # C++ simulation core
│   ├── apps/                # Visualization, simulation, and control executables
│   └── bindings/            # pybind11 extension build
├── src/cloth_opt/
│   ├── sim/                 # Engine, Env, Action, and rendering
│   │   └── engine/          # C++/pybind11 engine adapter
│   ├── policy/
│   │   ├── symmetric_fold/  # Symmetric-fold rule and rollout I/O
│   │   └── diagonal_fold/   # Diagonal-fold rule and rollout I/O
│   └── optimization/        # Policy-parameter optimizers
├── scripts/                 # Python rollout entry points
├── configs/                 # Hydra env, policy, render, and optimizer groups
├── pyproject.toml
└── CMakeLists.txt
```

## Usage Examples

### Basic Edge Pulling Demo

The main demo (`./bin/Optimization`) showcases:

1. **10×10 cloth grid** with realistic physics
2. **Edge control system** - pull bottom edge in different directions
3. **Interactive GUI** with real-time parameter adjustment
4. **Multiple pull directions**: Forward, Backward, Left, Right, Up

**Controls:**
- Select pull direction with radio buttons
- Adjust pull distance, height, and strength with sliders
- Click "Start Pulling" to apply control forces
- Use "Reset Cloth" to return to initial state

### Advanced Control Features

The control system (`controller.h/cpp`) provides:

```cpp
// Position control - move vertices to target positions
controller.addPositionControl(vertexIndex, targetPosition, gain, maxForce);

// Velocity control - set target velocities  
controller.addVelocityControl(vertexIndex, targetVelocity, gain, maxForce);

// Force control - apply external forces
controller.addForceControl(vertexIndex, force);

// Trajectory following - animated paths
controller.setTrajectory(vertexIndex, waypoints, times, loop);

// Motion patterns - circular and sinusoidal motion
controller.addCircularMotion(vertexIndex, center, radius, frequency);
controller.addSinusoidalMotion(vertexIndex, center, amplitude, frequency);
```

## Physics Configuration

### Cloth Properties
```cpp
cloth.properties.stiffness = 800.0;         // Spring stiffness
cloth.properties.bendingStiffness = 20.0;   // Bending resistance  
cloth.properties.damping = 0.9;             // Energy dissipation
cloth.properties.friction = 0.8;            // Surface friction
cloth.properties.gravity = Eigen::Vector3d(0, -9.81, 0);
```

### Performance Optimization
- **Parallel processing** with Intel TBB
- **Optimized collision detection** for real-time performance
- **Adaptive time stepping** for stability
- **Memory-efficient** data structures
