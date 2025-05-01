# CARLA LiDAR Simulation and Visualization

This repository contains code for simulating and visualizing LiDAR data in the CARLA autonomous driving simulator, focusing on phantom point detection and object recognition.

## Overview

This project uses the CARLA simulator to create realistic LiDAR point cloud data in various driving scenarios. It includes components for:

1. **Simulation**: Running a simulated vehicle with multiple LiDAR sensors through a scene with obstacles
2. **Autonomous Control**: Basic autonomous braking functionality based on LiDAR obstacle detection
3. **Visualization**: 3D point cloud visualization with color-coded detection results

The simulation can generate phantom points (false readings) in LiDAR data to test detection robustness. The visualization component allows for easy playback and analysis of simulation results.

## Repository Structure

```
├── CARLA_simulation/         # Simulation-related code
│   ├── config/               # Configuration parameters
│   ├── control/              # Autonomous control logic
│   ├── sensors/              # LiDAR and sensor processing
│   ├── simulation/           # Main simulation components
│   └── utils/                # Utility functions
├── visualization/            # Visualization code
│   ├── file_utils.py         # File handling utilities
│   ├── point_cloud_processor.py  # Point cloud processing
│   ├── ui_callbacks.py       # UI interaction handlers
│   └── visualization_manager.py  # Main visualization class
└── scripts/                  # Executable scripts
    ├── run_simulation.py     # Script to run simulations
    └── run_visualization.py  # Script to visualize results
```

## Requirements

- CARLA Simulator (tested with version 0.10.0)
- Python 3.7+
- NumPy
- Open3D (for visualization)

## Usage Instructions

### Running a Simulation

To run a simulation with default parameters:

```bash
python scripts/run_simulation.py
```

This will:
1. Connect to the CARLA server
2. Create a scene with a vehicle and pedestrian
3. Set up multiple LiDAR sensors
4. Run the simulation and collect data
5. Save the results to ../simulation_data/

### Visualizing Simulation Results

To visualize the most recent simulation:

```bash
python scripts/run_visualization.py
```

To visualize a specific simulation:

```bash
python scripts/run_visualization.py --dir="Simulation 1"
```

### Visualization Controls

- **SPACE**: Pause/Resume playback
- **LEFT/RIGHT**: Navigate frames
- **+/-**: Increase/decrease playback speed
- **Q**: Quit visualization

### Color Coding

- **BLUE**: Normal LiDAR points
- **RED**: Phantom points
- **GREEN**: Detection cluster points (points that triggered object detection)

## Simulation Features

- Multi-LiDAR setup (front bumper and roof-mounted)
- Configurable phantom point generation
- Realistic object detection
- Autonomous braking capabilities

## Simulation Output

Simulation results are stored in the `../simulation_data/` directory, organized by simulation number. Each simulation directory contains:

- Point cloud data for each LiDAR sensor
- Detection results
- Simulation summary