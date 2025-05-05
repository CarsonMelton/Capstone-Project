# CARLA LiDAR Simulation and Visualization

This repository contains code for simulating and visualizing LiDAR data in the CARLA autonomous driving simulator, focusing on phantom point detection and object recognition.

## Overview

This project uses the CARLA simulator to create realistic LiDAR point cloud data in various driving scenarios. It includes components for:

1. **Simulation**: Running a simulated vehicle with multiple LiDAR sensors through a scene with obstacles
2. **Autonomous Control**: Basic autonomous braking functionality based on LiDAR obstacle detection
3. **Visualization**: 3D point cloud visualization with color-coded detection results

The simulation can generate phantom points (false readings) in LiDAR data to test detection robustness. The visualization component allows for easy playback and analysis of simulation results.

## Dependency Requirements

| Dependency | Minimum Version |
|------------|-----------------|
| Python | 3.7+ |
| CARLA Simulator | 0.10.0 |
| NumPy | 1.19.0 |
| Open3D | 0.13.0 |
| SciPy | 1.6.0 |
| Matplotlib | 3.3.0 

## Repository Structure

```
├── CARLA_simulation/         # Simulation-related code
│   ├── config/               # Configuration parameters
│   │   └── simulation_config.py  # Simulation parameter settings
│   ├── control/              # Autonomous control logic
│   │   └── autonomous_controller.py  # Braking and control systems
│   ├── sensors/              # LiDAR and sensor processing
│   │   ├── lidar_processor.py    # Point cloud processing algorithms
│   │   └── sensor_callbacks.py   # Sensor data callback handlers
│   ├── simulation/           # Main simulation components
│   │   └── carla_simulation.py   # Main simulation manager
│   └── utils/                # Utility functions
│       └── file_manager.py   # Data storage and organization
├── visualization/            # Visualization code
│   ├── file_utils.py         # File handling utilities
│   ├── point_cloud_processor.py  # Point cloud processing
│   ├── ui_callbacks.py       # UI interaction handlers
│   └── visualization_manager.py  # Main visualization class
└── scripts/                  # Executable scripts
    ├── run_simulation.py     # Script to run simulations
    └── run_visualization.py  # Script to visualize results
```

## Installation Guide

### CARLA Installation

For installing CARLA 0.10.0, please refer to the official installation and setup instructions at:
[https://carla.org/2024/12/19/release-0.10.0/](https://carla.org/2024/12/19/release-0.10.0/)

The official guide provides detailed instructions for different platforms and configurations.

### Project Setup

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/CarsonMelton/Capstone-Project.git
cd Capstone-Project
```

## Usage Instructions

### Running a Simulation

1. Start the CARLA server first:
```bash
# Navigate to your CARLA directory
cd /path/to/carla

# Start CARLA (Linux)
./CarlaUE4.sh -quality-level=Low

# Or on Windows
# CarlaUE4.exe -quality-level=Low
```

2. In a different window, run the simulation with default parameters:
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

- **BLUE**: Normal LiDAR points (intensity modulated)
- **RED**: Phantom points
- **GREEN**: Detection cluster points (points that triggered object detection)

## Simulation Features

- Multi-LiDAR setup (front bumper and roof-mounted)
- Configurable phantom point generation
- Realistic object detection
- Autonomous braking capabilities
- Real-time visualization of point clouds
- Detection of false-positive and false-negative results

## Simulation Configuration

Key configuration parameters in `CARLA_simulation/config/simulation_config.py` include:

```python
# Simulation parameters
self.sync_mode = True
self.delta_seconds = 0.05  # Simulation step size
self.max_simulation_time = 120  # Max simulation duration (seconds)

# Vehicle parameters
self.vehicle_type = 'vehicle.dodge.charger'
self.vehicle_mass = 1800.0  # kg
self.throttle_value = 0.4  # Vehicle throttle (0-1)

# LiDAR parameters
self.lidar_channels = 64  # Number of vertical channels
self.lidar_points_per_second = 500000
self.lidar_frequency = 20  # Hz
self.lidar_range = 100  # meters

# Interference simulation
self.simulate_interference = True  # Master toggle
self.interference_base_level = 0.05  # Base interference percentage
```

## Simulation Output

Simulation results are stored in the `../simulation_data/` directory, organized by simulation number. Each simulation directory contains:

- Point cloud data for each LiDAR sensor (stored as NumPy .npy files)
- Detection results (JSON format)
- Simulation summary (text file)
- Separate data for each LiDAR sensor (front and roof)
