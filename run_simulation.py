#!/usr/bin/env python
"""
Main entry point for running CARLA simulation
This script configures and runs a pedestrian collision test scenario
"""

import sys
import traceback

# Import modules from the project structure
from config.simulation_config import SimulationConfig
from utils.file_manager import FileManager
from sensors.sensor_callbacks import SensorCallbacks
from simulation.carla_simulation import CarlaSimulation


def main():
    """Main function to run the simulation"""
    try:
        print("Starting CARLA Simulation")
        
        # Initialize configuration
        config = SimulationConfig()
        
        # Create simulation instance
        simulation = CarlaSimulation(config, FileManager, SensorCallbacks)
        
        # Run the simulation
        simulation.run_simulation()
        
        print("Simulation completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 0
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())