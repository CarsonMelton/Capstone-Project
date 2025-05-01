#!/usr/bin/env python
"""
Main entry point for running CARLA simulation
"""

import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from CARLA-simulation package
from CARLA_simulation.config.simulation_config import SimulationConfig
from CARLA_simulation.utils.file_manager import FileManager
from CARLA_simulation.sensors.sensor_callbacks import SensorCallbacks
from CARLA_simulation.simulation.carla_simulation import CarlaSimulation

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
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())