#!/usr/bin/env python
"""
Main entry point for LiDAR visualization
"""

import sys
import os
import argparse

# Add the parent directory to path to enable importing the visualization package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from visualization package
from visualization import LiDARVisualizer, find_simulation_directory


def main():
    """Main entry point for the visualization tool"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='LiDAR Visualization with Tagged Phantom Points and Detection Clusters'
    )
    parser.add_argument('--dir', type=str, help='Directory containing simulation data')
    
    args = parser.parse_args()
    
    # Print the parsed directory for debugging
    print(f"Directory parameter: {args.dir}")
    
    try:
        print("Starting LiDAR Visualization Tool")
        
        # Find simulation directory - either specified or most recent
        sim_dir = find_simulation_directory(args.dir)  # Pass args.dir directly
        # Create and run visualizer with default settings
        visualizer = LiDARVisualizer(sim_dir=sim_dir)
        
        # Run the visualization
        visualizer.run()
        
        print("Visualization completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
        return 0
    except Exception as e:
        print(f"Visualization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())