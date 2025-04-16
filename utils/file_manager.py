#!/usr/bin/env python
"""File and directory management utilities for CARLA simulation"""

import os
import json
import numpy as np
from datetime import datetime

class FileManager:
    """Handles directory creation and data saving"""
    
    @staticmethod
    def create_simulation_directory():
        """Create a new directory for this simulation run"""
        # Create base directory for all simulation data
        base_dir = "..\lidar_simulations"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # Find the next available simulation number
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Simulation ")]
        
        # Extract numbers from existing directories
        existing_numbers = []
        for d in existing_dirs:
            try:
                num = int(d.split(" ")[1])
                existing_numbers.append(num)
            except (IndexError, ValueError):
                pass
        
        # Determine the next simulation number
        next_num = 1
        if existing_numbers:
            next_num = max(existing_numbers) + 1
        
        # Create new directory
        sim_dir = os.path.join(base_dir, f"Simulation {next_num}")
        os.makedirs(sim_dir)
        
        print(f"Created directory: {sim_dir}")
        return sim_dir
    
    @staticmethod
    def save_lidar_data(sim_dir, lidar_data_list, collision_detected=False, enable_autonomous=True, frame_count=0, stopped_time=0):
        """Save LiDAR data to the simulation directory"""
        if not lidar_data_list:
            print("No LiDAR data to save.")
            return
        
        frames_dir = os.path.join(sim_dir, "frames")
        
        # Create frames directory
        os.makedirs(frames_dir, exist_ok=True)
        
        # Prepare metadata (just for NPZ file, not for naming)
        timestamps = [str(item['timestamp']) for item in lidar_data_list]
        frames = [int(item['frame']) for item in lidar_data_list]
        times_since_start_ms = [item.get('time_since_start_ms', 0) for item in lidar_data_list]
        
        # Extract detection results for metadata
        detections = []
        for item in lidar_data_list:
            if 'detection_results' in item and item['detection_results'].get('object_detected', False):
                detection = {
                    'frame': item['frame'],
                    'time_ms': item.get('time_since_start_ms', 0),
                    'distance': float(item['detection_results']['distance'])
                }
                detections.append(detection)
        
        # Save metadata
        metadata_file = os.path.join(sim_dir, "metadata.npz")
        np.savez_compressed(
            metadata_file,
            timestamps=timestamps,
            frames=frames,
            times_since_start_ms=times_since_start_ms
        )
        
        # Save detection summary separately as JSON for easier analysis
        if detections:
            detection_file = os.path.join(sim_dir, "object_detections.json")
            with open(detection_file, 'w') as f:
                json.dump(detections, f, indent=2)
        
        # Save individual frames
        print(f"Saving {len(lidar_data_list)} LiDAR frames...")
        for item in lidar_data_list:
            sim_time_ms = item.get('time_since_start_ms', 0)
            
            # Filename format: [milliseconds_since_start].npy
            frame_filename = os.path.join(frames_dir, f"{sim_time_ms:08d}.npy")
            
            # Save this individual frame's data
            np.save(frame_filename, item['data'])
        
        # Generate result summary file
        summary_file = os.path.join(sim_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Simulation Summary\n")
            f.write(f"=================\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Total LiDAR scans: {len(lidar_data_list)}\n")
            f.write(f"Autonomous mode: {enable_autonomous}\n")
            f.write(f"Collision detected: {collision_detected}\n")
            
            if collision_detected:
                f.write(f"OUTCOME: Collision occurred with pedestrian\n")
            else:
                if stopped_time >= 1.0:
                    f.write(f"OUTCOME: Vehicle successfully stopped before pedestrian\n")
                    f.write(f"Autonomous braking system prevented collision\n")
                else:
                    f.write(f"OUTCOME: No collision detected with pedestrian\n")
        
        print(f"Saved {len(lidar_data_list)} LiDAR frames:")
        print(f"- Metadata saved to: {metadata_file}")
        print(f"- Frame data saved to: {frames_dir}")
        if detections:
            print(f"- Detection data saved to: {detection_file}")
        print(f"- Summary saved to: {summary_file}")