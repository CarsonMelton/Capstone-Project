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
    def save_lidar_data(sim_dir, lidar_data_dict, collision_detected=False, enable_autonomous=True, frame_count=0, stopped_time=0):
        """Save LiDAR data to the simulation directory"""
        # Check if we have any data to save
        if not lidar_data_dict or (len(lidar_data_dict.get('front', [])) == 0 and len(lidar_data_dict.get('roof', [])) == 0):
            print("No LiDAR data to save.")
            return
        
        # Create directories for each LiDAR type
        for lidar_type in ['front', 'roof']:
            if lidar_type in lidar_data_dict and lidar_data_dict[lidar_type]:
                lidar_dir = os.path.join(sim_dir, f"lidar_{lidar_type}")
                frames_dir = os.path.join(lidar_dir, "frames")
                
                # Create LiDAR and frames directories
                os.makedirs(lidar_dir, exist_ok=True)
                os.makedirs(frames_dir, exist_ok=True)
                
                lidar_data_list = lidar_data_dict[lidar_type]
                
                # Prepare metadata
                timestamps = [str(item['timestamp']) for item in lidar_data_list]
                frames = [int(item['frame']) for item in lidar_data_list]
                times_since_start_ms = [item.get('time_since_start_ms', 0) for item in lidar_data_list]
                
                # Save metadata for this LiDAR
                metadata_file = os.path.join(lidar_dir, "metadata.npz")
                np.savez_compressed(
                    metadata_file,
                    timestamps=timestamps,
                    frames=frames,
                    times_since_start_ms=times_since_start_ms,
                    lidar_type=lidar_type
                )
                
                # Extract detection results for front LiDAR
                if lidar_type == 'front':
                    detections = []
                    for item in lidar_data_list:
                        if 'detection_results' in item and item['detection_results'].get('object_detected', False):
                            detection = {
                                'frame': item['frame'],
                                'time_ms': item.get('time_since_start_ms', 0),
                                'distance': float(item['detection_results']['distance'])
                            }
                            detections.append(detection)
                    
                    # Save detection summary separately as JSON for easier analysis
                    if detections:
                        detection_file = os.path.join(lidar_dir, "object_detections.json")
                        with open(detection_file, 'w') as f:
                            json.dump(detections, f, indent=2)
                
                # Save individual frames
                print(f"Saving {len(lidar_data_list)} {lidar_type} LiDAR frames...")
                for item in lidar_data_list:
                    sim_time_ms = item.get('time_since_start_ms', 0)
                    
                    # Filename format: [lidar_type]_[milliseconds_since_start].npy
                    frame_filename = os.path.join(frames_dir, f"{lidar_type}_lidar_{sim_time_ms:08d}.npy")
                    
                    # Save this individual frame's data
                    np.save(frame_filename, item['data'])
                
                print(f"Saved {len(lidar_data_list)} {lidar_type} LiDAR frames")
        
        # Generate result summary file
        summary_file = os.path.join(sim_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Simulation Summary\n")
            f.write(f"=================\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Total front LiDAR scans: {len(lidar_data_dict.get('front', []))}\n")
            f.write(f"Total roof LiDAR scans: {len(lidar_data_dict.get('roof', []))}\n")
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
        
        print(f"Saved LiDAR data:")
        print(f"- Front LiDAR: {len(lidar_data_dict.get('front', []))} frames")
        print(f"- Roof LiDAR: {len(lidar_data_dict.get('roof', []))} frames")
        print(f"- Summary saved to: {summary_file}")