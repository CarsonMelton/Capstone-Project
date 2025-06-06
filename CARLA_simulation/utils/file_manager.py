#!/usr/bin/env python
"""
File and Directory Management for CARLA Simulation
Utilities for managing simulation data directories and saving simulation results.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


class FileManager:
    """
    Handles directory creation and data storage for simulations
    
    This class provides static methods for creating simulation directories
    and saving LiDAR data, detection results, and simulation summaries.
    """
    
    @staticmethod
    def create_simulation_directory(base_dir: str = "..\simulation_data") -> str:
        """
        Create a new directory for this simulation run
        
        Args:
            base_dir: Base directory for all simulation data
            
        Returns:
            Path to the created simulation directory
        """
        # Create base directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # Find the next available simulation number
        existing_dirs = [
            d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Simulation ")
        ]
        
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
        
        print(f"Created simulation directory: {sim_dir}")
        return sim_dir
    
    @staticmethod
    def save_lidar_data(
        sim_dir: str,
        lidar_data_list: List[Dict[str, Any]],
        collision_detected: bool = False,
        enable_autonomous: bool = True,
        frame_count: int = 0,
        stopped_time: float = 0,
        sensor_name: str = "front"
    ) -> None:
        """
        Save LiDAR data to the simulation directory
        
        Args:
            sim_dir: Directory to save data to
            lidar_data_list: List of LiDAR data entries
            collision_detected: Whether a collision was detected
            enable_autonomous: Whether autonomous mode was enabled
            frame_count: Total number of simulation frames
            stopped_time: Time vehicle was stopped
            sensor_name: Name of the sensor (front or roof)
        """
        if not lidar_data_list:
            print(f"No {sensor_name} LiDAR data to save.")
            return
        
        # Create sensor-specific subdirectory
        sensor_dir = os.path.join(sim_dir, f"{sensor_name}_lidar")
        os.makedirs(sensor_dir, exist_ok=True)
        
        # Create frames directory inside sensor directory
        frames_dir = os.path.join(sensor_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Create clusters directory for saving detection cluster indices
        clusters_dir = os.path.join(sensor_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)
        
        # Prepare metadata
        timestamps = [str(item['timestamp']) for item in lidar_data_list]
        frames = [int(item['frame']) for item in lidar_data_list]
        times_since_start_ms = [item.get('time_since_start_ms', 0) for item in lidar_data_list]
        
        # Extract detection results for metadata (only for front LiDAR)
        detections = []
        if sensor_name == "front":
            for item in lidar_data_list:
                if 'detection_results' in item and item['detection_results'].get('object_detected', False):
                    detection = {
                        'frame': item['frame'],
                        'time_ms': item.get('time_since_start_ms', 0),
                        'distance': float(item['detection_results']['distance'])
                    }
                    detections.append(detection)
        
        # Save metadata
        metadata_file = os.path.join(sensor_dir, "metadata.npz")
        np.savez_compressed(
            metadata_file,
            sensor=sensor_name,
            timestamps=timestamps,
            frames=frames,
            times_since_start_ms=times_since_start_ms
        )
        
        # Save detection summary as JSON for easier analysis (front LiDAR only)
        if sensor_name == "front" and detections:
            detection_file = os.path.join(sensor_dir, "object_detections.json")
            with open(detection_file, 'w') as f:
                json.dump(detections, f, indent=2)
        
        # Save individual frames
        print(f"Saving {len(lidar_data_list)} {sensor_name} LiDAR frames...")
        for item in lidar_data_list:
            sim_time_ms = item.get('time_since_start_ms', 0)
            
            # Filename format: [milliseconds_since_start].npy
            frame_filename = os.path.join(frames_dir, f"{sim_time_ms:08d}.npy")
            
            # Save this individual frame's data
            np.save(frame_filename, item['data'])
            
            # If this frame has detection results with cluster indices, save them separately
            if (sensor_name == "front" and 'detection_results' in item and 
                    item['detection_results'].get('object_detected', False)):
                if 'cluster_indices' in item['detection_results']:
                    # Save cluster indices to a separate JSON file
                    cluster_indices = item['detection_results']['cluster_indices']
                    cluster_filename = os.path.join(clusters_dir, f"{sim_time_ms:08d}.json")
                    
                    try:
                        with open(cluster_filename, 'w') as f:
                            json.dump({'cluster_indices': cluster_indices}, f)
                    except Exception as e:
                        print(f"Error saving cluster indices: {e}")
        
        # Generate result summary file (for the front sensor only to avoid duplication)
        if sensor_name == "front":
            FileManager._create_summary_file(
                sim_dir, 
                frame_count, 
                len(lidar_data_list),
                collision_detected,
                enable_autonomous,
                stopped_time
            )
        
        print(f"Saved {len(lidar_data_list)} {sensor_name} LiDAR frames:")
        print(f"- Metadata saved to: {metadata_file}")
        print(f"- Frame data saved to: {frames_dir}")
        if sensor_name == "front" and detections:
            print(f"- Detection data saved to: {detection_file}")
            print(f"- Cluster data saved to: {clusters_dir}")
    
    @staticmethod
    def _create_summary_file(
        sim_dir: str, 
        frame_count: int, 
        lidar_scan_count: int,
        collision_detected: bool,
        enable_autonomous: bool,
        stopped_time: float
    ) -> None:
        """
        Create a summary file for the simulation
        
        Args:
            sim_dir: Directory to save data to
            frame_count: Total number of simulation frames
            lidar_scan_count: Number of LiDAR scans
            collision_detected: Whether a collision was detected
            enable_autonomous: Whether autonomous mode was enabled
            stopped_time: Time vehicle was stopped
        """
        summary_file = os.path.join(sim_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Simulation Summary\n")
            f.write(f"=================\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Total front LiDAR scans: {lidar_scan_count}\n")
            
            # Check if roof data is available and add to summary
            roof_dir = os.path.join(sim_dir, "roof_lidar")
            if os.path.exists(roof_dir):
                roof_frames = os.path.join(roof_dir, "frames")
                if os.path.exists(roof_frames):
                    roof_scan_count = len([f for f in os.listdir(roof_frames) if f.endswith('.npy')])
                    f.write(f"Total roof LiDAR scans: {roof_scan_count}\n")
            
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
        
        print(f"- Summary saved to: {summary_file}")