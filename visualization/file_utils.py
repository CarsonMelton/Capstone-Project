#!/usr/bin/env python
"""File utility functions for LiDAR visualization"""

import os
import re
import glob
import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

def extract_ms_from_filename(filename: str) -> int:
    """
    Extract milliseconds timestamp from a filename like '00000123.npy'
    
    Args:
        filename: Path to the file
        
    Returns:
        Timestamp in milliseconds
    """
    basename = os.path.basename(filename)
    
    # Look for the pattern ########.npy
    match = re.search(r'(\d+)\.npy$', basename)
    if match:
        return int(match.group(1))
    
    # Last resort, use file modification time
    try:
        return int(os.path.getmtime(filename) * 1000)
    except Exception:
        print(f"Warning: Could not extract timestamp from {basename}")
        return float('inf')


def load_lidar_data(file_path: str) -> Optional[np.ndarray]:
    """
    Load LiDAR data from .npy file
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        Array containing point cloud data or None if loading fails
    """
    try:
        points = np.load(file_path)
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_cluster_data(time_ms: int, clusters_dir: str) -> List[int]:
    """
    Load cluster indices for a specific frame
    
    Args:
        time_ms: Timestamp in milliseconds
        clusters_dir: Directory containing cluster data files
        
    Returns:
        List of indices of points in the detected object cluster
    """
    cluster_filename = os.path.join(clusters_dir, f"{time_ms:08d}.json")
    if os.path.exists(cluster_filename):
        try:
            with open(cluster_filename, 'r') as f:
                cluster_data = json.load(f)
            return cluster_data.get('cluster_indices', [])
        except Exception as e:
            print(f"Error loading cluster data: {e}")
    return []


def find_lidar_files(sim_dir: str) -> Tuple[List[str], Optional[str]]:
    """
    Find LiDAR frame files and related data in the simulation directory
    
    Args:
        sim_dir: Path to the simulation directory
        
    Returns:
        Tuple containing sorted list of LiDAR file paths and path to clusters directory
    """
    frames_dir = os.path.join(sim_dir, "frames")
    clusters_dir = None
    
    # If frames directory not found, look for it in subdirectories
    if not os.path.exists(frames_dir):
        # Look for potential LiDAR directories
        for item in os.listdir(sim_dir):
            item_path = os.path.join(sim_dir, item)
            if os.path.isdir(item_path) and ("lidar" in item.lower() or "sensor" in item.lower()):
                # Check if this directory has a frames subdirectory
                subdir_frames = os.path.join(item_path, "frames")
                if os.path.exists(subdir_frames):
                    frames_dir = subdir_frames
                    
                    # Look for clusters directory
                    subdir_clusters = os.path.join(item_path, "clusters")
                    if os.path.exists(subdir_clusters):
                        clusters_dir = subdir_clusters
                        print(f"Found clusters directory: {subdir_clusters}")
                    
                    print(f"Found frames directory in: {item}")
                    break
    else:
        # If frames directory exists directly, look for clusters in the parent directory
        parent_dir = os.path.dirname(frames_dir)
        potential_clusters_dir = os.path.join(parent_dir, "clusters")
        if os.path.exists(potential_clusters_dir):
            clusters_dir = potential_clusters_dir
            print(f"Found clusters directory: {potential_clusters_dir}")
    
    if not os.path.exists(frames_dir):
        print(f"No frames directory found in {sim_dir}")
        return [], None
    
    # Find all .npy files
    lidar_files = glob.glob(os.path.join(frames_dir, "*.npy"))
    
    if not lidar_files:
        print(f"No .npy files found in {frames_dir}")
        return [], None
    
    # Sort files by timestamp
    lidar_files.sort(key=extract_ms_from_filename)
    print(f"Found {len(lidar_files)} LiDAR frames")
    
    # Check if we have cluster data
    if clusters_dir and os.path.exists(clusters_dir):
        cluster_files = glob.glob(os.path.join(clusters_dir, "*.json"))
        print(f"Found {len(cluster_files)} cluster data files")
    
    return lidar_files, clusters_dir


def find_simulation_directory(specified_dir: Optional[str] = None) -> str:
    """
    Find the simulation directory to use in ../simulation_data
    
    Args:
        specified_dir: User-specified directory (if any)
        
    Returns:
        Path to the simulation directory
    """
    # Set the simulation data directory
    simulation_data_dir = "../simulation_data"
    
    # Check if simulation_data directory exists
    if not os.path.exists(simulation_data_dir) or not os.path.isdir(simulation_data_dir):
        print(f"Warning: {simulation_data_dir} directory not found. Using current directory.")
        return "."
    
    # If a specific directory was provided (and not None)
    if specified_dir is not None and specified_dir != "None":
        # Strip any quotes that might be in the specified directory
        specified_dir = specified_dir.strip('"\'')
        
        # Look for an exact match in the simulation_data directory
        potential_path = os.path.join(simulation_data_dir, specified_dir)
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            print(f"Using specified simulation: {specified_dir}")
            return potential_path
        
        # If no exact match, look for a partial match
        for item in os.listdir(simulation_data_dir):
            item_path = os.path.join(simulation_data_dir, item)
            if os.path.isdir(item_path) and specified_dir in item:
                print(f"Found matching simulation: {item}")
                return item_path
        
        print(f"Warning: Specified directory '{specified_dir}' not found in {simulation_data_dir}. Will use the most recent simulation.")
    
    # Find the most recent simulation
    sim_dirs = [d for d in os.listdir(simulation_data_dir) 
               if os.path.isdir(os.path.join(simulation_data_dir, d)) and d.startswith("Simulation ")]
    
    if not sim_dirs:
        print(f"No simulation directories found in {simulation_data_dir}. Using current directory.")
        return "."
    
    # Sort by simulation number to get the most recent one
    sim_dirs.sort(key=lambda x: int(x.split(" ")[1]) if len(x.split(" ")) > 1 and x.split(" ")[1].isdigit() else 0)
    latest_sim = sim_dirs[-1]
    latest_sim_path = os.path.join(simulation_data_dir, latest_sim)
    print(f"Using most recent simulation: {latest_sim}")
    return latest_sim_path