#!/usr/bin/env python
"""
Point cloud processing utilities for LiDAR visualization
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, Optional, Union, Any

# Constants
COLOR_NORMAL = [0.0, 0.5, 1.0]  # Blue
COLOR_PHANTOM = [1.0, 0.0, 0.0]  # Red
COLOR_CLUSTER = [0.0, 1.0, 0.0]  # Green
COLOR_HIGHLIGHT = [0.0, 1.0, 0.4]  # Bright green


def preprocess_points(
    points: np.ndarray, 
    filter_behind: bool = True, 
    original_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Preprocess the points for visualization
    
    Args:
        points: Array containing point cloud data
        filter_behind: Whether to filter out points behind the car
        original_indices: Whether to return the mapping of original indices
        
    Returns:
        Processed array and optionally index mapping
    """
    if points is None:
        return None if not original_indices else (None, None)
        
    # If 1D array, try to reshape it based on size
    if len(points.shape) == 1:
        if points.size % 3 == 0:
            points_reshaped = points.reshape(-1, 3)
        elif points.size % 4 == 0:
            points_reshaped = points.reshape(-1, 4)
        elif points.size % 5 == 0:  # Support for 5-column format with phantom flag
            points_reshaped = points.reshape(-1, 5)
        else:
            print(f"Warning: Cannot reshape array of size {points.size} into points")
            return None if not original_indices else (None, None)
    else:
        # Make a copy to avoid modifying the original data
        points_reshaped = points.copy()
    
    # Ensure we have at least 3 dimensions (x, y, z)
    if points_reshaped.shape[1] < 3:
        print(f"Warning: Point cloud needs at least 3 dimensions (x,y,z), but has {points_reshaped.shape[1]}")
        return None if not original_indices else (None, None)
    
    # Flip the Y coordinates to correct left/right orientation
    # LiDAR's coordinate system: Y+ is right, visualization wants Y+ as left
    points_reshaped[:, 1] = -points_reshaped[:, 1]
    
    # Filter out points behind the car if requested
    # In LiDAR coordinates, x is forward, so we keep points with x > 0
    if filter_behind and len(points_reshaped) > 0:
        forward_mask = points_reshaped[:, 0] > 0
        
        if original_indices:
            # Create a mapping from filtered indices to original indices
            original_idx = np.arange(len(points_reshaped))
            filtered_idx = original_idx[forward_mask]
            
            # Also keep the filtered points
            points_reshaped = points_reshaped[forward_mask]
            
            point_count = len(points_reshaped)
            print(f"Filtered to {point_count} points in front of the car")
            return points_reshaped, filtered_idx
        else:
            points_reshaped = points_reshaped[forward_mask]
            point_count = len(points_reshaped)
            print(f"Filtered to {point_count} points in front of the car")
    
    return points_reshaped if not original_indices else (points_reshaped, np.arange(len(points_reshaped)))


def map_original_indices_to_processed(
    original_indices: List[int], 
    index_mapping: np.ndarray
) -> List[int]:
    """
    Map indices from the original point cloud to indices in the processed point cloud
    
    Args:
        original_indices: Indices in the original point cloud
        index_mapping: Mapping from processed indices to original indices
        
    Returns:
        Indices in the processed point cloud
    """
    if original_indices is None or len(original_indices) == 0 or index_mapping is None:
        return []
    
    # Convert index_mapping to a dictionary for fast lookup
    processed_to_original = {orig_idx: proc_idx for proc_idx, orig_idx in enumerate(index_mapping)}
    
    # Map original indices to processed indices
    mapped_indices = []
    for orig_idx in original_indices:
        if orig_idx in processed_to_original:
            mapped_indices.append(processed_to_original[orig_idx])
    
    return mapped_indices


def find_points_by_coordinates(
    points: np.ndarray, 
    reference_points: np.ndarray, 
    tolerance: float = 1e-3
) -> List[int]:
    """
    Find indices of points in the point cloud that match the reference points
    
    Args:
        points: Point cloud data
        reference_points: Reference points to match
        tolerance: Tolerance for coordinate matching
        
    Returns:
        Indices of matching points
    """
    if points is None or reference_points is None or len(points) == 0 or len(reference_points) == 0:
        return []
    
    # Use KD-tree for efficient spatial lookup
    try:
        tree = cKDTree(points[:, :3])
        
        # Find the closest point to each reference point
        distances, indices = tree.query(reference_points[:, :3], k=1)
        
        # Only keep points that are within tolerance
        valid_indices = indices[distances < tolerance]
        
        return valid_indices.tolist()
    except Exception as e:
        print(f"Error finding points by coordinates: {e}")
        
        # Fallback to brute force method if KD-tree fails
        matching_indices = []
        for ref_point in reference_points:
            # Find points that match within tolerance
            matches = np.where(np.all(np.abs(points[:, :3] - ref_point[:3]) < tolerance, axis=1))[0]
            if len(matches) > 0:
                matching_indices.extend(matches)
        
        return list(set(matching_indices))  # Remove duplicates


# Removed the unused detect_phantom_points function


def create_open3d_point_cloud(
    points: np.ndarray, 
    phantom_indices: Optional[List[int]] = None, 
    cluster_indices: Optional[List[int]] = None
) -> Optional[o3d.geometry.PointCloud]:
    """
    Create an Open3D point cloud with color highlighting for phantom points and clusters
    
    Args:
        points: Array of shape (N, 3+) containing point cloud data
        phantom_indices: Indices of points that might be phantoms
        cluster_indices: Indices of points in the detection cluster
        
    Returns:
        Open3D point cloud object
    """
    if points is None or len(points) == 0:
        return None
        
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use only XYZ
    
    # Check if we have a phantom flag column (5th column)
    has_phantom_flags = points.shape[1] >= 5
    if has_phantom_flags:
        # Get phantom flags (5th column)
        phantom_flags = points[:, 4].astype(bool)
        actual_phantom_indices = np.where(phantom_flags)[0]
        
        # Use the actual phantom indices instead of detected ones
        if len(actual_phantom_indices) > 0:
            phantom_indices = actual_phantom_indices
            print(f"Using {len(phantom_indices)} tagged phantom points from simulation")
    
    # Use intensity as color if available, otherwise use base color
    if points.shape[1] >= 4:
        # Map intensity to a color gradient
        intensity = points[:, 3]
        
        # Normalize intensity to 0-1 range
        if np.max(intensity) > np.min(intensity):
            intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        else:
            intensity_normalized = np.zeros_like(intensity)
        
        # Apply intensity as a brightness modifier while keeping the base color
        colors = np.zeros((len(points), 3))
        for i in range(3):
            colors[:, i] = COLOR_NORMAL[i] * (0.3 + 0.7 * intensity_normalized)
    else:
        # Use constant color
        colors = np.tile(COLOR_NORMAL, (len(points), 1))
    
    # Highlight phantom points if provided
    if phantom_indices is not None and len(phantom_indices) > 0:
        # Ensure indices are within bounds
        valid_phantom_indices = [idx for idx in phantom_indices if idx < len(points)]
        if len(valid_phantom_indices) < len(phantom_indices):
            print(f"Warning: {len(phantom_indices) - len(valid_phantom_indices)} phantom indices were out of bounds")
        
        if valid_phantom_indices:
            colors[valid_phantom_indices] = COLOR_PHANTOM
    
    # Highlight cluster points if provided (detection cluster takes precedence over phantom points)
    if cluster_indices is not None and len(cluster_indices) > 0:
        # Ensure indices are within bounds
        valid_cluster_indices = [idx for idx in cluster_indices if idx < len(points)]
        if len(valid_cluster_indices) < len(cluster_indices):
            print(f"Warning: {len(cluster_indices) - len(valid_cluster_indices)} cluster indices were out of bounds")
            print(f"Point cloud size: {len(points)}, max index found: {max(cluster_indices) if cluster_indices else 0}")
        
        if valid_cluster_indices:
            colors[valid_cluster_indices] = COLOR_CLUSTER
            print(f"Highlighting {len(valid_cluster_indices)} points in detection cluster")
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def calculate_cluster_center(
    points: np.ndarray, 
    cluster_indices: List[int]
) -> Optional[np.ndarray]:
    """
    Calculate the center point of a cluster
    
    Args:
        points: Point cloud data
        cluster_indices: Indices of points in the cluster
        
    Returns:
        3D coordinates of cluster center
    """
    if not cluster_indices or len(cluster_indices) == 0:
        return None
    
    # Ensure all indices are within bounds
    valid_indices = [idx for idx in cluster_indices if idx < len(points)]
    if not valid_indices:
        return None
    
    cluster_points = points[valid_indices, :3]
    return np.mean(cluster_points, axis=0)


def apply_custom_view(vis: o3d.visualization.Visualizer) -> None:
    """
    Apply a custom view to the visualizer for better visualization
    
    Args:
        vis: Open3D visualizer object
    """
    view_control = vis.get_view_control()
    # Position behind and slightly above, car moving away from camera
    view_control.set_front([-0.8, 0, 0.6])   # Camera direction vector
    view_control.set_up([0, 0, 1])           # Up vector (Z is up)
    view_control.set_lookat([5, 0, 0])       # Look-at point (focus on area ahead)
    view_control.set_zoom(0.08)              # Zoom level