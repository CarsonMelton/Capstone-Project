#!/usr/bin/env python
"""
LiDAR Point Cloud Visualization
-------------------------------
Visualizes LiDAR point clouds with highlighting for:
- Phantom points (red) - ONLY explicitly tagged phantom points
- Object detection clusters (green)

Supports playback controls and visual analysis of autonomous driving data.
"""

import numpy as np
import argparse
import os
import open3d as o3d
import glob
import time
import re
import json
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, Optional, Union, Any

# Constants
DEFAULT_FPS = 10
DEFAULT_POINT_SIZE = 2.0
DEFAULT_PHANTOM_THRESHOLD = 0.2  # Kept for backward compatibility but not used
COLOR_NORMAL = [0.0, 0.5, 1.0]  # Blue
COLOR_PHANTOM = [1.0, 0.0, 0.0]  # Red
COLOR_CLUSTER = [0.0, 1.0, 0.0]  # Green
COLOR_HIGHLIGHT = [0.0, 1.0, 0.4]  # Bright green
COLOR_BACKGROUND = [0.1, 0.1, 0.1]  # Dark gray


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


# We keep this function for compatibility but we won't use it
def detect_phantom_points(
    points: np.ndarray, 
    threshold: float = DEFAULT_PHANTOM_THRESHOLD,
    max_points: int = 200
) -> List[int]:
    """
    Detect potential phantom points in the LiDAR data based on spatial isolation
    This function is kept for backward compatibility but is no longer used.
    
    Args:
        points: Points from LiDAR
        threshold: Distance threshold for considering a point as isolated
        max_points: Maximum number of phantom points to return
        
    Returns:
        Indices of potential phantom points
    """
    return []  # Return empty list as we don't want to use heuristics anymore


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


class LiDARVisualizer:
    """Class to handle LiDAR point cloud visualization"""
    
    def __init__(self, 
                 sim_dir: str, 
                 fps: float = DEFAULT_FPS, 
                 point_size: float = DEFAULT_POINT_SIZE, 
                 loop: bool = False, 
                 phantom_threshold: float = DEFAULT_PHANTOM_THRESHOLD):
        """
        Initialize the visualizer
        
        Args:
            sim_dir: Directory containing simulation data
            fps: Frames per second for playback
            point_size: Size of points in visualization
            loop: Whether to loop playback when finished
            phantom_threshold: Threshold for phantom point detection (not used anymore)
        """
        self.sim_dir = sim_dir
        self.fps = fps
        self.point_size = point_size
        self.loop = loop
        self.phantom_threshold = phantom_threshold  # Kept for backward compatibility but not used
        
        # State variables
        self.running = True
        self.paused = True
        self.current_idx = 0
        self.current_pcd = None
        self.current_highlight_sphere = None
        self.wait_time = 1.0 / fps
        
        # Find data files
        self.lidar_files, self.clusters_dir = find_lidar_files(sim_dir)
        
        # Initialize visualizer
        self.vis = None
        
    def setup_visualizer(self) -> bool:
        """Set up the Open3D visualizer window and controls"""
        if not self.lidar_files:
            print(f"No LiDAR data found in {self.sim_dir}")
            return False
            
        # Create visualization window
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        window_title = "LiDAR Visualization with Phantom Points and Detection Clusters"
        self.vis.create_window(window_name=window_title, width=1280, height=960)
        
        # Set rendering options
        opt = self.vis.get_render_option()
        opt.point_size = self.point_size
        opt.background_color = np.array(COLOR_BACKGROUND)
        
        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.vis.add_geometry(coord_frame)
        
        # Register keyboard callbacks
        self.vis.register_key_callback(ord("Q"), self.stop_playback)        # Q to quit
        self.vis.register_key_callback(ord(" "), self.toggle_pause)         # Space to pause/play
        self.vis.register_key_callback(262, self.next_frame)                # Right arrow
        self.vis.register_key_callback(263, self.prev_frame)                # Left arrow
        self.vis.register_key_callback(ord("+"), self.increase_speed)       # + to speed up
        self.vis.register_key_callback(ord("-"), self.decrease_speed)       # - to slow down
        self.vis.register_key_callback(ord("T"), self.increase_threshold)   # T increase threshold - kept for compatibility
        self.vis.register_key_callback(ord("G"), self.decrease_threshold)   # G decrease threshold - kept for compatibility
        
        return True
    
    def load_frame(self, idx: int) -> bool:
        """
        Load a point cloud frame and update the visualization
        
        Args:
            idx: Index of the frame to load
            
        Returns:
            True if frame loaded successfully, False otherwise
        """
        if idx >= len(self.lidar_files):
            return False
            
        file_path = self.lidar_files[idx]
        
        # Clear previous geometries
        self._clear_geometries()
        
        # Load the point cloud
        points = load_lidar_data(file_path)
        
        # Process points and keep track of original indices
        try:
            points_processed, index_mapping = preprocess_points(
                points, filter_behind=True, original_indices=True
            )
        except Exception as e:
            print(f"Error preprocessing points: {e}")
            points_processed = preprocess_points(points, filter_behind=True)
            index_mapping = None
        
        if points_processed is None or len(points_processed) == 0:
            print(f"No valid points in frame {idx}")
            return False
        
        # Identify phantom points (ONLY using tagged phantoms)
        phantom_indices = self._identify_phantom_points(points_processed)
        
        # Load and map cluster indices
        cluster_indices = self._load_cluster_indices(file_path, points, points_processed, index_mapping)
        
        # Create and display point cloud
        success = self._create_and_display_point_cloud(
            points_processed, phantom_indices, cluster_indices, idx
        )
        
        return success
    
    def _clear_geometries(self) -> None:
        """Remove existing geometries from the visualizer"""
        if self.current_pcd is not None:
            self.vis.remove_geometry(self.current_pcd)
        
        if self.current_highlight_sphere is not None:
            self.vis.remove_geometry(self.current_highlight_sphere)
            self.current_highlight_sphere = None
    
    def _identify_phantom_points(self, points: np.ndarray) -> List[int]:
        """
        Identify phantom points in the point cloud - ONLY using tagged phantom flags
        
        Args:
            points: Processed point cloud
            
        Returns:
            List of indices of phantom points
        """
        # Check if we have phantom flags (5th column)
        if points.shape[1] >= 5:
            phantom_flags = points[:, 4].astype(bool)
            phantom_indices = np.where(phantom_flags)[0]
            if len(phantom_indices) > 0:
                print(f"Found {len(phantom_indices)} tagged phantom points in the data")
                return phantom_indices
            else:
                print("No tagged phantom points found in this frame")
                return []
        else:
            print("Data does not contain phantom point tags (requires 5 columns)")
            return []
    
    def _load_cluster_indices(
        self, 
        file_path: str, 
        original_points: np.ndarray, 
        processed_points: np.ndarray,
        index_mapping: Optional[np.ndarray]
    ) -> List[int]:
        """
        Load cluster indices for the current frame
        
        Args:
            file_path: Path to the point cloud file
            original_points: Original point cloud data
            processed_points: Processed point cloud data
            index_mapping: Mapping between original and processed indices
            
        Returns:
            List of indices of cluster points in the processed point cloud
        """
        if not self.clusters_dir:
            return []
            
        # Extract timestamp from filename
        time_ms = extract_ms_from_filename(file_path)
        
        # Load original cluster indices
        original_cluster_indices = load_cluster_data(time_ms, self.clusters_dir)
        
        if not original_cluster_indices:
            return []
            
        print(f"Loaded {len(original_cluster_indices)} original cluster indices from saved data")
        
        # Try index mapping first
        if index_mapping is not None:
            cluster_indices = map_original_indices_to_processed(original_cluster_indices, index_mapping)
            if cluster_indices:
                print(f"Mapped {len(cluster_indices)} indices to processed point cloud")
                return cluster_indices
        
        # If mapping failed, try coordinate-based matching
        if original_points is not None and len(original_points) > 0:
            print("Trying to find cluster points by coordinates...")
            
            # Get cluster points from original point cloud
            try:
                # Filter indices that are within bounds of original points
                valid_indices = [idx for idx in original_cluster_indices if idx < len(original_points)]
                if valid_indices:
                    cluster_points = original_points[valid_indices]
                    
                    # Find matching points in processed point cloud
                    cluster_indices = find_points_by_coordinates(processed_points, cluster_points)
                    print(f"Found {len(cluster_indices)} matching points by coordinates")
                    return cluster_indices
            except Exception as e:
                print(f"Error in coordinate-based matching: {e}")
        
        return []
    
    def _create_and_display_point_cloud(
        self, 
        points: np.ndarray, 
        phantom_indices: List[int], 
        cluster_indices: List[int],
        frame_idx: int
    ) -> bool:
        """
        Create and display the point cloud with highlights
        
        Args:
            points: Processed point cloud data
            phantom_indices: Indices of phantom points
            cluster_indices: Indices of cluster points
            frame_idx: Index of the current frame
            
        Returns:
            True if successful, False otherwise
        """
        # Create point cloud with phantom points and clusters highlighted
        pcd = create_open3d_point_cloud(points, phantom_indices, cluster_indices)
        
        if pcd is None:
            return False
            
        # Add the point cloud to the visualizer
        self.vis.add_geometry(pcd)
        self.current_pcd = pcd
        
        # Add highlight sphere for cluster if available
        if cluster_indices and len(cluster_indices) > 0:
            self._add_cluster_highlight(points, cluster_indices)
        
        # Print statistics
        print(f"Frame: {frame_idx+1}/{len(self.lidar_files)} - {len(points)} points")
        
        # Apply custom view settings
        apply_custom_view(self.vis)
        
        # Update the visualization
        self.vis.update_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        return True
    
    def _add_cluster_highlight(self, points: np.ndarray, cluster_indices: List[int]) -> None:
        """
        Add a highlight sphere at the cluster center
        
        Args:
            points: Processed point cloud data
            cluster_indices: Indices of cluster points
        """
        try:
            # Filter invalid indices
            valid_indices = [idx for idx in cluster_indices if idx < len(points)]
            if not valid_indices:
                return
                
            # Calculate cluster center
            cluster_center = calculate_cluster_center(points, valid_indices)
            if cluster_center is None:
                return
                
            # Create a sphere to highlight the cluster center
            cluster_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
            cluster_sphere.translate(cluster_center)
            cluster_sphere.paint_uniform_color(COLOR_HIGHLIGHT)
            
            # Make the sphere semi-transparent
            cluster_sphere.compute_vertex_normals()
            self.vis.add_geometry(cluster_sphere)
            self.current_highlight_sphere = cluster_sphere
            
            # Print distance to cluster
            distance = np.linalg.norm(cluster_center)
            print(f"Detection cluster at distance: {distance:.2f}m")
        except Exception as e:
            print(f"Error adding cluster highlight: {e}")
    
    def stop_playback(self, vis) -> bool:
        """Stop playback and exit"""
        self.running = False
        return False
    
    def toggle_pause(self, vis) -> bool:
        """Toggle playback pause state"""
        self.paused = not self.paused
        if self.paused:
            print("Playback paused. Use LEFT/RIGHT to navigate frames manually.")
        else:
            print("Automatic playback started.")
        return False
    
    def next_frame(self, vis) -> bool:
        """Move to the next frame"""
        if not self.paused:
            self.paused = True
            print("Playback paused.")
        
        # Move to next frame
        self.current_idx = min(self.current_idx + 1, len(self.lidar_files) - 1)
        self.load_frame(self.current_idx)
        return False
    
    def prev_frame(self, vis) -> bool:
        """Move to the previous frame"""
        if not self.paused:
            self.paused = True
            print("Playback paused.")
            
        # Move to previous frame
        self.current_idx = max(self.current_idx - 1, 0)
        self.load_frame(self.current_idx)
        return False
    
    def increase_speed(self, vis) -> bool:
        """Increase playback speed"""
        self.fps = min(self.fps * 1.25, 60)  # Cap at 60fps
        self.wait_time = 1.0 / self.fps
        print(f"Speed increased. FPS: {self.fps:.1f}")
        return False
    
    def decrease_speed(self, vis) -> bool:
        """Decrease playback speed"""
        self.fps = max(self.fps / 1.25, 1)  # Minimum 1fps
        self.wait_time = 1.0 / self.fps
        print(f"Speed decreased. FPS: {self.fps:.1f}")
        return False
    
    def increase_threshold(self, vis) -> bool:
        """Increase phantom detection threshold (kept for compatibility, no effect)"""
        print("Phantom point threshold no longer used - using only tagged phantom points")
        return False
    
    def decrease_threshold(self, vis) -> bool:
        """Decrease phantom detection threshold (kept for compatibility, no effect)"""
        print("Phantom point threshold no longer used - using only tagged phantom points")
        return False
    
    def display_controls(self) -> None:
        """Display control instructions"""
        print("\nLiDAR Visualization Controls:")
        print("  SPACE: Pause/Resume playback")
        print("  LEFT/RIGHT: Previous/Next frame")
        print("  +/-: Increase/Decrease playback speed")
        print("  Q: Quit slideshow")
        print("\nColor coding:")
        print("  BLUE = Normal LiDAR points (intensity modulated)")
        print("  RED = Phantom points (explicitly tagged phantom points only)")
        print("  GREEN = Detection cluster points (points that triggered object detection)")
        print("\nStarted in manual mode. Use LEFT/RIGHT arrows to navigate frames.")
        print(f"Press SPACE to start automatic playback at {self.fps} FPS.\n")
    
    def run(self) -> None:
        """Run the visualization slideshow"""
        if not self.setup_visualizer():
            return
            
        self.display_controls()
        
        # Load and display the first frame
        if len(self.lidar_files) > 0:
            self.load_frame(self.current_idx)
        
        # Main visualization loop
        try:
            while self.running:
                if self.current_idx >= len(self.lidar_files):
                    if self.loop:
                        self.current_idx = 0
                        print("\nRestarting slideshow from beginning...")
                    else:
                        print("\nReached end of slideshow.")
                        break
                
                # Automatic playback logic
                if not self.paused:
                    # Load the current frame
                    if self.load_frame(self.current_idx):
                        # Move to next frame
                        self.current_idx += 1
                    else:
                        print(f"Error loading frame {self.current_idx}, skipping")
                        self.current_idx += 1
                else:
                    # Just update the renderer when paused
                    self.vis.poll_events()
                    self.vis.update_renderer()
                
                # Control playback speed
                time.sleep(self.wait_time)
        
        except KeyboardInterrupt:
            print("\nPlayback interrupted.")
        finally:
            if self.vis:
                self.vis.destroy_window()
            print("Slideshow ended.")


def find_simulation_directory(specified_dir: Optional[str] = None) -> str:
    """
    Find the simulation directory to use
    
    Args:
        specified_dir: User-specified directory (if any)
        
    Returns:
        Path to the simulation directory
    """
    if specified_dir:
        return specified_dir
    
    # Look for a simulation directory structure
    possible_dirs = [".", "simulation_data", "../simulation_data", ".."]
    
    for base_dir in possible_dirs:
        if os.path.exists(base_dir):
            sim_dirs = [d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Simulation ")]
            if sim_dirs:
                # Sort by simulation number to get the most recent one
                sim_dirs.sort(key=lambda x: int(x.split(" ")[1]) if x.split(" ")[1].isdigit() else 0)
                sim_dir = os.path.join(base_dir, sim_dirs[-1])
                print(f"Using most recent simulation: {sim_dirs[-1]}")
                return sim_dir
    
    # If no simulation directory was found, use current directory
    print("No simulation directory found, using current directory")
    return "."


def main() -> None:
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='LiDAR Visualization with Tagged Phantom Points and Detection Clusters')
    parser.add_argument('--dir', type=str, help='Directory containing simulation data')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help=f'Frames per second for slideshow (default: {DEFAULT_FPS})')
    parser.add_argument('--point_size', type=float, default=DEFAULT_POINT_SIZE, help=f'Size of points in visualization (default: {DEFAULT_POINT_SIZE})')
    parser.add_argument('--loop', action='store_true', help='Loop the slideshow when finished')
    parser.add_argument('--threshold', type=float, default=DEFAULT_PHANTOM_THRESHOLD, help=f'Threshold for phantom point detection (not used anymore)')
    
    args = parser.parse_args()
    
    # Find simulation directory
    sim_dir = find_simulation_directory(args.dir)
    
    # Create and run visualizer
    visualizer = LiDARVisualizer(
        sim_dir=sim_dir,
        fps=args.fps,
        point_size=args.point_size,
        loop=args.loop,
        phantom_threshold=args.threshold
    )
    
    visualizer.run()


if __name__ == "__main__":
    main()