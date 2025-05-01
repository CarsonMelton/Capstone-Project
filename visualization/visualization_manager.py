#!/usr/bin/env python
"""
LiDAR Visualization Manager
"""

import time
import open3d as o3d
import numpy as np
from typing import List, Dict, Optional, Union, Any, Callable, Tuple

from .file_utils import find_lidar_files, load_lidar_data, load_cluster_data, extract_ms_from_filename
from .point_cloud_processor import (
    preprocess_points, create_open3d_point_cloud, calculate_cluster_center, 
    map_original_indices_to_processed, find_points_by_coordinates, apply_custom_view
)
from .ui_callbacks import UICallbacks

# Constants for visualization
DEFAULT_FPS = 10
DEFAULT_POINT_SIZE = 2.0
DEFAULT_PHANTOM_THRESHOLD = 0.2  # Kept for backward compatibility but not used
COLOR_BACKGROUND = [0.1, 0.1, 0.1]  # Dark gray
COLOR_HIGHLIGHT = [0.0, 1.0, 0.4]  # Bright green

class LiDARVisualizer:
    """Class to handle LiDAR point cloud visualization"""
    
    def __init__(self, sim_dir: str):
        """
        Initialize the visualizer
        
        Args:
            sim_dir: Directory containing simulation data
        """
        self.sim_dir = sim_dir
        self.fps = DEFAULT_FPS
        self.point_size = DEFAULT_POINT_SIZE
        self.loop = False
        # We keep this as a class attribute for backwards compatibility, but it's not used
        self.phantom_threshold = DEFAULT_PHANTOM_THRESHOLD
        
        # State variables
        self.running = True
        self.paused = True
        self.current_idx = 0
        self.current_pcd = None
        self.current_highlight_sphere = None
        self.wait_time = 1.0 / self.fps
        
        # Find data files
        self.lidar_files, self.clusters_dir = find_lidar_files(sim_dir)
        
        # Initialize visualizer
        self.vis = None
        
    def setup_visualizer(self) -> bool:
        """
        Set up the Open3D visualizer window and controls
        
        Returns:
            bool: True if setup successful, False otherwise
        """
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
        
        # Register keyboard callbacks - we'll use the wrapper pattern to pass self to the callbacks
        self.vis.register_key_callback(ord("Q"), lambda vis: UICallbacks.stop_playback(vis, self))
        self.vis.register_key_callback(ord(" "), lambda vis: UICallbacks.toggle_pause(vis, self))
        self.vis.register_key_callback(262, lambda vis: UICallbacks.next_frame(vis, self))  # Right arrow
        self.vis.register_key_callback(263, lambda vis: UICallbacks.prev_frame(vis, self))  # Left arrow
        self.vis.register_key_callback(ord("+"), lambda vis: UICallbacks.increase_speed(vis, self))
        self.vis.register_key_callback(ord("-"), lambda vis: UICallbacks.decrease_speed(vis, self))
        
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
    
    def display_controls(self) -> None:
        """Display control instructions"""
        print("\nLiDAR Visualization Controls:")
        print("  SPACE: Pause/Resume playback")
        print("  LEFT/RIGHT: Previous/Next frame")
        print("  +/-: Increase/Decrease playback speed")
        print("  Q: Quit visualization")
        print("\nColor coding:")
        print("  BLUE = Normal LiDAR points (intensity modulated)")
        print("  RED = Phantom points (explicitly tagged phantom points)")
        print("  GREEN = Detection cluster points")
        print("\nStarted in manual mode. Use LEFT/RIGHT arrows to navigate frames.")
        print(f"Press SPACE to start automatic playback.\n")
    
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