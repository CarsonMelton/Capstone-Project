#!/usr/bin/env python
"""LiDAR point cloud processing for object detection"""

import numpy as np
import time

class LidarProcessor:
    """Processes LiDAR point clouds for object detection"""
    
    # Class variables for tracking time
    startup_time = time.time()
    vehicle_moving_time = 0
    
    @staticmethod
    def process_point_cloud(point_cloud):
        """
        Process point cloud to find potential objects ahead in the vehicle's path
        Modified to track and return indices of points in detection cluster
        """
        # Extract XYZ data from points [x, y, z, intensity] or [x, y, z, intensity, phantom_flag]
        if point_cloud.shape[1] >= 4:
            xyz = point_cloud[:, :3]
        else:
            xyz = point_cloud
        
        # Get points in a wider cone in front of the vehicle (forward-facing points)
        # Filter points to focus on a broader area ahead
        forward_mask = xyz[:, 0] >= 0.0
        forward_points = xyz[forward_mask]
        forward_indices = np.where(forward_mask)[0]  # Track indices
        
        if len(forward_points) == 0:
            print("No forward points detected")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Set height filtering for objects (between 0.0m and 3.0m height)
        height_mask = (forward_points[:, 2] > 0.2) & (forward_points[:, 2] < 3.0)
        object_candidate_points = forward_points[height_mask]
        object_candidate_indices = forward_indices[height_mask]  # Track indices
        
        # Set width of detection corridor
        wide_corridor_mask = np.abs(object_candidate_points[:, 1]) < 5.0
        object_candidate_points = object_candidate_points[wide_corridor_mask]
        object_candidate_indices = object_candidate_indices[wide_corridor_mask]  # Track indices
        
        if len(object_candidate_points) == 0:
            print("No candidate points detected")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Calculate distance for each potential object point
        distances = np.sqrt(np.sum(object_candidate_points[:, :3]**2, axis=1))
        
        # Find closest potential object point
        if len(distances) == 0:
            return {'object_detected': False, 'distance': float('inf')}
            
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        closest_point = object_candidate_points[min_distance_idx]
        
        # Minimum distance check
        if min_distance < 4.0:  # Minimum 4 meters for valid detection
            vehicle_moving_time = getattr(LidarProcessor, 'vehicle_moving_time', 0)
            current_time = time.time()
            
            if not hasattr(LidarProcessor, 'startup_time'):
                LidarProcessor.startup_time = current_time
                LidarProcessor.vehicle_moving_time = 0
            
            # If less than 5 seconds since startup, require more points for close object detection
            if current_time - LidarProcessor.startup_time < 3.0:
                # Look for clusters of points
                cluster_radius = 0.5  # meters, reduced radius for more precision
                distances_to_closest = np.sqrt(np.sum((object_candidate_points - closest_point)**2, axis=1))
                cluster_mask = distances_to_closest < cluster_radius
                cluster_points = object_candidate_points[cluster_mask]
                cluster_indices = object_candidate_indices[cluster_mask]  # Track indices
                
                # Require more points (3) for early detections to filter out phantom points
                if len(cluster_points) < 3:
                    return {'object_detected': False, 'distance': float('inf')}
        
        # Check if point is within corridor
        lateral_limit = 4.0
        
        if abs(closest_point[1]) > lateral_limit:
            print(f"Point rejected due to Y-deviation: {abs(closest_point[1]):.2f}m > {lateral_limit}m")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Look for clusters of points with fewer required points
        cluster_radius = 0.8  
        distances_to_closest = np.sqrt(np.sum((object_candidate_points - closest_point)**2, axis=1))
        cluster_mask = distances_to_closest < cluster_radius
        cluster_points = object_candidate_points[cluster_mask]
        cluster_indices = object_candidate_indices[cluster_mask]  # Track indices
        
        # Require fewer points to detect an object
        if len(cluster_points) < 2:
            return {'object_detected': False, 'distance': float('inf')}
        
        # Return detection results with cluster indices
        return {
            'object_detected': True,
            'distance': min_distance,
            'location': closest_point,
            'point_count': len(cluster_points),
            'cluster_points': cluster_points,
            'cluster_indices': cluster_indices.tolist()  # Convert to list for easier serialization
        }