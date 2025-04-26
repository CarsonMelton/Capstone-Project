#!/usr/bin/env python
"""LiDAR point cloud processing for object detection"""

import numpy as np
import time

class LidarProcessor:
    """Processes LiDAR point clouds for object detection"""
    
    @staticmethod
    def process_point_cloud(point_cloud):
        """
        Process point cloud to find potential objects ahead in the vehicle's path
        Modified to be more sensitive to phantom points
        """
        # Extract XYZ data from points [x, y, z, intensity] or [x, y, z, intensity, phantom_flag]
        if point_cloud.shape[1] >= 4:
            xyz = point_cloud[:, :3]
        else:
            xyz = point_cloud
        
        # Get points in a wider cone in front of the vehicle (forward-facing points)
        # Filter points to focus on a broader area ahead
        forward_mask = xyz[:, 0] > -1.0  # More permissive - include some points to the sides/rear
        forward_points = xyz[forward_mask]
        
        if len(forward_points) == 0:
            print("No forward points detected")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Use more permissive height filtering for objects (between 0.0m and 3.0m height)
        height_mask = (forward_points[:, 2] > 0.2) & (forward_points[:, 2] < 3.0)  # Keeping original height range
        object_candidate_points = forward_points[height_mask]
        
        # Add additional filter: use a much wider corridor for detection
        # This helps include objects that are to the sides
        wide_corridor_mask = np.abs(object_candidate_points[:, 1]) < 5.0  # Increased from 2.0 to 5.0 to widen corridor
        object_candidate_points = object_candidate_points[wide_corridor_mask]
        
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
        
        # Minimum distance check - maintain original logic
        if min_distance < 3.0:  # Minimum 3 meters for valid detection
            vehicle_moving_time = getattr(LidarProcessor, 'vehicle_moving_time', 0)
            current_time = time.time()
            
            if not hasattr(LidarProcessor, 'startup_time'):
                LidarProcessor.startup_time = current_time
                LidarProcessor.vehicle_moving_time = 0
            
            # If less than 5 seconds since startup, require more points for close object detection
            if current_time - LidarProcessor.startup_time < 5.0:
                # Look for clusters of points (improved object detection)
                cluster_radius = 0.5  # meters, reduced radius for more precision
                distances_to_closest = np.sqrt(np.sum((object_candidate_points - closest_point)**2, axis=1))
                cluster_points = object_candidate_points[distances_to_closest < cluster_radius]
                
                # Require more points (5) for early detections to filter out phantom points
                if len(cluster_points) < 5:
                    return {'object_detected': False, 'distance': float('inf')}
        
        # Check if point is within a much wider corridor in front
        lateral_limit = 5.0  # Increased from 3.0 to 5.0
        
        if abs(closest_point[1]) > lateral_limit:
            print(f"Point rejected due to Y-deviation: {abs(closest_point[1]):.2f}m > {lateral_limit}m")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Look for clusters of points with fewer required points
        cluster_radius = 1.0 
        distances_to_closest = np.sqrt(np.sum((object_candidate_points - closest_point)**2, axis=1))
        cluster_points = object_candidate_points[distances_to_closest < cluster_radius]
        
        # Require fewer points to detect an object
        if len(cluster_points) < 2:  # Reduced from 3 to 2
            return {'object_detected': False, 'distance': float('inf')}
        
        # Return detection results
        return {
            'object_detected': True,
            'distance': min_distance,
            'location': closest_point,
            'point_count': len(cluster_points),
            'cluster_points': cluster_points
        }