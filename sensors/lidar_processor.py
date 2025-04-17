#!/usr/bin/env python
"""LiDAR point cloud processing for object detection"""

import numpy as np

class LidarProcessor:
    """Processes LiDAR point clouds for object detection"""
    
    @staticmethod
    def process_point_cloud(point_cloud):
        """
        Process point cloud to find potential objects ahead in the vehicle's path
        """
        # Extract XYZ data from points [x, y, z, intensity]
        if point_cloud.shape[1] >= 4:
            xyz = point_cloud[:, :3]
        else:
            xyz = point_cloud
        
        # Get points in a cone in front of the vehicle (forward-facing points)
        # LiDAR coordinates: x is forward, y is right, z is up
        # Filter points to focus on area ahead - use a very permissive forward angle
        forward_mask = xyz[:, 0] > 0  # Points with positive x (forward)
        forward_points = xyz[forward_mask]
        
        if len(forward_points) == 0:
            print("No forward points detected in LiDAR data")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Use more permissive height filtering for objects (between 0.0m and 3.0m height)
        height_mask = (forward_points[:, 2] > 0.2) & (forward_points[:, 2] < 3.0)  # Increased minimum height to 0.2m to reduce ground noise
        object_candidate_points = forward_points[height_mask]
        
        # Add additional filter: focus on objects in a narrower corridor for the first detection
        # This helps to filter out objects that are far to the sides during initialization
        narrow_corridor_mask = np.abs(object_candidate_points[:, 1]) < 2.0  # Reduced from 4.0 to 2.0 to narrow corridor
        object_candidate_points = object_candidate_points[narrow_corridor_mask]
        
        if len(object_candidate_points) == 0:
            print("No object-height points detected")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Calculate distance for each potential object point
        distances = np.sqrt(np.sum(object_candidate_points[:, :3]**2, axis=1))
        
        # Find closest potential object point
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        closest_point = object_candidate_points[min_distance_idx]
        
        # Minimum distance check - NEW
        # Ignore very close detections during early startup as they're likely phantom points
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
                    print(f"Only {len(cluster_points)} points found in close cluster during startup, need at least 5 for early detection")
                    return {'object_detected': False, 'distance': float('inf')}
        
        # Check if point is within a reasonable corridor in front
        lateral_limit = 3.0  # meters - reduced from 10.0 to be more precise
        
        if abs(closest_point[1]) > lateral_limit:
            print(f"Point rejected due to Y-deviation: {abs(closest_point[1]):.2f}m > {lateral_limit}m")
            return {'object_detected': False, 'distance': float('inf')}
        
        # Look for clusters of points (improved object detection)
        # Find other points close to the closest point
        cluster_radius = 1.0  # meters
        distances_to_closest = np.sqrt(np.sum((object_candidate_points - closest_point)**2, axis=1))
        cluster_points = object_candidate_points[distances_to_closest < cluster_radius]
        
        # Check if we have enough points to confidently say we've detected an object
        if len(cluster_points) < 2:  # Reduced requirement: only need 2 points instead of 3
            print(f"Only {len(cluster_points)} points found in cluster, need at least 2 for detection")
            return {'object_detected': False, 'distance': float('inf')}
        
        print(f"OBJECT DETECTED at {min_distance:.2f}m with {len(cluster_points)} points in cluster. Y-offset: {abs(closest_point[1]):.2f}m")
        
        # Return detection results
        return {
            'object_detected': True,
            'distance': min_distance,
            'location': closest_point,
            'point_count': len(cluster_points),
            'cluster_points': cluster_points  # Return the cluster for visualization
        }