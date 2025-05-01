#!/usr/bin/env python
"""Sensor callback functions for CARLA simulation"""

import numpy as np
import time
from datetime import datetime
from sensors.lidar_processor import LidarProcessor
from control.autonomous_controller import AutonomousController

class SensorCallbacks:
    """Handles sensor callbacks and data processing"""
    
    # Static variables to store last readings from both sensors for interference simulation
    last_roof_lidar_data = None
    last_front_lidar_data = None
    
    # Keep track of focused cluster creation to avoid creating them too frequently
    last_focused_cluster_time = 0
    focused_cluster_cooldown = 5.0  # Seconds between focused clusters
    
    @staticmethod
    def create_focused_phantom_cluster(base_point, num_points=10, radius=0.5):
        """Create a tight cluster of phantom points around a base point"""
        cluster_points = []
        for _ in range(num_points):
            # Random offset within radius (only for XYZ coordinates)
            offset = np.random.normal(0, radius/2, 3)  # Tighter distribution for 3D coordinates
            
            # Create new point with same shape as base_point
            new_point = np.zeros_like(base_point)
            new_point[:3] = base_point[:3] + offset  # Apply offset only to XYZ coordinates
            
            # Copy intensity (and any other values beyond XYZ)
            if len(base_point) > 3:
                new_point[3:] = base_point[3:]
                
            cluster_points.append(new_point)
        return np.array(cluster_points)
    
    @staticmethod
    def simulate_lidar_interference(point_cloud, other_sensor_data, config):
        """
        Simulate interference between LiDAR sensors by injecting points
        Modified to place phantom points more strategically but less frequently
        """
        if other_sensor_data is None or len(other_sensor_data) == 0:
            # Add phantom flag column (0 = real point)
            phantom_flags = np.zeros((len(point_cloud), 1))
            return np.hstack([point_cloud, phantom_flags])
            
        # Create a copy of the original point cloud to avoid modifying the original
        modified_cloud = np.copy(point_cloud)
        
        # Add phantom flag column (0 = real point, 1 = phantom point)
        phantom_flags_original = np.zeros((len(modified_cloud), 1))
        
        # Combine original points with flags
        modified_cloud_with_flags = np.hstack([modified_cloud, phantom_flags_original])
        
        # Implement occasional skipping to reduce overall phantom frequency
        # Only generate phantoms 20% of the time, skip 80% of frames
        if np.random.random() > 0.2 and not hasattr(SensorCallbacks, 'force_phantom_generation'):
            return modified_cloud_with_flags
            
        # Reset the force flag if it was set
        if hasattr(SensorCallbacks, 'force_phantom_generation'):
            delattr(SensorCallbacks, 'force_phantom_generation')
        
        # Select a percentage of points from the other sensor to inject as phantom readings
        if config.simulate_interference:
            # Use parameters from config
            base_rate = config.interference_base_rate
            probability_factor = config.interference_probability_factor
            
            # Calculate number of points to potentially inject
            candidates = int(len(other_sensor_data) * base_rate)
            num_points_to_inject = int(candidates * probability_factor)
            
            # Add randomness - some frames may have more interference than others
            # Occasional "burst" of interference
            if np.random.random() < config.interference_burst_chance:
                num_points_to_inject = int(num_points_to_inject * config.interference_burst_multiplier)
                print("Interference Burst!")
            
            if num_points_to_inject > 0 and len(other_sensor_data) > 0:
                # Randomly select points from other sensor
                indices = np.random.choice(len(other_sensor_data), min(num_points_to_inject, len(other_sensor_data)), replace=False)
                phantom_points = other_sensor_data[indices]
                
                # Strategic phantom point placement in vehicle's path
                for i in range(len(phantom_points)):
                    # 90% chance to modify point to appear in vehicle's path
                    if np.random.random() < 0.90:
                        # Adjust y coordinate to be VERY close to the center line
                        phantom_points[i, 1] = phantom_points[i, 1] * 0.2 # Reduce lateral offset even more
                        
                        # Adjust height to be in detection range (0.3m to 2.0m)
                        phantom_points[i, 2] = np.random.uniform(0.3, 2.0)
                        
                        # Adjust distance to be in critical detection range (15-40 meters)
                        # This is a key range for emergency braking
                        distance = np.sqrt(phantom_points[i, 0]**2 + phantom_points[i, 1]**2)
                        if distance < 15 or distance > 40:
                            target_distance = np.random.uniform(20, 45)
                            scale_factor = target_distance / max(0.1, distance)
                            phantom_points[i, 0] *= scale_factor
                            phantom_points[i, 1] *= scale_factor
                
                # Create focused phantom clusters to ensure detection - but less often
                current_time = time.time()
                time_since_last_cluster = current_time - getattr(SensorCallbacks, 'last_focused_cluster_time', 0)
                
                if (num_points_to_inject >= 5 and 
                    np.random.random() < 0.15 and
                    time_since_last_cluster > SensorCallbacks.focused_cluster_cooldown):
                    
                    # Define an ideal base point (directly in front, good height, good distance)
                    ideal_distance = np.random.uniform(20, 35)  # Critical detection zone
                    base_point = np.array([ideal_distance, 0.0, 1.0, 1.0])  # x, y, z, intensity
                    
                    # Create a cluster of points around this base point
                    focused_cluster = SensorCallbacks.create_focused_phantom_cluster(
                        base_point, 
                        num_points=min(10, num_points_to_inject // 3),
                        radius=0.4  # Tight radius to ensure detection
                    )
                    
                    # Replace some phantom points with our focused cluster
                    if len(focused_cluster) > 0:
                        phantom_points[:len(focused_cluster)] = focused_cluster
                        print(f"Created focused phantom cluster with {len(focused_cluster)} points at {ideal_distance:.1f}m")
                        
                        # Update the last cluster time
                        SensorCallbacks.last_focused_cluster_time = current_time
                        
                        # Force phantom generation after a cooldown (to ensure we get some phantom points soon)
                        def schedule_phantom_generation():
                            SensorCallbacks.force_phantom_generation = True
                        
                        # Schedule phantom generation for next frame
                        schedule_phantom_generation()
                
                # More realistic distortion model
                # Points closer to sensor origin have less distortion
                distances = np.sqrt(np.sum(phantom_points[:, :3]**2, axis=1))
                distance_factor = np.minimum(distances / config.phantom_point_max_distance, 1.0)  # Normalize to 0-1 range, cap at 1
                
                # Apply stronger distortion
                distortion_scale = config.interference_distortion_base * 1.5 + (config.interference_distortion_range * distance_factor.reshape(-1, 1))
                distortion = (np.random.random(phantom_points.shape) - 0.5) * distortion_scale
                phantom_points = phantom_points + distortion
                
                # Concentration effect - phantom points tend to appear in clusters
                if num_points_to_inject >= 3 and np.random.random() < config.interference_cluster_chance:
                    # Create a few anchor points and cluster others around them
                    anchor_count = max(1, num_points_to_inject // 5)
                    anchor_indices = np.random.choice(num_points_to_inject, anchor_count, replace=False)
                    anchor_points = phantom_points[anchor_indices]
                    
                    # For non-anchor points, move them closer to a random anchor
                    for i in range(len(phantom_points)):
                        if i not in anchor_indices:
                            # Choose random anchor
                            anchor = anchor_points[np.random.randint(0, anchor_count)]
                            # Move closer to that anchor with configurable blend factor
                            blend_factor = config.interference_blend_factor_min + ((config.interference_blend_factor_max - config.interference_blend_factor_min) * np.random.random())
                            phantom_points[i, :3] = phantom_points[i, :3] * (1 - blend_factor) + anchor[:3] * blend_factor
                
                # Create phantom flags for the new points (1 = phantom point)
                phantom_flags_new = np.ones((len(phantom_points), 1))
                
                # Combine phantom points with flags
                phantom_points_with_flags = np.hstack([phantom_points, phantom_flags_new])
                
                # Add phantom points to the original point cloud
                result = np.vstack([modified_cloud_with_flags, phantom_points_with_flags])
                
                print(f"Added {num_points_to_inject} Phantom Points")
                return result
        
        # If no interference was added, return the original cloud with flags            
        return modified_cloud_with_flags
    
    @staticmethod
    def front_lidar_callback(point_cloud_data, lidar_data_list, vehicle=None, sim_start_time=None, enable_autonomous=True, config=None):
        """
        Process front LiDAR data with simulated interference, apply autonomous control, and store data
        
        Args:
            point_cloud_data: Raw LiDAR data from CARLA
            lidar_data_list: List to store processed data
            vehicle: CARLA vehicle actor (optional)
            sim_start_time: Timestamp when simulation started (optional)
            enable_autonomous: Whether to enable autonomous control
            config: SimulationConfig instance with interference parameters
        """
        # Convert data to numpy array with error handling for different CARLA versions
        try:
            # Try standard approach first
            data = np.copy(np.frombuffer(point_cloud_data.raw_data, dtype=np.dtype('f4')))
            point_count = len(data) // 4
            data = np.reshape(data, (point_count, 4))
        except Exception as e:
            print(f"Error processing front LiDAR data: {e}")
            
            # Alternative approach for CARLA 10.0
            try:
                data = np.frombuffer(point_cloud_data.raw_data, dtype=np.float32)
                # Try to detect the format based on data length
                if len(data) % 4 == 0:
                    data = np.reshape(data, (len(data)//4, 4))
                elif len(data) % 3 == 0:
                    data = np.reshape(data, (len(data)//3, 3))
                else:
                    print(f"Unknown LiDAR data format: {len(data)} points")
                    data = np.array([])  # Empty array as fallback
            except Exception as e2:
                print(f"Alternative processing also failed: {e2}")
                data = np.array([])  # Empty array as fallback
        
        # Store this reading for reference
        SensorCallbacks.last_front_lidar_data = data.copy() if len(data) > 0 else None
        
        # Apply interference from roof LiDAR (if available)
        if SensorCallbacks.last_roof_lidar_data is not None and len(data) > 0 and config is not None:
            # More realistic and variable interference
            base_interference = config.interference_base_level
            
            # Time-varying interference pattern
            if hasattr(SensorCallbacks, 'time_offset'):
                SensorCallbacks.time_offset += 1
            else:
                SensorCallbacks.time_offset = 0
            
            # Occasionally increase interference (simulating passing reflective surfaces, etc)
            time_factor = np.sin(SensorCallbacks.time_offset / 10.0) * config.interference_time_factor_amplitude
            
            # Ensure interference stays positive and doesn't exceed reasonable limits
            interference_level = max(0.01, min(0.15, base_interference + time_factor))
            
            # Apply simulated interference
            data = SensorCallbacks.simulate_lidar_interference(
                data, 
                SensorCallbacks.last_roof_lidar_data,
                config
            )
                
        # Print debug info about the point cloud periodically
        if len(lidar_data_list) % 20 == 0:
            print(f"Front LiDAR scan: {len(data)} points, shape: {data.shape}")
        
        # Current timestamp
        current_time = datetime.now()
        
        # Calculate time since simulation started (in milliseconds) if sim_start_time is provided
        time_since_start_ms = None
        if sim_start_time is not None:
            time_since_start_ms = int((current_time - sim_start_time).total_seconds() * 1000)
        
        # Process point cloud for object detection
        detection_results = {'object_detected': False, 'distance': float('inf')}
        
        # Only attempt detection and control if enabled and vehicle exists
        if enable_autonomous and vehicle is not None:
            # First, check if we have any points at all
            if len(data) > 0:
                # Process point cloud for detection
                detection_results = LidarProcessor.process_point_cloud(data)
                
                # Apply autonomous braking if needed
                if detection_results['object_detected']:
                    # Immediately get current control for modification
                    current_control = vehicle.get_control()
                    
                    # Apply autonomous braking
                    modified_control, _ = AutonomousController.brake_control(
                        vehicle, detection_results, current_control
                    )
                    
                    # Apply the modified control to the vehicle immediately
                    vehicle.apply_control(modified_control)
                    
                    # Print status message
                    print(f"Applied control: throttle={modified_control.throttle:.2f}, brake={modified_control.brake:.2f}")
            else:
                print("Warning: Front LiDAR returned empty point cloud")
        
        # Create data entry
        data_entry = {
            'timestamp': current_time,
            'time_since_start_ms': time_since_start_ms,
            'data': data.copy(),
            'frame': point_cloud_data.frame,
            'detection_results': detection_results,
            'sensor': 'front'
        }
        
        # Store the data entry
        lidar_data_list.append(data_entry)
    
    @staticmethod
    def roof_lidar_callback(point_cloud_data, lidar_data_list, vehicle=None, sim_start_time=None, enable_autonomous=True, config=None):
        """
        Process roof LiDAR data with NO simulated interference and store data
        
        Args:
            point_cloud_data: Raw LiDAR data from CARLA
            lidar_data_list: List to store processed data
            vehicle: CARLA vehicle actor (optional)
            sim_start_time: Timestamp when simulation started (optional)
            enable_autonomous: Whether to enable autonomous control
            config: SimulationConfig instance with interference parameters
        """
        # Convert data to numpy array with error handling
        try:
            # Try standard approach first
            data = np.copy(np.frombuffer(point_cloud_data.raw_data, dtype=np.dtype('f4')))
            point_count = len(data) // 4
            data = np.reshape(data, (point_count, 4))
        except Exception as e:
            print(f"Error processing roof LiDAR data: {e}")
            
            # Alternative approach
            try:
                data = np.frombuffer(point_cloud_data.raw_data, dtype=np.float32)
                # Try to detect the format based on data length
                if len(data) % 4 == 0:
                    data = np.reshape(data, (len(data)//4, 4))
                elif len(data) % 3 == 0:
                    data = np.reshape(data, (len(data)//3, 3))
                else:
                    print(f"Unknown LiDAR data format: {len(data)} points")
                    data = np.array([])  # Empty array as fallback
            except Exception as e2:
                print(f"Alternative processing also failed: {e2}")
                data = np.array([])  # Empty array as fallback
        
        # Store this reading for potential interference with front LiDAR
        SensorCallbacks.last_roof_lidar_data = data.copy() if len(data) > 0 else None
        
        # No interference applied to roof LiDAR - it remains clean
        # This is intentional as we only want the roof sensor to affect the front sensor
                
        # Print debug info about the point cloud periodically
        if len(lidar_data_list) % 20 == 0:
            print(f"Roof LiDAR scan: {len(data)} points, shape: {data.shape}")
        
        # Current timestamp
        current_time = datetime.now()
        
        # Calculate time since simulation started (in milliseconds) if sim_start_time is provided
        time_since_start_ms = None
        if sim_start_time is not None:
            time_since_start_ms = int((current_time - sim_start_time).total_seconds() * 1000)
        
        # Create data entry
        data_entry = {
            'timestamp': current_time,
            'time_since_start_ms': time_since_start_ms,
            'data': data.copy(),
            'frame': point_cloud_data.frame,
            'sensor': 'roof'
        }
        
        # Store the data entry
        lidar_data_list.append(data_entry)
    
    @staticmethod
    def create_collision_callback(vehicle, pedestrian):
        """
        Creates a collision callback function with closure over the vehicle and pedestrian
        
        Args:
            vehicle: CARLA vehicle actor
            pedestrian: CARLA pedestrian actor
            
        Returns:
            tuple: (callback_function, collision_detected_reference)
        """
        collision_detected = [False]  # Use a list to allow mutation from within the closure
        
        def collision_callback(event):
            # Calculate collision impact force
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            
            # Get information about what we collided with
            other_actor = event.other_actor
            
            # Check if collision was with the pedestrian
            if other_actor.id == pedestrian.id:
                collision_detected[0] = True
                
                # Get current car speed for impact reporting
                car_vel = vehicle.get_velocity()
                speed_ms = np.sqrt(car_vel.x**2 + car_vel.y**2 + car_vel.z**2)
                speed_mph = speed_ms * 2.23694  # Convert m/s to mph
                
                print("\n*** COLLISION DETECTED WITH PEDESTRIAN ***")
                print(f"Impact speed: {speed_mph:.2f} mph")
                print(f"Impact force: {intensity:.2f} N")
                print(f"Impact location: {event.transform.location}")
                
                return True  # Signal to exit the simulation
            else:
                # If we hit something other than the pedestrian, log it but don't end simulation
                print(f"Collision detected with actor ID: {other_actor.id}, impact: {intensity:.2f} N")
                return False
        
        return collision_callback, collision_detected