#!/usr/bin/env python
"""Sensor callback functions for CARLA simulation"""

import numpy as np
from datetime import datetime
from sensors.lidar_processor import LidarProcessor
from control.autonomous_controller import AutonomousController

class SensorCallbacks:
    """Handles sensor callbacks and data processing"""
    
    # Static variables to store last readings from both sensors for interference simulation
    last_roof_lidar_data = None
    last_front_lidar_data = None
    
    @staticmethod
    def simulate_lidar_interference(point_cloud, other_sensor_data, interference_level=0.2):
        """
        Simulate interference between LiDAR sensors by injecting points
        
        Args:
            point_cloud: Primary sensor's point cloud data
            other_sensor_data: Data from the other sensor that may cause interference
            interference_level: Level of interference (0.0-1.0)
            
        Returns:
            numpy.ndarray: Point cloud with simulated interference
        """
        if other_sensor_data is None or len(other_sensor_data) == 0:
            return point_cloud
            
        # Create a copy of the original point cloud to avoid modifying the original
        modified_cloud = np.copy(point_cloud)
        
        # Select a percentage of points from the other sensor to inject as phantom readings
        if interference_level > 0:
            # Determine how many points to inject based on interference level
            num_points_to_inject = int(len(other_sensor_data) * interference_level * 0.2)  # Up to 20% of other sensor's points
            
            if num_points_to_inject > 0 and len(other_sensor_data) > 0:
                # Randomly select points from other sensor
                indices = np.random.choice(len(other_sensor_data), min(num_points_to_inject, len(other_sensor_data)), replace=False)
                phantom_points = other_sensor_data[indices]
                
                # Apply slight distortion to phantom points to simulate interference effects
                distortion = (np.random.random(phantom_points.shape) - 0.5) * 0.5  # ±0.25m distortion
                phantom_points = phantom_points + distortion
                
                # Add phantom points to the original point cloud
                modified_cloud = np.vstack([modified_cloud, phantom_points])
                
                print(f"LiDAR Interference: Added {num_points_to_inject} phantom points from roof LiDAR sensor")
                
        return modified_cloud
    
    @staticmethod
    def front_lidar_callback(point_cloud_data, lidar_data_list, vehicle=None, sim_start_time=None, enable_autonomous=True):
        """
        Process front LiDAR data with simulated interference, apply autonomous control, and store data
        
        Args:
            point_cloud_data: Raw LiDAR data from CARLA
            lidar_data_list: List to store processed data
            vehicle: CARLA vehicle actor (optional)
            sim_start_time: Timestamp when simulation started (optional)
            enable_autonomous: Whether to enable autonomous control
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
        
        # Store this reading for reference (not used for interference since we're only applying one-way interference)
        SensorCallbacks.last_front_lidar_data = data.copy() if len(data) > 0 else None
        
        # Apply interference from roof LiDAR (if available)
        if SensorCallbacks.last_roof_lidar_data is not None and len(data) > 0:
            # Control interference level (could be made dynamic or configurable)
            interference_level = 0.2  # 20% interference
            
            # Apply simulated interference
            data = SensorCallbacks.simulate_lidar_interference(
                data, 
                SensorCallbacks.last_roof_lidar_data,
                interference_level
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
    def roof_lidar_callback(point_cloud_data, lidar_data_list, vehicle=None, sim_start_time=None, enable_autonomous=True):
        """
        Process roof LiDAR data with NO simulated interference and store data
        
        Args:
            point_cloud_data: Raw LiDAR data from CARLA
            lidar_data_list: List to store processed data
            vehicle: CARLA vehicle actor (optional)
            sim_start_time: Timestamp when simulation started (optional)
            enable_autonomous: Whether to enable autonomous control
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