#!/usr/bin/env python
"""Autonomous vehicle control systems"""

import numpy as np

class AutonomousController:
    """Handles autonomous vehicle control based on sensor data"""
    
    @staticmethod
    def brake_control(vehicle, detection_results, current_control, display_warning=True):
        """
        Implement autonomous braking based on detected objects
        
        Args:
            vehicle: CARLA vehicle actor
            detection_results: Results from point cloud processing
            current_control: Current vehicle control
            display_warning: Whether to display warning messages
            
        Returns:
            tuple: Modified vehicle control and braking status
        """
        # Start with the current control
        modified_control = current_control
        
        # Initialize braking status
        emergency_brake = False
        brake_intensity = 0.0
        brake_reason = ""
        
        # Check if object was detected
        if detection_results['object_detected']:
            distance = detection_results['distance']
            
            # Calculate time-to-collision (TTC) based on current speed
            vel = vehicle.get_velocity()
            speed_ms = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # Get Y-offset (lateral distance) for more accurate TTC calculation
            lateral_offset = abs(detection_results.get('location', [0, 0, 0])[1])
            
            # Print detection information for debugging
            print(f"OBJECT DETECTED at {distance:.2f}m, vehicle speed: {speed_ms:.2f}m/s")
            
            # Avoid division by zero
            if speed_ms < 0.1:
                ttc = float('inf')
            else:
                # For safety, calculate a more conservative TTC that accounts for the possibility
                # that objects with some lateral offset might still be in the path
                # Reduce effective distance for TTC calculation based on speed and lateral offset
                lateral_reduction_factor = max(0.5, min(1.0, (1.0 - lateral_offset/15.0)))
                effective_distance = distance * lateral_reduction_factor
                ttc = effective_distance / speed_ms
            
            # Braking thresholds - react to any object in the path
            # Start considering braking earlier at higher speeds - INCREASED range
            early_detection_distance = min(100.0, max(50.0, speed_ms * 6.0))  # Increased detection range
            
            if distance < early_detection_distance:
                # Special case for direct path objects - with more permissive threshold
                # If object has reasonable lateral offset, apply stronger braking earlier
                if lateral_offset < 3 and distance < 60.0:
                    direct_path_intensity = max(0.6, min(1.0, (60.0 - distance) / 40.0 + 0.2))
                    if direct_path_intensity > brake_intensity:
                        brake_intensity = direct_path_intensity
                        brake_reason = f"DIRECT PATH BRAKING - Object ahead at {distance:.1f}m (lateral offset: {lateral_offset:.1f}m)"
                        emergency_brake = True
                
                # Emergency braking for imminent collisions
                # Use dynamic TTC threshold based on speed
                ttc_threshold = min(6.0, max(3.0, 4.0 + (speed_ms - 10.0) * 0.2))
                
                if ttc < ttc_threshold and ttc != float('inf'):
                    # Calculate brake intensity based on TTC (harder braking for lower TTC)
                    if ttc < 1.5:
                        # Full emergency braking
                        brake_intensity = 1.0
                        brake_reason = f"EMERGENCY STOP - Collision imminent in {ttc:.1f}s"
                        emergency_brake = True
                    elif ttc < 2.0:
                        # Hard braking
                        brake_intensity = 0.95
                        brake_reason = f"HARD BRAKING - Collision risk in {ttc:.1f}s"
                        emergency_brake = True
                    elif ttc < 3.0:
                        # Medium braking
                        brake_intensity = 0.90
                        brake_reason = f"BRAKING - Object ahead in {ttc:.1f}s"
                        emergency_brake = True
                    else:
                        # Light braking - still treat as emergency for consistent behavior
                        brake_intensity = 0.80
                        brake_reason = f"PREEMPTIVE BRAKING - Object approaching in {ttc:.1f}s"
                        emergency_brake = True
                
                # Proximity-based braking - even more aggressive and with distance-based ramping
                elif distance < 55.0 and not emergency_brake:
                    # Reduce braking intensity for objects with high lateral offset
                    # But be more conservative - allow for more braking even with higher offsets
                    # Make lateral_factor more permissive to allow braking for points with higher offset
                    lateral_factor = max(0.7, 1.0 - (lateral_offset / 20.0))  # More permissive (0.7 minimum)
                    
                    # Only brake if object is somewhat in front (lateral_factor > 0)
                    if lateral_factor > 0:
                        # Calculate a more aggressive braking response based on speed
                        speed_factor = min(1.0, max(0.8, speed_ms / 10.0))
                        
                        # Distance-based ramping - stronger braking as we get closer
                        distance_factor = 1.0 - (distance / 55.0)  # Closer = stronger
                        
                        # Gentle braking based on proximity, reduced by lateral offset
                        base_intensity = (55.0 - distance) / 55.0 * 0.9
                        brake_intensity = max(brake_intensity, base_intensity * lateral_factor * speed_factor * (1.0 + distance_factor))
                        
                        brake_reason = f"SLOWING - Object nearby at {distance:.1f}m (offset: {lateral_offset:.1f}m)"
                    else:
                        # Object is too far to the side to matter at this distance
                        brake_reason = f"Object ignored - too far to the side: {lateral_offset:.1f}m"
                        brake_intensity = 0.0
        
        # Apply braking if needed
        if brake_intensity > 0:
            # Override throttle and apply brakes
            modified_control.throttle = 0.0
            modified_control.brake = brake_intensity
            
            # Display warning message if enabled
            if display_warning:
                print(f"\033[93m{brake_reason}\033[0m")  # Yellow text
                
                # Additional information for emergency situations
                if emergency_brake:
                    # Get current speed for context
                    vel = vehicle.get_velocity()
                    speed_ms = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
                    speed_mph = speed_ms * 2.23694  # Convert to mph
                    
                    print(f"\033[91mAUTONOMOUS BRAKING ACTIVATED at {speed_mph:.1f} mph\033[0m")  # Red text
        else:
            if detection_results['object_detected']:
                print(f"Object detected at {detection_results['distance']:.2f}m but no braking needed yet")
        
        # Return modified control and emergency status
        return modified_control, emergency_brake