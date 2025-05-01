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
            
            # Get vehicle speed
            vel = vehicle.get_velocity()
            speed_ms = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # Get lateral offset for more accurate TTC calculation
            lateral_offset = abs(detection_results.get('location', [0, 0, 0])[1])
            
            # Print detection information for debugging
            if display_warning:
                print(f"OBJECT DETECTED at {distance:.2f}m, vehicle speed: {speed_ms:.2f}m/s")
            
            # Calculate time-to-collision (TTC)
            ttc = AutonomousController._calculate_ttc(speed_ms, distance, lateral_offset)
            
            # Braking thresholds - react to any object in the path
            # Start considering braking earlier at higher speeds
            early_detection_distance = min(100.0, max(50.0, speed_ms * 6.0))
            
            if distance < early_detection_distance:
                # Direct path case - object directly ahead
                brake_data = AutonomousController._check_direct_path_braking(
                    lateral_offset, distance, brake_intensity)
                
                if brake_data['apply_brake']:
                    brake_intensity = brake_data['intensity']
                    brake_reason = brake_data['reason']
                    emergency_brake = True
                
                # Emergency braking based on TTC
                ttc_threshold = min(6.0, max(3.0, 4.0 + (speed_ms - 10.0) * 0.2))
                
                if ttc < ttc_threshold and ttc != float('inf'):
                    ttc_brake_data = AutonomousController._check_ttc_braking(ttc, brake_intensity)
                    
                    if ttc_brake_data['intensity'] > brake_intensity:
                        brake_intensity = ttc_brake_data['intensity']
                        brake_reason = ttc_brake_data['reason']
                        emergency_brake = True
                
                # Proximity-based braking
                elif distance < 55.0 and not emergency_brake:
                    prox_brake_data = AutonomousController._check_proximity_braking(
                        distance, lateral_offset, speed_ms, brake_intensity)
                    
                    if prox_brake_data['apply_brake']:
                        brake_intensity = prox_brake_data['intensity']
                        brake_reason = prox_brake_data['reason']
        
        # Apply braking if needed
        if brake_intensity > 0:
            # Override throttle and apply brakes
            modified_control.throttle = 0.0
            modified_control.brake = brake_intensity
            
            # Display warning message if enabled
            if display_warning:
                AutonomousController._display_brake_warning(
                    brake_reason, emergency_brake, vehicle)
        else:
            if detection_results['object_detected'] and display_warning:
                print(f"Object detected at {detection_results['distance']:.2f}m but no braking needed yet")
        
        # Return modified control and emergency status
        return modified_control, emergency_brake
    
    @staticmethod
    def _calculate_ttc(speed_ms, distance, lateral_offset):
        """Calculate time-to-collision based on speed and distance"""
        # Avoid division by zero
        if speed_ms < 0.1:
            return float('inf')
        
        # More conservative TTC calculation that accounts for lateral offset
        lateral_reduction_factor = max(0.5, min(1.0, (1.0 - lateral_offset/15.0)))
        effective_distance = distance * lateral_reduction_factor
        return effective_distance / speed_ms
    
    @staticmethod
    def _check_direct_path_braking(lateral_offset, distance, current_intensity):
        """Check for direct path braking conditions"""
        result = {
            'apply_brake': False,
            'intensity': 0.0,
            'reason': ""
        }
        
        # Special case for direct path objects - with more permissive threshold
        if lateral_offset < 3 and distance < 60.0:
            intensity = max(0.6, min(1.0, (60.0 - distance) / 40.0 + 0.2))
            if intensity > current_intensity:
                result['apply_brake'] = True
                result['intensity'] = intensity
                result['reason'] = f"DIRECT PATH BRAKING - Object ahead at {distance:.1f}m (lateral offset: {lateral_offset:.1f}m)"
        
        return result
    
    @staticmethod
    def _check_ttc_braking(ttc, current_intensity):
        """Check for TTC-based braking conditions"""
        result = {
            'intensity': 0.0,
            'reason': ""
        }
        
        # Calculate brake intensity based on TTC (harder braking for lower TTC)
        if ttc < 1.5:
            # Full emergency braking
            result['intensity'] = 1.0
            result['reason'] = f"EMERGENCY STOP - Collision imminent in {ttc:.1f}s"
        elif ttc < 2.0:
            # Hard braking
            result['intensity'] = 0.95
            result['reason'] = f"HARD BRAKING - Collision risk in {ttc:.1f}s"
        elif ttc < 3.0:
            # Medium braking
            result['intensity'] = 0.90
            result['reason'] = f"BRAKING - Object ahead in {ttc:.1f}s"
        else:
            # Light braking - still treat as emergency for consistent behavior
            result['intensity'] = 0.80
            result['reason'] = f"PREEMPTIVE BRAKING - Object approaching in {ttc:.1f}s"
        
        return result
    
    @staticmethod
    def _check_proximity_braking(distance, lateral_offset, speed_ms, current_intensity):
        """Check for proximity-based braking conditions"""
        result = {
            'apply_brake': False,
            'intensity': 0.0,
            'reason': ""
        }
        
        # Reduce braking intensity for objects with high lateral offset
        lateral_factor = max(0.7, 1.0 - (lateral_offset / 20.0))
        
        # Only brake if object is somewhat in front
        if lateral_factor > 0:
            # Calculate factors for braking intensity
            speed_factor = min(1.0, max(0.8, speed_ms / 10.0))
            distance_factor = 1.0 - (distance / 55.0)  # Closer = stronger
            
            # Gentle braking based on proximity, reduced by lateral offset
            base_intensity = (55.0 - distance) / 55.0 * 0.9
            intensity = max(current_intensity, 
                          base_intensity * lateral_factor * speed_factor * (1.0 + distance_factor))
            
            result['apply_brake'] = True
            result['intensity'] = intensity
            result['reason'] = f"SLOWING - Object nearby at {distance:.1f}m (offset: {lateral_offset:.1f}m)"
        else:
            # Object is too far to the side to matter at this distance
            result['reason'] = f"Object ignored - too far to the side: {lateral_offset:.1f}m"
        
        return result
    
    @staticmethod
    def _display_brake_warning(brake_reason, emergency_brake, vehicle):
        """Display warning message for braking"""
        print(f"\033[93m{brake_reason}\033[0m")  # Yellow text
        
        # Additional information for emergency situations
        if emergency_brake:
            # Get current speed for context
            vel = vehicle.get_velocity()
            speed_ms = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
            speed_mph = speed_ms * 2.23694  # Convert to mph
            
            print(f"\033[91mAUTONOMOUS BRAKING ACTIVATED at {speed_mph:.1f} mph\033[0m")  # Red text