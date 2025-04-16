#!/usr/bin/env python
"""Configuration parameters for CARLA simulation"""

class SimulationConfig:
    """Configuration class for simulation parameters"""
    def __init__(self):
        # Simulation parameters
        self.sync_mode = True
        self.delta_seconds = 0.05
        self.no_rendering = False
        self.max_simulation_time = 120  # 2 minutes timeout

        # Weather parameters (0.0 to 100.0 scale)
        self.weather_enabled = True
        self.precipitation = 80.0       # Heavy rain
        self.precipitation_deposits = 80.0  # Puddles
        self.wind_intensity = 30.0
        self.fog_density = 85.0         # Heavy fog
        self.fog_distance = 10.0        # Low visibility distance
        self.fog_falloff = 1.0
        self.cloudiness = 90.0
        self.wetness = 80.0
        
        # Vehicle parameters
        self.vehicle_type = 'vehicle.dodge.charger'
        self.vehicle_mass = 1800.0  # kg
        self.vehicle_max_rpm = 5000.0
        self.throttle_value = 0.4
        
        # LiDAR parameters
        self.lidar_channels = 64
        self.lidar_points_per_second = 500000
        self.lidar_frequency = 20
        self.lidar_range = 100
        self.lidar_upper_fov = 15.0
        self.lidar_lower_fov = -25.0
        
        # Scene setup parameters (in meters, converted from feet)
        self.feet_to_meters = 0.3048
        self.pedestrian_distance = 100 * self.feet_to_meters
        self.car_right_offset = 5 * self.feet_to_meters
        self.car_setback = 300 * self.feet_to_meters
        self.pedestrian_distance_adjusted = 500 * self.feet_to_meters
        
        # Original position reference points
        self.original_x = 3.047784
        self.original_y = 130.210068 + 1.524
        
        # Features toggles
        self.enable_autonomous = True