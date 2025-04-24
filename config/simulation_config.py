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
        
        # Vehicle parameters - Adding these back
        self.vehicle_type = 'vehicle.dodge.charger'
        self.vehicle_mass = 1800.0  # kg
        self.vehicle_max_rpm = 5000.0
        self.throttle_value = 0.4
        
        # LiDAR parameters (base configuration for all LiDARs)
        self.lidar_channels = 64
        self.lidar_points_per_second = 500000
        self.lidar_frequency = 20
        self.lidar_range = 100
        self.lidar_upper_fov = 15.0
        self.lidar_lower_fov = -25.0
   
        # Multi-LiDAR configuration
        self.enable_multi_lidar = True  # Toggle for multiple LiDARs
        self.lidar_configs = [
            # Primary roof LiDAR (standard position)
            {
                'name': 'roof_lidar',
                'position': (0.0, 0.0, 2.0),  # x, y, z relative to vehicle
                'rotation': (0.0, 0.0, 0.0),  # pitch, yaw, roll
                'id_wavelength': 905,  # in nm - typical 905nm for standard LiDAR
            },
            # Front bumper LiDAR
            {
                'name': 'front_lidar',
                'position': (2.5, 0.0, 0.5),  # front bumper position
                'rotation': (0.0, 0.0, 0.0),
                'id_wavelength': 915,  # slightly different wavelength
            },
            # Rear LiDAR
            {
                'name': 'rear_lidar',
                'position': (-2.5, 0.0, 0.7),  # rear position
                'rotation': (0.0, 180.0, 0.0),  # facing backward
                'id_wavelength': 925,  # different wavelength
            }
        ]
   
        # Interference simulation - Master switches
        self.simulate_interference = True
        

        self.interference_base_level = 0.1  # Base percentage of interference applied
        self.interference_time_factor_amplitude = 0.03  # Controls the amplitude of time-varying oscillation
        
        self.interference_base_rate = 0.1  # Percentage of points considered as candidates
        self.interference_probability_factor = 0.2 # Fraction of candidate points that become phantom points
        self.interference_burst_chance = 0.1  # Probability of a sudden "burst" of increased interference
        self.interference_burst_multiplier = 4.0  # Multiplier for phantom points during a burst
        self.interference_distortion_base = 0.2  # Minimum distortion applied to phantom points
        self.interference_distortion_range = 0.6  # Additional distortion applied based on distance
        self.interference_cluster_chance = 0.3  # Probability of grouping phantom points into clusters
        self.interference_blend_factor_min = 0.3  # Minimum blending for clustering phantom points
        self.interference_blend_factor_max = 0.7  # Maximum blending for clustering phantom points
        
        self.direct_interference_probability = 0.01  # Controls probability of direct interference
        self.scattered_interference_probability = 0.002  # Controls probability of scattered interference
        self.phantom_point_max_distance = 20.0  # Maximum distance for phantom points to appear
        self.interference_max_range_error = 15.0  # Maximum range measurement error due to interference
       
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