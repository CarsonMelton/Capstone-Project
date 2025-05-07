#!/usr/bin/env python
"""Main simulation manager for CARLA testing"""

import carla
import time
import numpy as np
from datetime import datetime

class CarlaSimulation:
    """Main simulation manager class"""
    
    def __init__(self, config, file_manager, sensor_callbacks):
        """
        Initialize the CARLA simulation
        
        Args:
            config: SimulationConfig object
            file_manager: FileManager class
            sensor_callbacks: SensorCallbacks class
        """
        self.config = config
        self.file_manager = file_manager
        self.sensor_callbacks = sensor_callbacks
        
        self.sim_dir = file_manager.create_simulation_directory()
        self.front_lidar_data_list = []  
        self.roof_lidar_data_list = []  
        self.collision_detected = False
        self.stopped_time = 0.0
        self.vehicle = None
        self.pedestrian = None
        self.front_lidar = None
        self.roof_lidar = None
        self.collision_sensor = None
        self.world = None
        self.client = None
        self.original_settings = None
        self.sim_start_time = None
        self.exit_requested = False
        self.frame = 0
        self.movement_started = False
        
    def setup_carla_connection(self):
        """Connect to CARLA server and initialize world"""
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Save original settings to restore them later
        self.original_settings = self.world.get_settings()
        
        # Configure synchronous mode settings
        settings = self.world.get_settings()
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        settings.fixed_delta_seconds = self.config.delta_seconds
        settings.synchronous_mode = self.config.sync_mode
        settings.no_rendering_mode = self.config.no_rendering
        self.world.apply_settings(settings)
        
        return True
    
    def clear_existing_actors(self):
        """Clear all existing actors from the simulation"""
        actor_list = self.world.get_actors()
        
        # Group actors by type for systematic cleanup
        vehicles = [actor for actor in actor_list if 'vehicle' in actor.type_id]
        walkers = [actor for actor in actor_list if 'walker' in actor.type_id]
        sensors = [actor for actor in actor_list if 'sensor' in actor.type_id]
        controllers = [actor for actor in actor_list if 'controller' in actor.type_id]
        
        # First destroy controllers
        for controller in controllers:
            try:
                controller.stop()
                controller.destroy()
            except Exception as e:
                print(f"Error destroying controller: {e}")
        
        # Then destroy sensors
        for sensor in sensors:
            try:
                sensor.destroy()
            except Exception as e:
                print(f"Error destroying sensor: {e}")
        
        # Then vehicles and walkers
        for vehicle in vehicles:
            try:
                vehicle.destroy()
            except Exception as e:
                print(f"Error destroying vehicle: {e}")
                
        for walker in walkers:
            try:
                walker.destroy()
            except Exception as e:
                print(f"Error destroying walker: {e}")
        
        # Wait for cleanup to complete
        time.sleep(1.0)
        return True
    
    def setup_vehicle(self):
        """Set up the test vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Position the vehicle
        vehicle_bp = blueprint_library.find(self.config.vehicle_type)
        vehicle_spawn_point = carla.Transform(
            carla.Location(
                x=self.config.original_x - self.config.car_setback, 
                y=self.config.original_y + self.config.car_right_offset, 
                z=0.600000
            ), 
            carla.Rotation(yaw=0)
        )
        
        # Set vehicle physics properties for more realistic impact simulation
        physics_control = carla.VehiclePhysicsControl()
        physics_control.mass = self.config.vehicle_mass
        physics_control.max_rpm = self.config.vehicle_max_rpm
        physics_control.moi = 1.0
        physics_control.damping_rate_full_throttle = 0.25
        physics_control.damping_rate_zero_throttle_clutch_engaged = 2.0
        physics_control.damping_rate_zero_throttle_clutch_disengaged = 0.35
        
        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)
            # Apply physics control to the vehicle after spawning
            self.vehicle.apply_physics_control(physics_control)
            return True
        except RuntimeError as e:
            print(f"VEHICLE SPAWN ERROR: {str(e)}")
            return False
    
    def setup_pedestrian(self):
        """Set up the pedestrian in the scene"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Find a walker blueprint
        try:
            walker_bps = blueprint_library.filter('walker')
            if walker_bps:
                pedestrian_bp = walker_bps[0]
            else:
                raise RuntimeError("No walker blueprints found in the blueprint library")
        except Exception as e:
            print(f"Error finding walker blueprint: {e}")
            # Try a direct approach as last resort
            pedestrian_bp = blueprint_library.find('walker.pedestrian.0001')
            if not pedestrian_bp:
                # Just get any blueprint with 'walker' in the name
                available_bps = [bp.id for bp in blueprint_library if 'walker' in bp.id.lower()]
                if available_bps:
                    print(f"Available walker blueprints: {available_bps}")
                    pedestrian_bp = blueprint_library.find(available_bps[0])
                else:
                    raise RuntimeError("No walker blueprints found")
        
        # Configure collision and physics properties
        pedestrian_bp.set_attribute('is_invincible', 'false')
        
        # Calculate a position directly in the car's path
        pedestrian_spawn_point = carla.Transform(
            carla.Location(
                x=self.config.original_x - self.config.car_setback + self.config.pedestrian_distance_adjusted, 
                y=self.config.original_y + self.config.car_right_offset,  # Same Y as the car 
                z=1.2  # Higher off the ground for better visibility
            ),  
            carla.Rotation(yaw=180)  # Facing toward the car
        )
        
        try:
            self.pedestrian = self.world.spawn_actor(pedestrian_bp, pedestrian_spawn_point)
            
            # Ensure pedestrian has proper physics - critical for collision detection
            self.pedestrian.set_simulate_physics(True)
                
            # Set up proper collision detection for the pedestrian
            if hasattr(self.pedestrian, 'set_enable_gravity'):
                self.pedestrian.set_enable_gravity(True)
            
            # Give physics a moment to initialize properly
            self.world.tick()
            time.sleep(0.2)
            
            return True
                
        except RuntimeError as e:
            # If we got here, the vehicle spawn succeeded but pedestrian spawn failed
            if self.vehicle:
                self.vehicle.destroy()
            print(f"PEDESTRIAN SPAWN ERROR: {str(e)}")
            return False
    
    def setup_front_lidar(self):
        """Set up the front LiDAR sensor on the vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Configure LiDAR settings with improved object detection for CARLA 10.0
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        
        # Basic settings that should work in most CARLA versions
        lidar_bp.set_attribute('channels', str(self.config.lidar_channels))
        lidar_bp.set_attribute('points_per_second', str(self.config.lidar_points_per_second))
        lidar_bp.set_attribute('rotation_frequency', str(self.config.lidar_frequency))
        lidar_bp.set_attribute('range', str(self.config.lidar_range))
        
        # Set these if available in CARLA 10.0
        try:
            lidar_bp.set_attribute('upper_fov', str(self.config.lidar_upper_fov))
            lidar_bp.set_attribute('lower_fov', str(self.config.lidar_lower_fov))
        except Exception as e:
            print(f"Could not set front LiDAR FOV parameters: {e}")
        
        # Try to set additional parameters if available
        try:
            lidar_bp.set_attribute('sensor_tick', '0.0')  # Try to synchronize
        except:
            print("Could not set sensor_tick parameter")
        
        # Try to set noise parameters if available
        for param in ['dropoff_general_rate', 'dropoff_intensity_limit', 'dropoff_zero_intensity', 'noise_stddev']:
            try:
                lidar_bp.set_attribute(param, '0.0')
            except:
                print(f"Parameter {param} not available in this CARLA version")

        # Spawn LiDAR with proper transform - front mounted position
        lidar_transform = carla.Transform(
            carla.Location(x=2.0, z=0.7),  # Front of vehicle, bumper height
            carla.Rotation(pitch=0, yaw=0, roll=0)  # Forward facing
        )
        
        try:
            self.front_lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            
            # Set up LiDAR callback for data collection with autonomous control
            if self.config.enable_autonomous:
                # Pass the config to the callback
                self.front_lidar.listen(lambda data: self.sensor_callbacks.front_lidar_callback(
                    data, self.front_lidar_data_list, self.vehicle, self.sim_start_time, 
                    self.config.enable_autonomous, self.config))
            else:
                # Pass the config even when autonomous mode is disabled
                self.front_lidar.listen(lambda data: self.sensor_callbacks.front_lidar_callback(
                    data, self.front_lidar_data_list, None, self.sim_start_time, 
                    False, self.config))
            
            return True
        except Exception as e:
            print(f"Error setting up front LiDAR: {e}")
            return False
    
    def setup_roof_lidar(self):
        """Set up the roof LiDAR sensor on the vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Configure LiDAR settings for roof sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        
        # Basic settings similar to front LiDAR, but with wider FOV
        lidar_bp.set_attribute('channels', str(self.config.lidar_channels))
        lidar_bp.set_attribute('points_per_second', str(self.config.lidar_points_per_second))
        lidar_bp.set_attribute('rotation_frequency', str(self.config.lidar_frequency))
        lidar_bp.set_attribute('range', str(self.config.lidar_range))
        
        try:
            # Wider vertical FOV for roof LiDAR
            lidar_bp.set_attribute('upper_fov', str(self.config.lidar_upper_fov + 5.0))  # Increased upper FOV
            lidar_bp.set_attribute('lower_fov', str(self.config.lidar_lower_fov - 5.0))  # Increased lower FOV
        except Exception as e:
            print(f"Could not set roof LiDAR FOV parameters: {e}")
        
        # Try to set additional parameters if available
        try:
            lidar_bp.set_attribute('sensor_tick', '0.0')  # Try to synchronize
        except:
            print("Could not set sensor_tick parameter for roof LiDAR")
        
        # Try to set noise parameters if available
        for param in ['dropoff_general_rate', 'dropoff_intensity_limit', 'dropoff_zero_intensity', 'noise_stddev']:
            try:
                lidar_bp.set_attribute(param, '0.0')
            except:
                print(f"Parameter {param} not available in this CARLA version")

        # Spawn LiDAR with proper transform - roof mounted position
        lidar_transform = carla.Transform(
            carla.Location(x=0.0, z=2.0),  # Top center of vehicle for better overall visibility
            carla.Rotation(pitch=0, yaw=0, roll=0)  # Forward facing
        )
        
        try:
            self.roof_lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            
            # Set up LiDAR callback for data collection (without autonomous control for this sensor)
            # Pass the config to the callback
            self.roof_lidar.listen(lambda data: self.sensor_callbacks.roof_lidar_callback(
                data, self.roof_lidar_data_list, self.vehicle, self.sim_start_time, 
                False, self.config))
            
            return True
        except Exception as e:
            print(f"Error setting up roof LiDAR: {e}")
            return False
        
    def setup_collision_sensor(self):
        """Set up collision sensor on the vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Create collision sensor to detect when vehicle hits pedestrian
        collision_bp = blueprint_library.find('sensor.other.collision')
        
        try:
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            
            # Create and register collision callback
            collision_callback, collision_detected_ref = self.sensor_callbacks.create_collision_callback(
                self.vehicle, self.pedestrian)
            
            # Set up the callback
            self.collision_sensor.listen(collision_callback)
            
            # Store the collision detection reference
            self.collision_detected = collision_detected_ref[0]
            
            return True
        except Exception as e:
            print(f"Error setting up collision sensor: {e}")
            return False
    
    def initialize_simulation(self):
        """Initialize all components of the simulation"""
        
        # Setup CARLA connection
        if not self.setup_carla_connection():
            return False
        
        # Clear existing actors
        self.clear_existing_actors()
        
        # Setup vehicle
        if not self.setup_vehicle():
            return False
        
        # Setup pedestrian
        if not self.setup_pedestrian():
            return False
        
        # Setup front LiDAR (for detecting obstacles and autonomous braking)
        if not self.setup_front_lidar():
            return False
        
        # Setup roof LiDAR (for wider visibility and interference simulation)
        if not self.setup_roof_lidar():
            return False
        
        # Setup collision sensor
        if not self.setup_collision_sensor():
            return False
        
        # Record the simulation start time
        self.sim_start_time = datetime.now()
        
        return True
    
    def run_simulation(self):
        """Run the main simulation loop"""
        if not self.initialize_simulation():
            print("Failed to initialize simulation. Exiting.")
            return False
            
        # Set up vehicle control with reduced throttle for slower acceleration
        vehicle_control = carla.VehicleControl()
        vehicle_control.throttle = self.config.throttle_value
        vehicle_control.steer = 0.0
        vehicle_control.brake = 0.0
        vehicle_control.hand_brake = False
        vehicle_control.reverse = False
        
        # Apply initial control but don't start moving yet
        # Ensure vehicle is stopped initially
        stop_control = carla.VehicleControl()
        stop_control.throttle = 0.0
        stop_control.brake = 1.0  # Full brake
        stop_control.hand_brake = True
        self.vehicle.apply_control(stop_control)
        
        # Main simulation loop
        start_time = datetime.now()
        
        try:
            while not self.exit_requested:
                current_time = datetime.now()
                elapsed_seconds = (current_time - start_time).total_seconds()
                
                # Start vehicle movement after 1 second
                if not self.movement_started and elapsed_seconds > 1.0:
                    print("Starting Movement...")
                    # Make sure autopilot is off
                    self.vehicle.set_autopilot(False)
                    
                    # Make sure the hand brake is released and any initial braking is released
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                    self.world.tick()
                    
                    # Apply control with very aggressive throttle to overcome any residual braking
                    control = carla.VehicleControl()
                    control.throttle = 1.0  # Maximum throttle for guaranteed start
                    control.brake = 0.0
                    control.hand_brake = False
                    self.vehicle.apply_control(control)
                    
                    # Wait a bit longer for physics to initiate with the new strong throttle
                    time.sleep(0.2)
                    
                    # Wait a moment for physics to initiate
                    for _ in range(5):  # Multiple ticks to ensure physics are applied
                        self.world.tick()
                        time.sleep(0.01)
                    
                    # Now apply regular control
                    self.vehicle.apply_control(vehicle_control)
                    self.movement_started = True
                
                # Update the simulation
                self.world.tick()
                
                # This small sleep helps prevent simulation issues
                time.sleep(0.005)
                
                # Display distance between vehicle and pedestrian
                if self.frame % 20 == 0:  # Update stats every 20 frames
                    # Get current positions
                    car_pos = self.vehicle.get_location()
                    ped_pos = self.pedestrian.get_location()
                    
                    # Calculate distance
                    distance = np.sqrt((car_pos.x - ped_pos.x)**2 + (car_pos.y - ped_pos.y)**2)
                    
                    # Get vehicle speed
                    vel = self.vehicle.get_velocity()
                    speed_ms = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    speed_mph = speed_ms * 2.23694  # Convert to mph
                    
                    # Monitor for simulation timeout or completion
                    if elapsed_seconds > self.config.max_simulation_time:
                        print(f"Simulation timeout reached ({self.config.max_simulation_time} seconds). Ending simulation.")
                        self.exit_requested = True
                    
                    # Monitor for vehicle passing the pedestrian without collision
                    if car_pos.x > ped_pos.x + 10 and not self.collision_detected:
                        self.exit_requested = True
                
                # Increment frame counter
                self.frame += 1
                
        except KeyboardInterrupt:
            print("\nUser interrupted the simulation.")
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            # Cleanup and save data
            self.cleanup()
            
        return True
    
    def cleanup(self):
        """Clean up simulation resources and save data"""
        print("Cleaning up and saving data...")
        
        # Save the collected LiDAR data from both sensors
        self.file_manager.save_lidar_data(
            self.sim_dir, 
            self.front_lidar_data_list, 
            self.collision_detected, 
            self.config.enable_autonomous, 
            self.frame,
            self.stopped_time,
            sensor_name="front"
        )
        
        # Save the roof LiDAR data
        self.file_manager.save_lidar_data(
            self.sim_dir, 
            self.roof_lidar_data_list, 
            self.collision_detected, 
            False,  # Roof LiDAR doesn't control vehicle
            self.frame,
            self.stopped_time,
            sensor_name="roof"
        )
                
        print(f"Total frames: {self.frame}")
        print(f"Total front LiDAR scans: {len(self.front_lidar_data_list)}")
        print(f"Total roof LiDAR scans: {len(self.roof_lidar_data_list)}")
        
        # Clean up actors
        if hasattr(self, 'collision_sensor') and self.collision_sensor:
            self.collision_sensor.destroy()
        if hasattr(self, 'front_lidar') and self.front_lidar:
            self.front_lidar.destroy()
        if hasattr(self, 'roof_lidar') and self.roof_lidar:
            self.roof_lidar.destroy()
        if hasattr(self, 'pedestrian') and self.pedestrian:
            self.pedestrian.destroy()
        if hasattr(self, 'vehicle') and self.vehicle:
            self.vehicle.destroy()
        
        # Reset world settings
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)
        