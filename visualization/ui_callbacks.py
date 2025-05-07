#!/usr/bin/env python
"""
UI callback functions for the LiDAR visualization
"""

class UICallbacks:
    """Class containing UI callback methods for visualization"""
    
    @staticmethod
    def stop_playback(vis, visualizer_instance):
        """
        Stop playback and exit
        
        Args:
            vis: Open3D visualizer
            visualizer_instance: Instance of LiDARVisualizer
            
        Returns:
            bool: False to continue processing
        """
        visualizer_instance.running = False
        return False
    
    @staticmethod
    def toggle_pause(vis, visualizer_instance):
        """
        Toggle playback pause state
        
        Args:
            vis: Open3D visualizer
            visualizer_instance: Instance of LiDARVisualizer
            
        Returns:
            bool: False to continue processing
        """
        visualizer_instance.paused = not visualizer_instance.paused
        if visualizer_instance.paused:
            print("Playback paused. Use LEFT/RIGHT to navigate frames manually.")
        else:
            print("Automatic playback started.")
        return False
    
    @staticmethod
    def next_frame(vis, visualizer_instance):
        """
        Move to the next frame
        
        Args:
            vis: Open3D visualizer
            visualizer_instance: Instance of LiDARVisualizer
            
        Returns:
            bool: False to continue processing
        """
        if not visualizer_instance.paused:
            visualizer_instance.paused = True
            print("Playback paused.")
        
        # Move to next frame
        visualizer_instance.current_idx = min(
            visualizer_instance.current_idx + 1, 
            len(visualizer_instance.lidar_files) - 1
        )
        visualizer_instance.load_frame(visualizer_instance.current_idx)
        return False
    
    @staticmethod
    def prev_frame(vis, visualizer_instance):
        """
        Move to the previous frame
        
        Args:
            vis: Open3D visualizer
            visualizer_instance: Instance of LiDARVisualizer
            
        Returns:
            bool: False to continue processing
        """
        if not visualizer_instance.paused:
            visualizer_instance.paused = True
            print("Playback paused.")
            
        # Move to previous frame
        visualizer_instance.current_idx = max(visualizer_instance.current_idx - 1, 0)
        visualizer_instance.load_frame(visualizer_instance.current_idx)
        return False
    
    @staticmethod
    def increase_speed(vis, visualizer_instance):
        """
        Increase playback speed
        
        Args:
            vis: Open3D visualizer
            visualizer_instance: Instance of LiDARVisualizer
            
        Returns:
            bool: False to continue processing
        """
        visualizer_instance.fps = min(visualizer_instance.fps * 1.25, 60)  # Cap at 60fps
        visualizer_instance.wait_time = 1.0 / visualizer_instance.fps
        print(f"Speed increased. FPS: {visualizer_instance.fps:.1f}")
        return False
    
    @staticmethod
    def decrease_speed(vis, visualizer_instance):
        """
        Decrease playback speed
        
        Args:
            vis: Open3D visualizer
            visualizer_instance: Instance of LiDARVisualizer
            
        Returns:
            bool: False to continue processing
        """
        visualizer_instance.fps = max(visualizer_instance.fps / 1.25, 1)  # Minimum 1fps
        visualizer_instance.wait_time = 1.0 / visualizer_instance.fps
        print(f"Speed decreased. FPS: {visualizer_instance.fps:.1f}")
        return False