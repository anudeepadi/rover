import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time
import random
from enum import Enum
import matplotlib.colors as mcolors

class RoverState(Enum):
    STOPPED = 0
    MOVING = 1
    TURNING = 2
    EXCAVATING = 3
    CONSTRUCTING = 4

class LunaboticsSimulation:
    def __init__(self, arena_size=(10, 10), obstacle_count=15, fancy_graphics=True):
        """
        Enhanced NASA Lunabotics rover simulation environment
        
        Args:
            arena_size: Size of the arena in meters (width, height)
            obstacle_count: Number of obstacles to place in the arena
            fancy_graphics: Whether to use enhanced graphics
        """
        # Arena parameters
        self.arena_width, self.arena_height = arena_size
        self.grid_resolution = 0.05  # 5cm grid cells for more detailed arena
        self.grid_width = int(self.arena_width / self.grid_resolution)
        self.grid_height = int(self.arena_height / self.grid_resolution)
        
        # Create lunar surface (0 = free space, values 0-1 for terrain height/texture)
        self.terrain = np.zeros((self.grid_height, self.grid_width))
        
        # Rover parameters
        self.rover_position = np.array([1.0, 1.0])  # (x, y) in meters
        self.rover_heading = 0.0  # radians, 0 = positive x-axis
        self.rover_size = 0.5  # meters
        self.rover_state = RoverState.STOPPED
        self.rover_path = [self.rover_position.copy()]  # Track rover's path
        self.rover_speed = 0  # Current speed
        self.max_speed = 0.3  # Max speed
        self.acceleration = 0.05  # Acceleration rate
        self.deceleration = 0.1  # Deceleration rate
        self.turn_rate = np.pi/12  # Radians per step
        self.obstacle_detected = False
        self.collision_count = 0
        self.energy_consumption = 0
        self.dust_particles = []
        
        # Goals and obstacles
        self.obstacles = []  # List to store obstacle positions and sizes
        self.generate_terrain()
        self.place_obstacles(obstacle_count)
        
        # Sensor parameters
        self.sensor_range = 3.0  # meters
        self.sensor_fov = np.pi * 2/3  # radians (120 degrees)
        self.sensor_resolution = 15  # number of rays
        
        # Navigation parameters
        self.goal_position = np.array([8.0, 8.0])  # (x, y) in meters
        self.min_obstacle_distance = 0.6  # meters
        self.move_step = 0.1  # meters per step
        
        # Visualization settings
        self.fancy_graphics = fancy_graphics
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # Create color maps for terrain
        self.terrain_cmap = plt.cm.gray
        self.obstacle_cmap = plt.cm.binary
        
        # Setup dashboard
        self.setup_dashboard()
        
        # Previous action for better obstacle avoidance
        self.previous_actions = ['stop'] * 5  # Store last 5 actions
        self.stuck_counter = 0
        self.turn_direction = 1  # 1 for right, -1 for left
        
    def generate_terrain(self):
        """Generate realistic lunar terrain with craters and surface texture"""
        # Base terrain with random noise
        base = np.random.normal(0.7, 0.05, (self.grid_height, self.grid_width))
        
        # Add lunar craters
        num_craters = random.randint(5, 10)
        for _ in range(num_craters):
            x = random.randint(10, self.grid_width - 10)
            y = random.randint(10, self.grid_height - 10)
            radius = random.randint(10, 30)
            depth = random.uniform(0.2, 0.4)
            
            # Create crater
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    dist = np.sqrt(i**2 + j**2)
                    if dist <= radius:
                        # Calculate crater depth based on distance from center
                        crater_factor = 0
                        if dist < radius * 0.2:  # Crater center
                            crater_factor = depth
                        elif dist < radius * 0.8:  # Crater slope
                            crater_factor = depth * (1 - (dist - radius*0.2) / (radius*0.6))
                        else:  # Crater rim
                            crater_factor = depth * -0.2 * (1 - (dist - radius*0.8) / (radius*0.2))
                            
                        # Apply to terrain
                        ny, nx = y + i, x + j
                        if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                            base[ny, nx] -= crater_factor
        
        self.terrain = np.clip(base, 0.3, 1.0)  # Ensure terrain values are in [0.3, 1.0]
    
    def place_obstacles(self, count):
        """Place random boulders in the arena"""
        # Clear existing obstacles
        self.obstacles = []
        
        # Add random obstacles (boulders 30-40cm)
        for _ in range(count):
            # Random position (avoid edges)
            x = random.uniform(1.5, self.arena_width - 1.5)
            y = random.uniform(1.5, self.arena_height - 1.5)
            
            # Random size (30-40cm diameter as specified in NASA requirements)
            size = random.uniform(0.3, 0.4)
            
            # Add boulder
            self.obstacles.append({
                'position': np.array([x, y]),
                'size': size,
                'type': 'boulder'
            })
        
        # Define excavation and construction zones
        self.excavation_zone = {
            'x': (3, 5),
            'y': (3, 5)
        }
        
        self.construction_zone = {
            'x': (6, 8),
            'y': (6, 8)
        }
    
    def setup_dashboard(self):
        """Setup dashboard axes for displaying rover stats"""
        self.dashboard_height = 0.15  # Height of dashboard in figure coordinates
        
        # Main plot area
        self.ax.set_position([0.05, self.dashboard_height, 0.9, 0.85 - self.dashboard_height])
        
        # Create dashboard axis
        self.dashboard_ax = self.fig.add_axes([0.05, 0.02, 0.9, self.dashboard_height - 0.05])
        self.dashboard_ax.set_facecolor('black')
        self.dashboard_ax.axis('off')
        
        # Create text elements for dashboard
        self.dashboard_elements = {
            'title': self.dashboard_ax.text(
                0.5, 0.85, 'NASA LUNABOTICS ROVER SIMULATION', 
                ha='center', va='center', fontsize=14, color='white',
                fontweight='bold'
            ),
            'position': self.dashboard_ax.text(
                0.1, 0.55, 'POSITION: (0.00, 0.00)', 
                ha='left', va='center', fontsize=10, color='#00FF00'
            ),
            'heading': self.dashboard_ax.text(
                0.1, 0.25, 'HEADING: 0.0°', 
                ha='left', va='center', fontsize=10, color='#00FF00'
            ),
            'state': self.dashboard_ax.text(
                0.4, 0.55, 'STATE: STOPPED', 
                ha='left', va='center', fontsize=10, color='#00FF00'
            ),
            'speed': self.dashboard_ax.text(
                0.4, 0.25, 'SPEED: 0.00 m/s', 
                ha='left', va='center', fontsize=10, color='#00FF00'
            ),
            'energy': self.dashboard_ax.text(
                0.7, 0.55, 'ENERGY: 0.00 Wh', 
                ha='left', va='center', fontsize=10, color='#00FF00'
            ),
            'collisions': self.dashboard_ax.text(
                0.7, 0.25, 'COLLISIONS: 0', 
                ha='left', va='center', fontsize=10, color='#FFFF00'
            ),
        }
    
    def update_dashboard(self):
        """Update dashboard with current rover stats"""
        # Update dashboard text elements
        self.dashboard_elements['position'].set_text(
            f'POSITION: ({self.rover_position[0]:.2f}, {self.rover_position[1]:.2f})'
        )
        self.dashboard_elements['heading'].set_text(
            f'HEADING: {self.rover_heading * 180/np.pi:.1f}°'
        )
        self.dashboard_elements['state'].set_text(
            f'STATE: {self.rover_state.name}'
        )
        self.dashboard_elements['speed'].set_text(
            f'SPEED: {self.rover_speed:.2f} m/s'
        )
        self.dashboard_elements['energy'].set_text(
            f'ENERGY: {self.energy_consumption:.2f} Wh'
        )
        self.dashboard_elements['collisions'].set_text(
            f'COLLISIONS: {self.collision_count}'
        )
        
        # Change color based on state
        state_color = '#00FF00'  # Default green
        if self.rover_state == RoverState.STOPPED:
            state_color = '#FF0000'  # Red
        elif self.rover_state == RoverState.TURNING:
            state_color = '#FFFF00'  # Yellow
        self.dashboard_elements['state'].set_color(state_color)
    
    def sensor_scan(self):
        """
        Simulate sensors (LIDAR or camera-based)
        
        Returns:
            readings: List of (angle, distance) tuples
        """
        readings = []
        
        # Calculate sensor angles based on rover heading
        angles = np.linspace(
            self.rover_heading - self.sensor_fov/2,
            self.rover_heading + self.sensor_fov/2,
            self.sensor_resolution
        )
        
        for angle in angles:
            # Cast ray to detect obstacles
            distance = self._ray_cast(angle)
            readings.append((angle, distance))
        
        return readings
    
    def _ray_cast(self, angle):
        """
        Cast a ray from rover position at specified angle until it hits an obstacle
        
        Args:
            angle: Angle in radians
            
        Returns:
            distance: Distance to obstacle in meters
        """
        # Direction vector
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Start at rover position
        position = self.rover_position.copy()
        
        # Step size for ray casting
        step_size = 0.05
        
        # Cast ray
        distance = 0.0
        while distance < self.sensor_range:
            # Update position
            position += direction * step_size
            distance += step_size
            
            # Check if out of bounds
            if (position[0] < 0 or position[0] >= self.arena_width or
                position[1] < 0 or position[1] >= self.arena_height):
                return distance
            
            # Check if hit obstacle
            for obstacle in self.obstacles:
                obs_pos = obstacle['position']
                obs_size = obstacle['size']
                
                # Check distance to obstacle
                if np.linalg.norm(position - obs_pos) <= obs_size/2:
                    return distance
        
        # No obstacle found within range
        return self.sensor_range
        
    def move_rover(self, action):
        """
        Move the rover based on action with improved collision recovery
        
        Args:
            action: 'forward', 'left', 'right', 'stop', 'accelerate', 'decelerate'
        """
        # Update action history
        self.previous_actions.pop(0)
        self.previous_actions.append(action)
        
        # Force acceleration after multiple turns to break turning loops
        if action == 'forward' and all(act in ['left', 'right'] for act in self.previous_actions[-4:-1]):
            self.rover_speed = self.max_speed * 0.5  # Set to half max speed to push through
        
        # Acceleration/deceleration physics
        if action == 'forward' or action == 'accelerate':
            self.rover_speed = min(self.rover_speed + self.acceleration, self.max_speed)
            self.rover_state = RoverState.MOVING
        elif action == 'decelerate':
            self.rover_speed = max(self.rover_speed - self.deceleration, 0)
            if self.rover_speed < 0.01:
                self.rover_state = RoverState.STOPPED
            else:
                self.rover_state = RoverState.MOVING
        elif action == 'stop':
            self.rover_speed = 0
            self.rover_state = RoverState.STOPPED
        
        # Movement based on current speed
        if self.rover_speed > 0:
            # Calculate new position
            new_position = self.rover_position + np.array([
                np.cos(self.rover_heading), 
                np.sin(self.rover_heading)
            ]) * self.rover_speed
            
            # Check if new position is valid (no collision)
            collision = self._check_collision(new_position)
            if not collision:
                self.rover_position = new_position
                self.rover_path.append(self.rover_position.copy())
                
                # Simulate energy consumption
                self.energy_consumption += self.rover_speed * 0.05
                
                # Generate dust particles when moving
                if random.random() < 0.3:
                    self._generate_dust_particles()
            else:
                # Collision occurred - stop and back up slightly
                self.collision_count += 1
                self.rover_speed = 0
                self.rover_state = RoverState.STOPPED
                
                # Significant bounce back after collision
                self.rover_position -= np.array([
                    np.cos(self.rover_heading), 
                    np.sin(self.rover_heading)
                ]) * 0.3  # Increased bounce back distance
                
                # Add randomization to prevent getting stuck in same pattern
                self.rover_heading += random.uniform(-0.3, 0.3)
        
        # Handle turning
        if action == 'left':
            self.rover_heading += self.turn_rate
            self.rover_state = RoverState.TURNING
            self.energy_consumption += 0.01  # Energy for turning
        elif action == 'right':
            self.rover_heading -= self.turn_rate
            self.rover_state = RoverState.TURNING
            self.energy_consumption += 0.01  # Energy for turning
                
    def _check_collision(self, position):
        """
        Check if position would cause collision with obstacle
        
        Args:
            position: (x, y) position in meters
            
        Returns:
            collision: True if collision would occur
        """
        # Check boundary
        rover_radius = self.rover_size / 2
        if (position[0] - rover_radius < 0 or 
            position[0] + rover_radius > self.arena_width or
            position[1] - rover_radius < 0 or 
            position[1] + rover_radius > self.arena_height):
            return True
        
        # Check each obstacle
        for obstacle in self.obstacles:
            obstacle_pos = obstacle['position']
            obstacle_radius = obstacle['size'] / 2
            
            # Simple collision detection based on distance
            distance = np.linalg.norm(position - obstacle_pos)
            if distance < (rover_radius + obstacle_radius):
                return True
        
        return False
    
    def _generate_dust_particles(self):
        """Generate dust particles behind the rover"""
        if not self.fancy_graphics:
            return
            
        # Generate 1-3 particles
        num_particles = random.randint(1, 3)
        for _ in range(num_particles):
            # Position slightly behind the rover
            offset_distance = random.uniform(0.1, 0.3)
            offset_angle = self.rover_heading + np.pi + random.uniform(-0.5, 0.5)
            
            position = self.rover_position + np.array([
                np.cos(offset_angle), 
                np.sin(offset_angle)
            ]) * offset_distance
            
            # Particle properties
            self.dust_particles.append({
                'position': position,
                'velocity': np.array([
                    np.cos(offset_angle), 
                    np.sin(offset_angle)
                ]) * random.uniform(0.05, 0.15),
                'life': 1.0,  # Life from 1.0 to 0.0
                'decay': random.uniform(0.05, 0.1)
            })
    
    def _update_dust_particles(self):
        """Update dust particle positions and lifetimes"""
        if not self.fancy_graphics:
            return
            
        updated_particles = []
        for particle in self.dust_particles:
            # Update position
            particle['position'] += particle['velocity']
            
            # Decay life
            particle['life'] -= particle['decay']
            
            # Keep if still alive
            if particle['life'] > 0:
                updated_particles.append(particle)
        
        self.dust_particles = updated_particles
    
    def plan_path(self):
        """
        Enhanced obstacle avoidance algorithm with improved recovery
        
        Returns:
            action: 'forward', 'left', 'right', 'stop', etc.
        """
        # Get sensor readings
        readings = self.sensor_scan()
        
        # Process readings to find obstacles
        front_distances = []
        min_front_distance = float('inf')
        min_left_distance = float('inf')
        min_right_distance = float('inf')
        
        for angle, distance in readings:
            # Calculate angle relative to heading
            relative_angle = (angle - self.rover_heading + 2*np.pi) % (2*np.pi)
            
            # Check obstacles in front (within 30 degrees of heading)
            if relative_angle < np.pi/6 or relative_angle > 2*np.pi - np.pi/6:
                front_distances.append(distance)
                if distance < min_front_distance:
                    min_front_distance = distance
            
            # Check obstacles to the left
            elif np.pi/6 <= relative_angle < np.pi/2:
                if distance < min_left_distance:
                    min_left_distance = distance
            
            # Check obstacles to the right
            elif 2*np.pi - np.pi/2 < relative_angle <= 2*np.pi - np.pi/6:
                if distance < min_right_distance:
                    min_right_distance = distance
        
        # Calculate average front distance if we have readings
        avg_front_distance = 0
        if front_distances:
            avg_front_distance = sum(front_distances) / len(front_distances)
        
        # Determine if obstacle is detected
        self.obstacle_detected = min_front_distance < self.min_obstacle_distance
        
        # Check if we're stuck in a turning pattern
        turning_pattern = all(action in ['left', 'right'] for action in self.previous_actions[-3:])
        
        # Analyze if we're making progress - are we moving?
        if self.rover_speed < 0.01 and turning_pattern:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # Force a forward movement if we've been turning too long
        if self.stuck_counter > 8:
            self.stuck_counter = 0
            return 'forward'  # Force forward movement to break the pattern
        
        # Path planning logic
        if self.obstacle_detected:
            # Obstacle in front, need to turn
            if min_left_distance > min_right_distance * 1.2:  # Left has 20% more clearance
                return 'left'
            elif min_right_distance > min_left_distance * 1.2:  # Right has 20% more clearance
                return 'right'
            else:
                # If distances are similar or unclear, change direction
                self.turn_direction *= -1
                return 'left' if self.turn_direction > 0 else 'right'
                
        elif turning_pattern and not self.obstacle_detected:
            # We've been turning but path is clear now, move forward
            return 'forward'
        else:
            # No immediate obstacle, head toward goal
            to_goal = self.goal_position - self.rover_position
            goal_distance = np.linalg.norm(to_goal)
            
            # Calculate angle to goal
            goal_angle = np.arctan2(to_goal[1], to_goal[0])
            angle_diff = (goal_angle - self.rover_heading + 2*np.pi) % (2*np.pi)
            
            # Normalize to [-pi, pi]
            if angle_diff > np.pi:
                angle_diff -= 2*np.pi
            
            # If we're close to goal, stop
            if goal_distance < 0.5:
                return 'stop'
            
            # If we're facing the wrong way, turn
            if abs(angle_diff) > 0.3:
                if angle_diff > 0:
                    return 'left'
                else:
                    return 'right'
            else:
                # We're heading in the right direction, move forward
                return 'forward'
    
    def visualize(self):
        """Enhanced visualization of the simulation state"""
        self.ax.clear()
        
        # Plot lunar terrain
        if self.fancy_graphics:
            # Use the terrain grid for enhanced visualization
            terrain_img = self.ax.imshow(
                self.terrain, 
                cmap='gray', 
                origin='lower',
                extent=(0, self.arena_width, 0, self.arena_height),
                vmin=0, vmax=1
            )
        else:
            # Simple background
            self.ax.add_patch(patches.Rectangle(
                (0, 0), self.arena_width, self.arena_height,
                color='gray', alpha=0.3
            ))
        
        # Draw excavation zone
        excavation_rect = patches.Rectangle(
            (self.excavation_zone['x'][0], self.excavation_zone['y'][0]),
            self.excavation_zone['x'][1] - self.excavation_zone['x'][0],
            self.excavation_zone['y'][1] - self.excavation_zone['y'][0],
            linewidth=2, edgecolor='green', facecolor='green', alpha=0.2
        )
        self.ax.add_patch(excavation_rect)
        self.ax.text(
            self.excavation_zone['x'][0] + 0.1,
            self.excavation_zone['y'][0] + 0.1,
            'EXCAVATION ZONE',
            color='green', fontsize=9, fontweight='bold'
        )
        
        # Draw construction zone
        construction_rect = patches.Rectangle(
            (self.construction_zone['x'][0], self.construction_zone['y'][0]),
            self.construction_zone['x'][1] - self.construction_zone['x'][0],
            self.construction_zone['y'][1] - self.construction_zone['y'][0],
            linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.2
        )
        self.ax.add_patch(construction_rect)
        self.ax.text(
            self.construction_zone['x'][0] + 0.1,
            self.construction_zone['y'][0] + 0.1,
            'CONSTRUCTION ZONE',
            color='blue', fontsize=9, fontweight='bold'
        )
        
        # Plot obstacles (boulders)
        for obstacle in self.obstacles:
            obstacle_pos = obstacle['position']
            obstacle_size = obstacle['size']
            
            if self.fancy_graphics:
                # Fancy boulder with shading
                boulder = patches.Circle(
                    obstacle_pos, obstacle_size/2,
                    facecolor='#555555', edgecolor='#333333',
                    linewidth=1, alpha=0.9
                )
                self.ax.add_patch(boulder)
                
                # Add a highlight to give 3D appearance
                highlight = patches.Circle(
                    obstacle_pos + np.array([0.05, 0.05]), obstacle_size/4,
                    facecolor='#777777', alpha=0.7
                )
                self.ax.add_patch(highlight)
            else:
                # Simple boulder
                boulder = patches.Circle(
                    obstacle_pos, obstacle_size/2,
                    facecolor='black', alpha=0.7
                )
                self.ax.add_patch(boulder)
        
        # Plot dust particles
        if self.fancy_graphics:
            self._update_dust_particles()
            for particle in self.dust_particles:
                dust = patches.Circle(
                    particle['position'], 0.05 * particle['life'],
                    facecolor='#DDDDDD', alpha=0.5 * particle['life']
                )
                self.ax.add_patch(dust)
        
        # Plot rover path
        path = np.array(self.rover_path)
        self.ax.plot(path[:, 0], path[:, 1], color='#77AAFF', linestyle='--', alpha=0.6, linewidth=1)
        
        # Plot rover - use different appearance based on state
        if self.fancy_graphics:
            # Fancy rover with orientation and state indicator
            rover_color = 'blue'
            if self.rover_state == RoverState.MOVING:
                rover_color = '#44FF44'  # Green when moving
            elif self.rover_state == RoverState.TURNING:
                rover_color = '#FFFF00'  # Yellow when turning
            elif self.rover_state == RoverState.STOPPED:
                rover_color = '#FF4444'  # Red when stopped
            
            # Main rover body
            rover_rect = patches.Rectangle(
                self.rover_position - np.array([self.rover_size/2, self.rover_size/2]),
                self.rover_size, self.rover_size,
                angle=np.degrees(self.rover_heading),
                facecolor=rover_color, edgecolor='white',
                linewidth=1.5, alpha=0.8
            )
            self.ax.add_patch(rover_rect)
            
            # Directional indicator
            head_pos = self.rover_position + np.array([
                np.cos(self.rover_heading), 
                np.sin(self.rover_heading)
            ]) * (self.rover_size/2)
            
            head_indicator = patches.Circle(
                head_pos, self.rover_size/6,
                facecolor='white', edgecolor='black',
                linewidth=1, alpha=0.9
            )
            self.ax.add_patch(head_indicator)
        else:
            # Simple rover representation
            rover_circle = patches.Circle(
                self.rover_position, self.rover_size/2,
                facecolor='blue', alpha=0.7
            )
            self.ax.add_patch(rover_circle)
            
            # Heading indicator
            heading_line = np.array([
                self.rover_position,
                self.rover_position + np.array([
                    np.cos(self.rover_heading), 
                    np.sin(self.rover_heading)
                ]) * self.rover_size
            ])
            self.ax.plot(heading_line[:, 0], heading_line[:, 1], 'r-')
        
        # Plot sensor readings (visualization of distances)
        readings = self.sensor_scan()
        for angle, distance in readings:
            # Skip if distance is at max range
            if distance >= self.sensor_range:
                continue
                
            end_point = self.rover_position + np.array([
                np.cos(angle), 
                np.sin(angle)
            ]) * distance
            
            # Color based on distance (red for close, yellow for medium, green for far)
            if distance < self.min_obstacle_distance:
                color = '#FF0000'  # Red for close obstacles
                linewidth = 2
            elif distance < self.min_obstacle_distance * 2:
                color = '#FFFF00'  # Yellow for medium distance
                linewidth = 1.5
            else:
                color = '#00FF00'  # Green for far distance
                linewidth = 1
                
            # Draw sensor line
            self.ax.plot(
                [self.rover_position[0], end_point[0]],
                [self.rover_position[1], end_point[1]],
                color=color, linestyle='-', linewidth=linewidth, alpha=0.7
            )
            
            # Add point at collision
            self.ax.plot(
                end_point[0], end_point[1],
                marker='o', markersize=3, color=color
            )
        
        # Plot goal
        goal_circle = patches.Circle(
            self.goal_position, 0.3,
            facecolor='none', edgecolor='#00FF00',
            linewidth=2, linestyle='--'
        )
        self.ax.add_patch(goal_circle)
        self.ax.plot(
            self.goal_position[0], self.goal_position[1],
            marker='*', markersize=15, color='#00FF00'
        )
        self.ax.text(
            self.goal_position[0] + 0.3,
            self.goal_position[1] + 0.3,
            'GOAL',
            color='#00FF00', fontsize=10, fontweight='bold'
        )
        
        # Set plot properties
        self.ax.set_xlim(0, self.arena_width)
        self.ax.set_ylim(0, self.arena_height)
        self.ax.set_aspect('equal')
        
        # Add title and labels with styling
        self.ax.set_title(
            'NASA Lunabotics Rover Navigation Simulation',
            color='white', fontsize=14, fontweight='bold', pad=10
        )
        self.ax.set_xlabel('X (meters)', color='white', fontsize=10)
        self.ax.set_ylabel('Y (meters)', color='white', fontsize=10)
        
        # Style ticks
        self.ax.tick_params(colors='white')
        
        # Update dashboard
        self.update_dashboard()
    
    def run_simulation(self, num_steps=200):
        """Run simulation for specified number of steps"""
        # Setup interactive mode
        plt.ion()
        
        try:
            for step in range(num_steps):
                # Plan and execute action
                action = self.plan_path()
                self.move_rover(action)
                
                # Visualize
                self.visualize()
                plt.draw()
                plt.pause(0.05)  # Faster updates
                
                # Check if goal reached
                if np.linalg.norm(self.rover_position - self.goal_position) < 0.5:
                    self.dashboard_elements['title'].set_text('MISSION ACCOMPLISHED!')
                    self.dashboard_elements['title'].set_color('#00FF00')
                    plt.draw()
                    print("Goal reached in", step, "steps!")
                    break
            
            plt.ioff()
            plt.show()
            
        except KeyboardInterrupt:
            print("Simulation stopped by user")
            plt.ioff()

# Example usage
if __name__ == "__main__":
    # Create simulation environment
    simulation = LunaboticsSimulation(arena_size=(10, 10), obstacle_count=20, fancy_graphics=True)
    
    # Run simulation
    simulation.run_simulation(num_steps=300)

