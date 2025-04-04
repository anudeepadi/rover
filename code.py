# Tracked rover geometry (in meters)
ROVER_LENGTH = 0.52  # Length from front to back
ROVER_WIDTH = 0.48   # Width from left to right track
TRACK_LENGTH = 0.45  # Length of each track

class Track(object):
    """Information about a track in our rover.
    Axis orientation: +X is forward, +Y is left, +Z is up
    """

    def __init__(self, name, offset_left):
        """Initialize a track with its position relative to center.
        Args:
            name: name for this track (e.g., 'left', 'right')
            offset_left: left/right position relative to center, positive left.
        """
        self.name = name
        self.offset_left = offset_left
        # For a tracked system, tracks always align with the rover's long axis
        self.angle = 0.0

class TrackVelocity(object):
    """Results of calculation for a named track."""

    def __init__(self, name, velocity):
        """Initialize a track with its calculated velocity.
        Args:
            name: name for this track
            velocity: velocity for this track in meters/second
        """
        self.name = name
        self.velocity = velocity

# Define our tracked rover with left and right tracks
rover_tracks = [
    Track('left', ROVER_WIDTH/2),
    Track('right', -ROVER_WIDTH/2)
]

import math

# Commanded inputs
velocityAngular = 0.5  # radians/sec, positive is counterclockwise rotation
velocityLinear = 0.2   # meters/sec, positive is forward

def calculate_track_velocities(tracks, vel_linear, vel_angular):
    """Calculate velocities for tracks based on linear and angular velocity commands.
    
    Args:
        tracks: List of Track objects
        vel_linear: Linear velocity in m/s (positive = forward)
        vel_angular: Angular velocity in rad/s (positive = counterclockwise)
        
    Returns:
        List of TrackVelocity objects
    """
    results = []
    
    for track in tracks:
        # Calculate velocity contribution from angular velocity
        # Track on the outside of the turn moves faster
        angular_component = vel_angular * track.offset_left
        
        # Total track velocity is sum of linear and angular components
        track_velocity = vel_linear + angular_component
        
        results.append(TrackVelocity(track.name, track_velocity))
        
    return results

# Calculate track velocities
track_velocities = calculate_track_velocities(rover_tracks, velocityLinear, velocityAngular)

# Save results in a dictionary for easy access
velocity_dict = {}
for result in track_velocities:
    velocity_dict[result.name] = result.velocity

import matplotlib.pyplot as plt
import numpy as np
# if using a Jupyter notebook, include:
# %matplotlib inline

def plot_tracked_rover(tracks, velocities, rover_length, rover_width, vel_linear, vel_angular):
    """Plot a simple visualization of the tracked rover and its track velocities."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot rover body outline
    half_length = rover_length / 2
    half_width = rover_width / 2
    rover_corners = np.array([
        [-half_length, -half_width],  # back right
        [half_length, -half_width],   # front right
        [half_length, half_width],    # front left
        [-half_length, half_width],   # back left
        [-half_length, -half_width]   # back to start to close the outline
    ])
    ax.plot(rover_corners[:, 0], rover_corners[:, 1], 'k-', linewidth=2)
    
    # Plot tracks
    track_positions = {}
    for track in tracks:
        y_pos = track.offset_left
        track_x = np.array([-half_length, half_length])
        track_y = np.array([y_pos, y_pos])
        ax.plot(track_x, track_y, 'g-', linewidth=6)
        
        # Store track position for velocity arrows
        track_positions[track.name] = (0, y_pos)
    
    # Add velocity arrows
    max_velocity = max(abs(v) for v in velocities.values())
    if max_velocity > 0:
        arrow_scale = rover_length / (max_velocity * 2)
    else:
        arrow_scale = 1
    
    for track_name, velocity in velocities.items():
        x, y = track_positions[track_name]
        dx = velocity * arrow_scale
        ax.arrow(x, y, dx, 0, head_width=0.05, head_length=0.1, fc='r', ec='r', linewidth=2)
        
        # Add velocity text
        ax.text(x + dx/2, y + 0.1, f"{velocity:.2f} m/s", ha='center')
    
    # Set plot properties
    ax.set_xlim(-rover_length, rover_length)
    ax.set_ylim(-rover_width, rover_width)
    ax.set_aspect('equal')
    ax.set_title(f'Tracked Rover Motion (Linear: {vel_linear} m/s, Angular: {vel_angular} rad/s)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True)
    
    plt.tight_layout()
    return fig, ax

# Plot the rover and track velocities
fig, ax = plot_tracked_rover(rover_tracks, velocity_dict, ROVER_LENGTH, ROVER_WIDTH, velocityLinear, velocityAngular)
plt.show()  # Explicitly show the plot

def calculate_turn_radius(vel_linear, vel_angular):
    """Calculate the instantaneous turn radius for given velocity commands.
    
    Args:
        vel_linear: Linear velocity in m/s
        vel_angular: Angular velocity in rad/s
        
    Returns:
        Turn radius in meters (positive = counterclockwise turn, negative = clockwise)
        Returns infinity for straight-line motion
    """
    if vel_angular == 0:
        return float('inf')  # Straight line motion
    else:
        return vel_linear / vel_angular

# Calculate turn radius
turn_radius = calculate_turn_radius(velocityLinear, velocityAngular)
print(f"Turn radius: {turn_radius:.2f} meters")

def point_turn(tracks, angular_speed=0.5):
    """Calculate track velocities for turning in place.
    
    Args:
        tracks: List of Track objects
        angular_speed: Speed of rotation in rad/s
        
    Returns:
        List of TrackVelocity objects
    """
    return calculate_track_velocities(tracks, 0, angular_speed)

def neutral_turn(tracks, linear_speed=0.2, turning_radius=1.0):
    """Calculate track velocities for a turn with a specific radius.
    
    Args:
        tracks: List of Track objects
        linear_speed: Forward speed in m/s
        turning_radius: Desired turn radius in meters
        
    Returns:
        List of TrackVelocity objects
    """
    if turning_radius == 0:
        return point_turn(tracks)
    
    angular_speed = linear_speed / turning_radius
    return calculate_track_velocities(tracks, linear_speed, angular_speed)

# Example: Calculate speeds for a point turn
point_turn_velocities = point_turn(rover_tracks)
print("\nPoint turn velocities:")
for result in point_turn_velocities:
    print(f"{result.name}: {result.velocity:.2f} m/s")

# Example: Calculate speeds for a turn with 1-meter radius
neutral_turn_velocities = neutral_turn(rover_tracks)
print("\nNeutral turn velocities (1m radius):")
for result in neutral_turn_velocities:
    print(f"{result.name}: {result.velocity:.2f} m/s")

def velocity_to_motor_signal(velocity, max_velocity=0.5, max_signal=255):
    """Convert a velocity value to a motor control signal.
    
    Args:
        velocity: Track velocity in m/s
        max_velocity: Maximum track velocity in m/s
        max_signal: Maximum motor control signal value
        
    Returns:
        Motor control signal value
    """
    # Clamp velocity to max_velocity
    clamped_velocity = max(min(velocity, max_velocity), -max_velocity)
    
    # Convert to signal value
    signal = int((clamped_velocity / max_velocity) * max_signal)
    
    return signal

# Calculate motor signals for our track velocities
print("\nMotor control signals:")
for track_name, velocity in velocity_dict.items():
    signal = velocity_to_motor_signal(velocity)
    print(f"{track_name}: {signal} (from {velocity:.2f} m/s)")