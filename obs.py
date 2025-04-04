#!/usr/bin/env python3
# Simple Rover Obstacle Detection and Avoidance System using OpenCV
# For use with Raspberry Pi and L298N motor controller

import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

class ObstacleAvoidanceRover:
    def __init__(self):
        """Initialize the rover with GPIO pins and camera setup"""
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)  # Use first camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set up GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Motor A (left side)
        self.motor_a_enable = 17
        self.motor_a_pin1 = 27
        self.motor_a_pin2 = 22
        
        # Motor B (right side)
        self.motor_b_enable = 18
        self.motor_b_pin1 = 23
        self.motor_b_pin2 = 24
        
        # Configure GPIO pins
        GPIO.setup(self.motor_a_enable, GPIO.OUT)
        GPIO.setup(self.motor_a_pin1, GPIO.OUT)
        GPIO.setup(self.motor_a_pin2, GPIO.OUT)
        GPIO.setup(self.motor_b_enable, GPIO.OUT)
        GPIO.setup(self.motor_b_pin1, GPIO.OUT)
        GPIO.setup(self.motor_b_pin2, GPIO.OUT)
        
        # Set up PWM for speed control
        self.pwm_a = GPIO.PWM(self.motor_a_enable, 100)
        self.pwm_b = GPIO.PWM(self.motor_b_enable, 100)
        self.pwm_a.start(0)
        self.pwm_b.start(0)
        
        # Obstacle detection parameters
        self.obstacle_threshold = 0.15  # Edge density threshold for obstacle detection
        self.frame_count = 0
        self.last_action_time = time.time()
        self.action_delay = 0.5  # Time to wait between actions (seconds)
        
        print("Rover initialized and ready")
        
    def __del__(self):
        """Clean up GPIO and camera when object is destroyed"""
        self.stop_motors()
        self.camera.release()
        GPIO.cleanup()
        
    def stop_motors(self):
        """Stop all motors"""
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        GPIO.output(self.motor_a_pin1, GPIO.LOW)
        GPIO.output(self.motor_a_pin2, GPIO.LOW)
        GPIO.output(self.motor_b_pin1, GPIO.LOW)
        GPIO.output(self.motor_b_pin2, GPIO.LOW)
        
    def move_forward(self, speed=70):
        """Move the rover forward at the specified speed"""
        # Left motor forward
        GPIO.output(self.motor_a_pin1, GPIO.HIGH)
        GPIO.output(self.motor_a_pin2, GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(speed)
        
        # Right motor forward
        GPIO.output(self.motor_b_pin1, GPIO.HIGH)
        GPIO.output(self.motor_b_pin2, GPIO.LOW)
        self.pwm_b.ChangeDutyCycle(speed)
        
    def turn_left(self, speed=60):
        """Turn the rover left at the specified speed"""
        # Left motor reverse
        GPIO.output(self.motor_a_pin1, GPIO.LOW)
        GPIO.output(self.motor_a_pin2, GPIO.HIGH)
        self.pwm_a.ChangeDutyCycle(speed)
        
        # Right motor forward
        GPIO.output(self.motor_b_pin1, GPIO.HIGH)
        GPIO.output(self.motor_b_pin2, GPIO.LOW)
        self.pwm_b.ChangeDutyCycle(speed)
        
    def turn_right(self, speed=60):
        """Turn the rover right at the specified speed"""
        # Left motor forward
        GPIO.output(self.motor_a_pin1, GPIO.HIGH)
        GPIO.output(self.motor_a_pin2, GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(speed)
        
        # Right motor reverse
        GPIO.output(self.motor_b_pin1, GPIO.LOW)
        GPIO.output(self.motor_b_pin2, GPIO.HIGH)
        self.pwm_b.ChangeDutyCycle(speed)
    
    def preprocess_frame(self, frame):
        """Process camera frame to detect obstacles"""
        # Resize for faster processing (optional)
        frame = cv2.resize(frame, (320, 240))
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection to find edges (potential obstacles)
        edges = cv2.Canny(blurred, 50, 150)
        
        return frame, edges
    
    def detect_obstacles(self, frame, edges):
        """Detect obstacles in the processed frame"""
        height, width = edges.shape
        
        # Define the region of interest (lower half of the image)
        roi_height = height // 2
        roi = edges[roi_height:, :]
        
        # Divide the ROI into three sections (left, center, right)
        left_region = roi[:, :width//3]
        center_region = roi[:, width//3:2*width//3]
        right_region = roi[:, 2*width//3:]
        
        # Calculate the density of edges in each region
        left_density = np.sum(left_region > 0) / left_region.size
        center_density = np.sum(center_region > 0) / center_region.size
        right_density = np.sum(right_region > 0) / right_region.size
        
        # Create a visualization frame with the regions marked
        vis_frame = frame.copy()
        
        # Draw ROI rectangles
        cv2.rectangle(vis_frame, (0, roi_height), (width//3, height), (0, 0, 255), 2)
        cv2.rectangle(vis_frame, (width//3, roi_height), (2*width//3, height), (0, 0, 255), 2)
        cv2.rectangle(vis_frame, (2*width//3, roi_height), (width, height), (0, 0, 255), 2)
        
        # Add density text
        cv2.putText(vis_frame, f"L: {left_density:.3f}", (10, height-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, f"C: {center_density:.3f}", (width//3 + 10, height-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, f"R: {right_density:.3f}", (2*width//3 + 10, height-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Determine which regions have obstacles
        obstacle_regions = {
            'left': left_density > self.obstacle_threshold,
            'center': center_density > self.obstacle_threshold,
            'right': right_density > self.obstacle_threshold
        }
        
        return obstacle_regions, vis_frame
    
    def decide_action(self, obstacle_regions):
        """Decide the rover's next action based on obstacle detection"""
        current_time = time.time()
        
        # Only change action if enough time has passed since last action
        if current_time - self.last_action_time < self.action_delay:
            return None
        
        self.last_action_time = current_time
        
        # No obstacles detected, move forward
        if not any(obstacle_regions.values()):
            return "FORWARD"
        
        # Center is blocked
        if obstacle_regions['center']:
            # Both sides are clear, choose one
            if not obstacle_regions['left'] and not obstacle_regions['right']:
                # Alternate between left and right
                return "LEFT" if self.frame_count % 2 == 0 else "RIGHT"
            # Left is clear, turn left
            elif not obstacle_regions['left']:
                return "LEFT"
            # Right is clear, turn right
            elif not obstacle_regions['right']:
                return "RIGHT"
            # All paths blocked, stop
            else:
                return "STOP"
        
        # Center is clear but side(s) have obstacles
        if obstacle_regions['left'] and not obstacle_regions['right']:
            return "RIGHT"  # Left blocked, turn right
        elif obstacle_regions['right'] and not obstacle_regions['left']:
            return "LEFT"  # Right blocked, turn left
        
        return "FORWARD"  # Default is forward
    
    def execute_action(self, action):
        """Execute the decided action by controlling the motors"""
        if action == "FORWARD":
            self.move_forward()
            print("Moving forward")
        elif action == "LEFT":
            self.turn_left()
            print("Turning left")
        elif action == "RIGHT":
            self.turn_right()
            print("Turning right")
        elif action == "STOP":
            self.stop_motors()
            print("Stopping")
    
    def run(self):
        """Main loop for the obstacle avoidance rover"""
        print("Starting obstacle avoidance routine...")
        
        try:
            while True:
                # Capture frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame from camera")
                    break
                
                self.frame_count += 1
                
                # Process the frame
                processed_frame, edges = self.preprocess_frame(frame)
                
                # Detect obstacles
                obstacle_regions, vis_frame = self.detect_obstacles(processed_frame, edges)
                
                # Decide action
                action = self.decide_action(obstacle_regions)
                
                # Execute action if one was decided
                if action:
                    self.execute_action(action)
                    # Display the action on the visualization frame
                    cv2.putText(vis_frame, f"Action: {action}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the frame (comment out when running headless)
                cv2.imshow('Rover View', vis_frame)
                cv2.imshow('Edge Detection', edges)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Optional: add a short delay to reduce CPU usage
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            # Clean up
            self.stop_motors()
            cv2.destroyAllWindows()
            print("Rover stopped and cleaned up")

# Main execution
if __name__ == "__main__":
    rover = ObstacleAvoidanceRover()
    rover.run()