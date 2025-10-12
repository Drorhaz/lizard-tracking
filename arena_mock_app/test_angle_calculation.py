#!/usr/bin/env python3
"""
Test the angle calculation functions
"""
import math
import sys
import os

# Add current directory to path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import calculate_head_angle_to_target, AngleKalmanFilter

def test_angle_calculation():
    """Test angle calculation with known scenarios"""
    
    print("Testing head angle calculation...")
    
    # Test scenario 1: Head pointing directly right (toward right edge)
    nose = (400, 300)
    ear_left = (380, 290)
    ear_right = (380, 310)
    target_line = 'right'
    target_line_position = 800
    frame_width = 800
    frame_height = 600
    
    angle = calculate_head_angle_to_target(
        nose, ear_left, ear_right, target_line, target_line_position, 
        frame_width, frame_height
    )
    
    print(f"Test 1 - Head pointing right:")
    print(f"  Nose: {nose}, Ear Left: {ear_left}, Ear Right: {ear_right}")
    print(f"  Angle to {target_line} edge: {angle:.1f}°")
    print(f"  Expected: Close to 0° (pointing at target)")
    
    # Test scenario 2: Head pointing up (perpendicular to right edge)
    nose = (400, 300)
    ear_left = (390, 300)
    ear_right = (410, 300)
    
    angle = calculate_head_angle_to_target(
        nose, ear_left, ear_right, target_line, target_line_position, 
        frame_width, frame_height
    )
    
    print(f"\nTest 2 - Head pointing up:")
    print(f"  Nose: {nose}, Ear Left: {ear_left}, Ear Right: {ear_right}")
    print(f"  Angle to {target_line} edge: {angle:.1f}°")
    print(f"  Expected: Close to 90° (perpendicular to target)")
    
    # Test scenario 3: Head pointing left (away from right edge)
    nose = (400, 300)
    ear_left = (420, 290)
    ear_right = (420, 310)
    
    angle = calculate_head_angle_to_target(
        nose, ear_left, ear_right, target_line, target_line_position, 
        frame_width, frame_height
    )
    
    print(f"\nTest 3 - Head pointing left:")
    print(f"  Nose: {nose}, Ear Left: {ear_left}, Ear Right: {ear_right}")
    print(f"  Angle to {target_line} edge: {angle:.1f}°")
    print(f"  Expected: Close to 180° or -180° (pointing away from target)")

def test_kalman_filter():
    """Test Kalman filter for angle smoothing"""
    
    print("\n\nTesting Kalman filter...")
    
    kalman = AngleKalmanFilter(process_noise=1e-4, measurement_noise=1e-1)
    
    # Simulate noisy angle measurements
    true_angles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # Smooth progression
    noise_std = 5.0  # Standard deviation of measurement noise
    
    print("Frame | True Angle | Noisy Measurement | Kalman Filtered | Angular Velocity")
    print("-" * 75)
    
    for i, true_angle in enumerate(true_angles):
        # Add some random noise to simulate measurement uncertainty
        import random
        noise = random.gauss(0, noise_std)
        noisy_measurement = true_angle + noise
        
        # Filter the noisy measurement
        smoothed_angle = kalman.update(noisy_measurement)
        angular_velocity = kalman.get_angular_velocity()
        
        print(f"{i:5d} | {true_angle:10.1f} | {noisy_measurement:17.1f} | {smoothed_angle:15.1f} | {angular_velocity:16.2f}")

if __name__ == "__main__":
    test_angle_calculation()
    test_kalman_filter()