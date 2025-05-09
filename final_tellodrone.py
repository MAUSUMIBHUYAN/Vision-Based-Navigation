import cv2
import numpy as np
from djitellopy import Tello
import time
import sys
import signal
from typing import Tuple, Optional, List

class TelloControlledLanding:
    def __init__(self):
        self.tello = Tello()
        self.tello.LOGGER.setLevel("ERROR") # Reduce Tello log spam

        # Color detection parameters (Blue)
        self.blue_lower = np.array([90, 120, 80])  # Hue, Saturation, Value
        self.blue_upper = np.array([130, 255, 255])
        
        # Detection parameters
        self.min_contour_area = 300      # Minimum area to consider as a valid marker part
        self.ignore_area_threshold = 100 # Ignore very small blue areas during initial contour finding
        self.takeoff_height = 100        # Initial height after takeoff (cm) - Tello's move_up is relative
        self.stabilize_time = 2.0        # Time to stabilize with marker in view before descending

        self.running = False
        
        # Landing control parameters
        self.descent_speed = 20          # Max descent speed (cm/s) when far
        self.min_descent_speed = 10      # Min descent speed (cm/s) when close
        self.target_bbox_area = 50000    # Target area of the marker's bounding box for landing
        self.bbox_area_tolerance = 10000 # Tolerance around target area (+/-)
        
        # Frame processing
        self.frame_width = 640 # Reduced for potentially faster processing
        self.frame_height = 480
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        # PID-like Proportional gains and limits for centering
        self.P_GAIN_LR = 0.15            # Proportional gain for left/right error (X-axis)
        self.P_GAIN_FB = 0.15            # Proportional gain for forward/backward error (Y-axis)
        self.MAX_SPEED_LR_FB = 20      # Max speed for left/right and forward/backward adjustments (cm/s)
        # --- MODIFIED THRESHOLDS ---
        self.CENTER_THRESHOLD_X = 20     # Pixel threshold for considering X-axis centered (Increased from 10)
        self.CENTER_THRESHOLD_Y = 20     # Pixel threshold for considering Y-axis centered (Increased from 10)
        # --- END MODIFICATION ---
        self.LAND_DECISION_AREA = 60000  # If area exceeds this, and centered, land immediately

        # Square spiral search parameters
        self.search_speed = 30           # Movement speed during search (cm/s)
        self.spiral_initial_step = 50    # Initial step size (cm)
        self.spiral_step_increment = 30  # Step size increment (cm)
        self.spiral_max_cycles = 3       # Maximum number of square cycles
        self.search_pause = 1.0          # Pause between movements to detect markers (increased slightly)
        
        # Visualization
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        
        # State tracking
        self.last_marker_time = 0
        self.search_state = "STOPPED"    # STOPPED, SQUARE_SPIRAL
        self.spiral_cycle = 0
        self.spiral_step_size = 0
        self.spiral_steps_taken = 0
        self.search_start_time = 0
        self.last_move_time = 0
        self.current_error_x = 0 # For display
        self.current_error_y = 0 # For display

    def detect_markers(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """Detect all blue markers and return their bounding boxes and mask."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], mask
            
        # Get all contours that meet area threshold
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.ignore_area_threshold]
        bboxes = [cv2.boundingRect(c) for c in valid_contours]
        
        # Sort by area (largest first) and filter by minimum area for a primary marker
        bboxes = [b for b in bboxes if b[2]*b[3] >= self.min_contour_area]
        bboxes.sort(key=lambda b: b[2]*b[3], reverse=True)
        
        return bboxes, mask

    def signal_handler(self, sig, frame):
        print("\nCtrl+C detected. Landing safely...")
        self.cleanup()
        sys.exit(0)

    def connect(self) -> bool:
        """Connect to Tello drone and start video stream."""
        print("Connecting to Tello...")
        try:
            self.tello.connect()
            print(f"Battery: {self.tello.get_battery()}%")
            if self.tello.get_battery() < 20:
                print("Battery low. Please charge.")
                return False
            self.tello.streamon()
            time.sleep(2) # Allow time for stream to start
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def takeoff(self) -> bool:
        """Perform takeoff and ascend to initial height."""
        try:
            print("Taking off...")
            self.tello.takeoff()
            time.sleep(2) # Wait for takeoff to complete
            # Tello's takeoff height is variable, so go to a known height if possible
            # For simplicity, we assume takeoff gets it somewhat high, then we adjust.
            # If using Tello EDU, `go_xyz_speed` to a specific height would be better.
            # For regular Tello, `move_up` is relative. We'll rely on visual servoing.
            # Let's ensure it's at least `takeoff_height` by moving up.
            print(f"Moving up to approx {self.takeoff_height}cm from current hover height.")
            self.tello.move_up(self.takeoff_height) # This is relative
            time.sleep(3) # Allow time for movement and stabilization
            return True
        except Exception as e:
            print(f"Takeoff or initial ascent failed: {e}")
            self.tello.land() # Attempt to land if takeoff sequence fails
            return False

    def add_visual_info(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                        detection_stable_timer_start: float, battery: int, state: str) -> np.ndarray:
        """Add visual information to the display frame."""
        # Add battery info
        cv2.putText(frame, f"Battery: {battery}%", (10, 20), 
                    self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Add state info
        cv2.putText(frame, f"State: {state}", (10, 40), 
                    self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Add centering errors
        cv2.putText(frame, f"Error X: {self.current_error_x}px", (10, 60),
                    self.font, self.font_scale, (0, 255, 255), self.font_thickness)
        cv2.putText(frame, f"Error Y: {self.current_error_y}px", (10, 80),
                    self.font, self.font_scale, (0, 255, 255), self.font_thickness)

        # Add search info if in search mode
        y_offset = 100
        if self.search_state != "STOPPED":
            cv2.putText(frame, f"Search: {self.search_state}", (10, y_offset), 
                        self.font, self.font_scale, (0, 255, 255), self.font_thickness)
            y_offset += 20
            cv2.putText(frame, f"Cycle: {self.spiral_cycle+1}/{self.spiral_max_cycles}", (10, y_offset), 
                        self.font, self.font_scale, (0, 255, 255), self.font_thickness)
            y_offset += 20
            cv2.putText(frame, f"Step size: {self.spiral_step_size}cm", (10, y_offset), 
                        self.font, self.font_scale, (0, 255, 255), self.font_thickness)
            y_offset += 20
        
        # Draw all detected markers
        for i, (x, y, w, h) in enumerate(bboxes):
            bbox_area = w * h
            marker_center_x_viz = x + w // 2
            marker_center_y_viz = y + h // 2
            
            # Draw bounding box and center point
            color = (0, 255, 0) if i == 0 else (0, 0, 255) # Primary marker green, others red
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.circle(frame, (marker_center_x_viz, marker_center_y_viz), 5, color, -1) # Center of marker
            
            # Add area info
            cv2.putText(frame, f"Area: {bbox_area}", (x, y-5 if y > 10 else y + h + 15), 
                        self.font, self.font_scale, color, self.font_thickness)
            
            # For primary marker, draw lines to frame center
            if i == 0:
                cv2.line(frame, self.frame_center, (marker_center_x_viz, marker_center_y_viz), (255, 255, 0), 1)

        # Add stabilization countdown if needed
        if detection_stable_timer_start > 0: # Timer has started
            elapsed_time = time.time() - detection_stable_timer_start
            remaining = max(0, self.stabilize_time - elapsed_time)
            if remaining > 0:
                 cv2.putText(frame, f"Stabilizing: {remaining:.1f}s", (10, y_offset), 
                            self.font, self.font_scale, (0, 255, 255), self.font_thickness)
        
        return frame

    def control_descent(self, primary_bbox: Tuple[int, int, int, int]) -> bool:
        """Control the drone's centering and descent based on the primary marker.
        Returns True if ready to land, False otherwise.
        """
        x, y, w, h = primary_bbox
        bbox_area = w * h
        marker_center_x = x + w // 2
        marker_center_y = y + h // 2

        # Calculate errors from the frame center
        self.current_error_x = marker_center_x - self.frame_center[0]
        self.current_error_y = marker_center_y - self.frame_center[1] # Positive: marker below center

        # Initialize RC control values
        left_right_velocity = 0
        forward_backward_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0 # Yaw is not actively controlled for centering here, assuming drone faces general direction

        # --- Centering Logic (X and Y axes) ---
        # Check if the drone is centered within the defined thresholds
        centered_x = abs(self.current_error_x) <= self.CENTER_THRESHOLD_X
        centered_y = abs(self.current_error_y) <= self.CENTER_THRESHOLD_Y

        if not centered_x:
            # Move drone left/right to center the marker
            # Positive error_x means marker is to the right, drone should move right.
            left_right_velocity = int(np.clip(self.current_error_x * self.P_GAIN_LR, 
                                              -self.MAX_SPEED_LR_FB, self.MAX_SPEED_LR_FB))
        
        if not centered_y:
            # Move drone forward/backward to center the marker
            # Positive error_y means marker is below frame center, drone needs to move backward.
            # Negative error_y means marker is above frame center, drone needs to move forward.
            forward_backward_velocity = int(np.clip(-self.current_error_y * self.P_GAIN_FB,
                                                    -self.MAX_SPEED_LR_FB, self.MAX_SPEED_LR_FB))

        # If not centered according to the (now slightly looser) thresholds, prioritize centering. 
        # Descend very slowly while attempting to center.
        if not centered_x or not centered_y:
            up_down_velocity = -5 # Gentle descent while centering
            self.tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
            print(f"Centering: ErrX({self.current_error_x}), ErrY({self.current_error_y}) -> LR_Vel({left_right_velocity}), FB_Vel({forward_backward_velocity}), UD_Vel({up_down_velocity})")
            return False # Not ready for final land decision, continue centering

        # --- Descent Logic (if centered within the new thresholds) ---
        # This section is reached if abs(error_x) <= CENTER_THRESHOLD_X AND abs(error_y) <= CENTER_THRESHOLD_Y
        print(f"Centered (within {self.CENTER_THRESHOLD_X}px X, {self.CENTER_THRESHOLD_Y}px Y)! Area: {bbox_area}, Target: {self.target_bbox_area}")

        # If marker area is very large (very close) and centered, decide to land
        if bbox_area > self.LAND_DECISION_AREA:
            print(f"Marker area {bbox_area} > LAND_DECISION_AREA {self.LAND_DECISION_AREA}. Final landing.")
            self.tello.send_rc_control(0,0,0,0) # Stop all movement
            return True # Ready to land

        # If marker area is within the target range (and already considered centered enough)
        if (self.target_bbox_area - self.bbox_area_tolerance) <= bbox_area <= (self.target_bbox_area + self.bbox_area_tolerance):
            print("Final landing position reached! Area within tolerance and centered enough.")
            self.tello.send_rc_control(0, 0, 0, 0) # Stop all movement
            return True # Proceed to land

        # If marker is too small (too high), descend
        elif bbox_area < (self.target_bbox_area - self.bbox_area_tolerance):
            # Speed up descent if further away (smaller area_ratio)
            area_ratio = bbox_area / self.target_bbox_area
            dynamic_descent_rate = (1.0 - max(0, min(1, area_ratio))) # 1 when far, 0 when at target
            current_descent_speed = int(self.min_descent_speed + (self.descent_speed - self.min_descent_speed) * dynamic_descent_rate)
            up_down_velocity = -max(self.min_descent_speed, current_descent_speed) # Ensure at least min_descent_speed
            
            print(f"Descending: Area {bbox_area} < Target. Speed: {up_down_velocity} cm/s")
            self.tello.send_rc_control(0, 0, up_down_velocity, 0) # Centered, so only descend
            return False # Continue descent
        
        # If marker is too large (too low), ascend slightly
        elif bbox_area > (self.target_bbox_area + self.bbox_area_tolerance):
            up_down_velocity = 5 # Gentle ascent
            print(f"Too close: Area {bbox_area} > Target. Ascending slightly.")
            self.tello.send_rc_control(0, 0, up_down_velocity, 0)
            return False # Adjust height
        
        # Fallback, should ideally be covered by above conditions if centered.
        # If it reaches here, it means it's centered but area is not in any specific landing/adjusting category.
        # This might indicate a need to adjust area tolerances or target_bbox_area.
        # For safety, we can just hover or continue gentle descent.
        print(f"Centered, but area ({bbox_area}) not in land/adjust range. Holding/Gentle Descent.")
        self.tello.send_rc_control(0,0,-5,0) # Gentle descent or hover
        return False


    def start_square_spiral_search(self):
        """Initialize the square spiral search pattern."""
        if self.search_state == "SQUARE_SPIRAL": # Already searching
            return
        self.search_state = "SQUARE_SPIRAL"
        self.spiral_cycle = 0
        self.spiral_step_size = self.spiral_initial_step
        self.spiral_steps_taken = 0 # Number of sides completed in the current length
        self.search_start_time = time.time()
        self.last_move_time = time.time() # Allow first move immediately
        print(f"Starting square spiral search: Initial step {self.spiral_step_size}cm")
        self.tello.send_rc_control(0,0,0,0) # Ensure stationary before search starts
        time.sleep(0.5)


    def update_square_spiral_search(self):
        """Execute the next step in the square spiral search pattern using non-blocking RC commands."""
        current_time = time.time()

        if current_time - self.last_move_time < self.search_pause: # Wait for pause duration
            return

        # Stop any previous RC command before issuing a new move command
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1) # Brief pause to ensure command is processed
        
        # Move forward by current step size
        print(f"Spiral Search: Moving forward {self.spiral_step_size}cm (Cycle {self.spiral_cycle+1})")
        try:
            self.tello.move_forward(self.spiral_step_size)
            time.sleep(1.0) # Allow move to complete and stabilize a bit
        except Exception as e:
            print(f"Error during spiral move_forward: {e}")
            self.search_state = "STOPPED" # Stop search on error
            return
        
        self.last_move_time = time.time() # Reset timer after move
        self.spiral_steps_taken += 1

        # Rotate 90 degrees clockwise
        print("Spiral Search: Rotating 90 degrees clockwise")
        try:
            self.tello.rotate_clockwise(90)
            time.sleep(1.5) # Allow rotation to complete
        except Exception as e:
            print(f"Error during spiral rotation: {e}")
            self.search_state = "STOPPED" # Stop search on error
            return
        
        self.last_move_time = time.time() # Reset timer after rotation

        # After every 2 steps (sides of the expanding square), increase step size for next pair
        if self.spiral_steps_taken % 2 == 0:
            self.spiral_step_size += self.spiral_step_increment
            print(f"Spiral Search: Increased step size to {self.spiral_step_size}cm")
        
        if self.spiral_steps_taken >= (self.spiral_max_cycles * 2) : 
            self.search_state = "STOPPED"
            print("Spiral search completed (max cycles reached). Hovering.")
            self.tello.send_rc_control(0, 0, 0, 0) # Hover
            self.last_marker_time = time.time() 
            return

    def run(self):
        """Main control loop for detection and landing."""
        signal.signal(signal.SIGINT, self.signal_handler) # Setup Ctrl+C handler

        if not self.connect():
            self.cleanup()
            return
        
        if self.tello.get_battery() < 20:
            print("Battery too low to start. Please charge.")
            self.cleanup()
            return

        if not self.takeoff():
            self.cleanup()
            return

        self.running = True
        detection_stable_timer_start = 0 
        detection_stable = False
        landing_phase_initiated = False 

        self.last_marker_time = time.time() 

        try:
            while self.running:
                frame_read_obj = self.tello.get_frame_read()
                if frame_read_obj is None:
                    time.sleep(0.01)
                    continue
                frame = frame_read_obj.frame
                
                if frame is None:
                    time.sleep(0.01) 
                    continue
                
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                bboxes, mask = self.detect_markers(frame)
                display_frame = frame.copy()
                
                battery = self.tello.get_battery()
                if battery < 10:
                    print("Critical battery! Landing now.")
                    break 

                current_time = time.time()
                current_state_display = "HOVERING"

                if bboxes: 
                    self.last_marker_time = current_time
                    primary_bbox = bboxes[0]
                    
                    if self.search_state != "STOPPED":
                        print("Marker detected - stopping search pattern.")
                        self.tello.send_rc_control(0, 0, 0, 0) 
                        self.search_state = "STOPPED"
                        time.sleep(0.5) 

                    current_state_display = "DETECTED"
                    if not detection_stable:
                        if detection_stable_timer_start == 0: 
                            print("Marker detected, starting stabilization timer.")
                            detection_stable_timer_start = current_time
                        elif current_time - detection_stable_timer_start >= self.stabilize_time:
                            print("Marker detection stable.")
                            detection_stable = True
                            landing_phase_initiated = True 
                            current_state_display = "STABILIZED/LANDING"
                        else:
                            pass 
                    
                    if detection_stable and landing_phase_initiated:
                        current_state_display = "LANDING_CONTROL"
                        should_land_now = self.control_descent(primary_bbox)
                        if should_land_now:
                            print("Control_descent returned TRUE. Proceeding to final land.")
                            break 
                
                else: 
                    detection_stable_timer_start = 0 
                    detection_stable = False
                    self.current_error_x = 0 
                    self.current_error_y = 0

                    if landing_phase_initiated: 
                        print("Marker lost during landing phase. Attempting to hold position / slow descent.")
                        self.tello.send_rc_control(0, 0, -5, 0) 
                        current_state_display = "RECOVERING"
                        if current_time - self.last_marker_time > 5.0: 
                            print("Marker lost for too long during recovery. Stopping landing phase. Re-evaluating.")
                            landing_phase_initiated = False 
                            self.start_square_spiral_search() 
                    
                    elif self.search_state == "STOPPED" and (current_time - self.last_marker_time > 3.0):
                        print("No marker detected for 3s and not searching. Starting spiral search.")
                        self.start_square_spiral_search()
                    
                    if self.search_state == "SQUARE_SPIRAL":
                        current_state_display = "SEARCHING"
                        self.update_square_spiral_search()
                    elif self.search_state == "STOPPED" and not landing_phase_initiated:
                        self.tello.send_rc_control(0,0,0,0)
                        current_state_display = "HOVERING (No Marker)"

                display_frame = self.add_visual_info(display_frame, bboxes, detection_stable_timer_start, battery, current_state_display)
                
                cv2.drawMarker(display_frame, self.frame_center, (0, 0, 255), 
                               cv2.MARKER_CROSS, markerSize=20, thickness=2)
                
                cv2.imshow("Tello Camera - Press Q to quit", display_frame)
                if mask is not None and mask.size > 0 : # Check if mask is valid before showing
                    cv2.imshow("Detection Mask", mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Q pressed. Landing...")
                    break
                elif key == ord('s'): 
                    if self.search_state == "STOPPED":
                        print("Manual 's' pressed: Starting spiral search.")
                        self.start_square_spiral_search()
                    else:
                        print("Manual 's' pressed: Stopping search.")
                        self.search_state = "STOPPED"
                        self.tello.send_rc_control(0,0,0,0)

                time.sleep(0.03)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources and land safely."""
        print("Cleaning up and landing...")
        self.running = False
        try:
            if hasattr(self.tello, 'send_rc_control'): 
                 self.tello.send_rc_control(0, 0, 0, 0)
                 time.sleep(0.1)

            if hasattr(self.tello, 'is_flying') and self.tello.is_flying:
                print("Landing drone...")
                self.tello.land()
                time.sleep(3) 
            
            if hasattr(self.tello, 'streamoff'):
                self.tello.streamoff()

        except Exception as e:
            print(f"Error during Tello cleanup: {e}")
        finally:
            cv2.destroyAllWindows()
            print("Cleanup complete.")


if __name__ == "__main__":
    print("Tello Controlled Blue Marker Landing System - V2 (Precision w. Looser Thresholds)")
    controller = TelloControlledLanding()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main. Cleaning up...")
        controller.cleanup()
    except Exception as e:
        print(f"Fatal error in script execution: {e}")
        import traceback
        traceback.print_exc()
        if 'controller' in locals() and controller is not None:
            controller.cleanup()
        sys.exit(1)
