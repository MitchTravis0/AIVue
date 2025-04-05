import sys
import os
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from filterpy.kalman import KalmanFilter
# Ensure this import is present and correct
try:
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    print("Error: filterpy library not found or Q_discrete_white_noise not available.")
    print("Please install or update filterpy: pip install filterpy")
    exit()


class HeadGazeTracker:
    def __init__(self):
        
        # Set up MediaPipe face mesh with more robust settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7, # Lowered slightly, KF will smooth
            min_tracking_confidence=0.7  # Lowered slightly, KF will smooth
        )

        # More comprehensive landmark selection for better head pose estimation
        self.HEAD_LANDMARKS = [1, 4, 6, 33, 61, 199, 263, 291, 362, 389, 398] # Keep your selection

        # Eye landmarks (more specific for EAR calculation)
        self.LEFT_EYE_V = [159, 145]
        self.RIGHT_EYE_V = [386, 374]
        self.LEFT_EYE_H = [33, 133]
        self.RIGHT_EYE_H = [362, 263]
        self.LEFT_EYE_ALL = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_ALL = [263, 362, 385, 386, 387, 373, 374, 380]

        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

        # Improved calibration settings
        self.calibration_points =[
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),  #corners
            (0.5, 0.1), (0.1, 0.5), (0.9, 0.5), (0.5, 0.9),  #midpoints
            (0.5, 0.5)                                       #dead center
        ]
        self.calibration_data = []
        self.calibration_complete = False
        self.calibration_matrix = None
        self.reference_head_pose = None # Neutral position

        # --- Kalman Filter Setup ---
        # State vector: [x, y, vx, vy] (position and velocity)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Initial State (position and velocity) - start at center with zero velocity
        self.kf.x = np.array([self.screen_width / 2, self.screen_height / 2, 0., 0.]).T

        # State Transition Matrix (F) - predicts next state based on current state
        # Updated dynamically with dt in the loop
        self.kf.F = np.array([[1., 0., 1., 0.],  # x = x + vx*dt
                              [0., 1., 0., 1.],  # y = y + vy*dt
                              [0., 0., 1., 0.],  # vx = vx
                              [0., 0., 0., 1.]]) # vy = vy

        # Measurement Function (H) - maps state space to measurement space
        # We only measure position (x, y)
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])

        # Measurement Noise Covariance (R) - uncertainty in the measurement
        # Tune this based on how noisy the raw mapped coordinates are
        measurement_noise_std = 15.0 # Increased slightly, maybe helps with jitter? TUNABLE
        self.kf.R = np.diag([measurement_noise_std**2, measurement_noise_std**2])

        # Process Noise Covariance (Q) - uncertainty in the process model (dynamic update)
        # Accounts for unmodeled accelerations. Calculated dynamically using dt.
        # Initial placeholder, will be calculated using Q_discrete_white_noise correctly.
        self.kf.Q = np.eye(4)

        # Initial State Covariance (P) - uncertainty in the initial state estimate
        self.kf.P = np.eye(4) * 500. # Start with higher uncertainty

        self.last_time = time.time() # For dt calculation

        # --- Wink Detection Parameters ---
        self.EAR_THRESHOLD = 0.20
        self.WINK_CONSEC_FRAMES = 2
        self.CLICK_COOLDOWN = 0.5 # Time before another click of the *same type* can occur
        self.left_wink_counter = 0
        self.right_wink_counter = 0
        self.left_eye_closed = False
        self.right_eye_closed = False
        self.last_left_click_time = 0 # Tracks when the *action* (click/double) happened for cooldown
        self.last_right_click_time = 0 # Tracks when the right click *action* happened

        # --- NEW: Double Click Parameters ---
        self.DOUBLE_CLICK_WINDOW = 0.4 # Max seconds between two left wink *detections* for double click (tune this!)
        self.last_left_wink_detect_time = 0 # Tracks when a left wink was *detected* (met WINK_CONSEC_FRAMES)
        # --- End Double Click Parameters ---

        self.running = True 
    
    def stop_tracking(self):
        """Stops the tracking loop and cleans up resources."""
        self.running = False
    # --- Helper functions (_calculate_distance, _calculate_ear) remain the same ---
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _calculate_ear(self, landmarks, eye_v_indices, eye_h_indices):
        """Calculate Eye Aspect Ratio (EAR) for one eye."""
        try:
            # Ensure landmarks are accessible and indices are valid
            if not landmarks or max(eye_v_indices + eye_h_indices) >= len(landmarks):
                 print(f"Warning: Landmark index out of bounds (max index needed: {max(eye_v_indices + eye_h_indices)}, available: {len(landmarks)})")
                 return 0.3 # Return a default value
            p_top = landmarks[eye_v_indices[0]]
            p_bottom = landmarks[eye_v_indices[1]]
            v_dist = self._calculate_distance(p_top, p_bottom)
            p_left = landmarks[eye_h_indices[0]]
            p_right = landmarks[eye_h_indices[1]]
            h_dist = self._calculate_distance(p_left, p_right)
            if h_dist < 1e-6: return 0.0 # Avoid division by zero or near-zero
            ear = v_dist / h_dist
            return ear
        except IndexError as e:
            # This catch might be redundant with the check above but kept for safety
            print(f"Warning: IndexError during EAR calculation: {e}. Indices: V={eye_v_indices}, H={eye_h_indices}")
            return 0.3 # Return a value unlikely to trigger a wink
        except Exception as e:
            print(f"Unexpected error during EAR calculation: {e}")
            return 0.3


    # --- _detect_winks function remains the same ---
    def _detect_winks(self, landmarks):
        """Detects left and right winks based on EAR, and handles double clicks."""
        current_time = time.time()
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_V, self.LEFT_EYE_H)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_V, self.RIGHT_EYE_H)

        # Update eye closed status (used for visual feedback and logic)
        self.left_eye_closed = left_ear < self.EAR_THRESHOLD
        self.right_eye_closed = right_ear < self.EAR_THRESHOLD

        # --- Left Wink / Double Click Logic ---
        if self.left_eye_closed and not self.right_eye_closed:
            self.left_wink_counter += 1
        else:
            self.left_wink_counter = 0

        if self.left_wink_counter == self.WINK_CONSEC_FRAMES and \
           current_time - self.last_left_click_time > self.CLICK_COOLDOWN:
            time_since_last_detect = current_time - self.last_left_wink_detect_time
            if time_since_last_detect < self.DOUBLE_CLICK_WINDOW:
                print("Double Left Click Triggered!")
                pyautogui.doubleClick(button='left')
                self.last_left_click_time = current_time
                self.last_left_wink_detect_time = 0 # Prevent triple+ clicks
            else:
                print("Left Click Triggered!")
                pyautogui.click(button='left')
                self.last_left_click_time = current_time
                self.last_left_wink_detect_time = current_time # Start double click window
            self.left_wink_counter = 0 # Reset counter after action

        elif self.left_wink_counter > self.WINK_CONSEC_FRAMES:
             self.left_wink_counter = 0 # Reset if held too long

        # --- Right Wink Logic ---
        if self.right_eye_closed and not self.left_eye_closed:
             self.right_wink_counter += 1
        else:
             self.right_wink_counter = 0

        if self.right_wink_counter == self.WINK_CONSEC_FRAMES and \
           current_time - self.last_right_click_time > self.CLICK_COOLDOWN:
             print("Right Click Triggered!")
             pyautogui.click(button='right')
             self.last_right_click_time = current_time
             self.right_wink_counter = 0 # Reset counter after action
        elif self.right_wink_counter > self.WINK_CONSEC_FRAMES:
            self.right_wink_counter = 0 # Reset if held too long

        return left_ear, right_ear


    # --- _get_head_pose function remains the same ---
    def _get_head_pose(self, landmarks, image):
        """Extract improved head pose information from face landmarks"""
        h, w = image.shape[:2]
        # Check if essential landmarks exist
        required_indices = [4, 10, 152, 133, 362, 33, 263] + self.HEAD_LANDMARKS
        if not all(idx < len(landmarks) for idx in required_indices):
            print("Warning: Not all required landmarks available for head pose.")
            ref_pose = self.reference_head_pose if self.reference_head_pose else (0.5, 0.5, 0.0, 0.0)
            return ref_pose[0], ref_pose[1], ref_pose[2], ref_pose[3]

        head_landmarks_coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in self.HEAD_LANDMARKS])
        avg_x = np.mean(head_landmarks_coords[:, 0])
        avg_y = np.mean(head_landmarks_coords[:, 1])

        try:
            nose_tip = landmarks[4]
            forehead = landmarks[10]
            chin = landmarks[152]
            left_eye_inner = landmarks[133]
            right_eye_inner = landmarks[362]
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
        except IndexError:
             # Should be caught by the check above, but defensive programming
             print("Warning: Landmark index out of bounds during head pose calculation (should have been caught).")
             ref_pose = self.reference_head_pose if self.reference_head_pose else (0.5, 0.5, 0.0, 0.0)
             return ref_pose[0], ref_pose[1], ref_pose[2], ref_pose[3]


        eye_center_x = (left_eye_inner.x + right_eye_inner.x) / 2.0
        nose_offset = nose_tip.x - eye_center_x
        head_yaw = nose_offset * 4.0 # Sensitivity multiplier for yaw - TUNABLE

        eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2.0
        forehead_to_chin = chin.y - forehead.y
        if forehead_to_chin <= 1e-6: forehead_to_chin = 1e-6
        nose_to_eye_ratio = (nose_tip.y - eye_center_y) / forehead_to_chin
        neutral_pitch_ratio = 0.18 # Approximate neutral vertical ratio - TUNABLE
        head_pitch = (nose_to_eye_ratio - neutral_pitch_ratio) * 5.0 # Sensitivity multiplier for pitch - TUNABLE

        return avg_x/w, avg_y/h, head_yaw, head_pitch


    # --- Calibration functions remain the same ---
    def run_calibration(self, cap):
        """Run the improved calibration process"""
        print("Starting calibration process...")
        print("Look at the red circles by turning your head as they appear on the screen.")

        self.calibration_data = []
        self.reference_head_pose = None # Reset reference pose

        # First, capture neutral head pose with better feedback
        print("Look straight ahead at the center of the screen for neutral position calibration...")
        neutral_data = []
        # Ensure windows are created before setting properties or showing images
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Feed", 320, 240)
        cv2.moveWindow("Camera Feed", self.screen_width - 350, self.screen_height - 270)


        calib_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        center_x = int(0.5 * self.screen_width)
        center_y = int(0.5 * self.screen_height)
        cv2.circle(calib_img, (center_x, center_y), 30, (194, 165, 248), 2)
        cv2.circle(calib_img, (center_x, center_y), 10, (62, 33, 22), -1)
        cv2.putText(calib_img, "Look here for neutral position",
                   (center_x - 200, center_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(calib_img, "Keep your head still",
                   (center_x - 150, center_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Calibration", calib_img)
        cv2.waitKey(1) # Needed to display the initial image
        time.sleep(2) # Give user time to focus

        start_time = time.time()
        collection_time = 3
        progress_bar_length = 400

        while time.time() - start_time < collection_time:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
             # Make frame non-writeable BEFORE processing for potential performance gain
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            # Make frame writeable again AFTER processing for drawing
            frame.flags.writeable = True


            elapsed = time.time() - start_time
            progress = int((elapsed / collection_time) * progress_bar_length)
            progress_img = calib_img.copy() # Recopy base image each time
            cv2.rectangle(progress_img, (center_x - progress_bar_length//2, center_y + 100),
                         (center_x - progress_bar_length//2 + progress, center_y + 120), (62, 33, 22), -1)
            cv2.rectangle(progress_img, (center_x - progress_bar_length//2, center_y + 100),
                         (center_x + progress_bar_length//2, center_y + 120), (194, 165, 248), 2)
            cv2.imshow("Calibration", progress_img)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                try:
                    head_x, head_y, head_yaw, head_pitch = self._get_head_pose(landmarks, frame)
                    neutral_data.append((head_x, head_y, head_yaw, head_pitch))
                    self._draw_landmarks(frame, landmarks) # Draw landmarks on feed
                except Exception as e:
                    print(f"Error getting head pose during neutral calibration: {e}")

            cv2.putText(frame, "Keep still, looking at center",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 cv2.destroyAllWindows()
                 return False

        if len(neutral_data) > 10: # Require a minimum number of good readings
            avg_x = np.mean([d[0] for d in neutral_data])
            avg_y = np.mean([d[1] for d in neutral_data])
            avg_yaw = np.mean([d[2] for d in neutral_data])
            avg_pitch = np.mean([d[3] for d in neutral_data])
            self.reference_head_pose = (avg_x, avg_y, avg_yaw, avg_pitch)
            print(f"Neutral position calibrated: Yaw={avg_yaw:.3f}, Pitch={avg_pitch:.3f}")
        else:
            print("Failed to detect face consistently for neutral position. Please try again.")
            cv2.destroyAllWindows()
            return False

        # --- Calibration Points Loop ---
        progress_bar_length = 300 # Smaller bar for points
        for i, (x_ratio, y_ratio) in enumerate(self.calibration_points):
            x_screen = int(x_ratio * self.screen_width)
            y_screen = int(y_ratio * self.screen_height)
            # Base image for this point
            point_base_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(point_base_img, (x_screen, y_screen), 30, (194, 165, 248), 2)
            cv2.circle(point_base_img, (x_screen, y_screen), 10, (62, 33, 22), -1)
            cv2.putText(point_base_img, f"Point {i+1}/{len(self.calibration_points)}: Look here",
                       (x_screen - 150 if x_screen > 150 else 20, y_screen - 50 if y_screen > 50 else 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Brief display before collection starts
            cv2.imshow("Calibration", point_base_img)
            print(f"Point {i+1}/{len(self.calibration_points)}: Turn your head to look...")
            if cv2.waitKey(1500) & 0xFF == ord('q'): # Wait 1.5s, allow quit
                 cv2.destroyAllWindows()
                 return False

            point_data = []
            start_time = time.time()
            collection_time = 1.5 # Shorter collection time per point

            while time.time() - start_time < collection_time:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                frame.flags.writeable = False # Perf opt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                frame.flags.writeable = True # Allow drawing

                # Show progress bar on the calibration screen
                elapsed = time.time() - start_time
                progress = int((elapsed / collection_time) * progress_bar_length)
                progress_img = point_base_img.copy()
                bar_y = y_screen + 50 if y_screen < self.screen_height - 70 else y_screen - 50
                cv2.rectangle(progress_img, (x_screen - progress_bar_length//2, bar_y),
                             (x_screen - progress_bar_length//2 + progress, bar_y + 20), (62, 33, 22), -1)
                cv2.rectangle(progress_img, (x_screen - progress_bar_length//2, bar_y),
                             (x_screen + progress_bar_length//2, bar_y + 20), (194, 165, 248), 2)
                cv2.imshow("Calibration", progress_img)


                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    try:
                        _, _, head_yaw, head_pitch = self._get_head_pose(landmarks, frame)
                        if self.reference_head_pose:
                            rel_yaw = head_yaw - self.reference_head_pose[2]
                            rel_pitch = head_pitch - self.reference_head_pose[3]
                            point_data.append((rel_yaw, rel_pitch))
                        self._draw_landmarks(frame, landmarks) # Draw landmarks on feed
                    except Exception as e:
                         print(f"Error getting head pose during point calibration: {e}")

                cv2.putText(frame, f"Looking at point {i+1}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     cv2.destroyAllWindows()
                     return False

            if len(point_data) > 3: # Need a few good readings
                avg_yaw = np.mean([p[0] for p in point_data])
                avg_pitch = np.mean([p[1] for p in point_data])
                self.calibration_data.append((avg_yaw, avg_pitch, x_ratio, y_ratio))
                print(f"Calibration point {i+1} recorded (RelYaw={avg_yaw:.3f}, RelPitch={avg_pitch:.3f})")
            else:
                print(f"Failed to detect face consistently for point {i+1}. Skipping.")


        cv2.destroyWindow("Calibration")
        cv2.destroyWindow("Camera Feed") # Explicitly destroy

        if len(self.calibration_data) >= 5: # Need enough points for regression
            self._compute_calibration_matrix()
            # Check if matrix computation succeeded
            if self.calibration_complete:
                print("Calibration complete!")
                # --- Reset Kalman filter state AFTER successful calibration ---
                self.kf.x = np.array([self.screen_width / 2, self.screen_height / 2, 0., 0.]).T # Reset position to center
                self.kf.P = np.eye(4) * 500. # Reset uncertainty
                self.last_time = time.time() # Reset time for dt calc
                # --- Reset click states after calibration ---
                self.left_wink_counter = 0
                self.right_wink_counter = 0
                self.last_left_click_time = 0
                self.last_right_click_time = 0
                self.last_left_wink_detect_time = 0
                return True
            else:
                print("Calibration matrix computation failed. Try again.")
                return False
        else:
            print(f"Not enough calibration data ({len(self.calibration_data)} points). Need at least 5. Try again.")
            self.calibration_complete = False
            return False

    # --- _compute_calibration_matrix remains the same ---
    def _compute_calibration_matrix(self):
        """Compute the transformation matrix using polynomial regression"""
        if not self.calibration_data:
            print("Error: No calibration data to compute matrix.")
            self.calibration_matrix = None
            self.calibration_complete = False
            return

        head_yaw = np.array([p[0] for p in self.calibration_data])
        head_pitch = np.array([p[1] for p in self.calibration_data])
        screen_x = np.array([p[2] for p in self.calibration_data])
        screen_y = np.array([p[3] for p in self.calibration_data])

        # [yaw, pitch, yaw^2, pitch^2, yaw*pitch, 1]
        feature_matrix = np.column_stack([
            head_yaw, head_pitch,
            head_yaw**2, head_pitch**2,
            head_yaw*head_pitch,
            np.ones(len(head_yaw))
        ])

        try:
            # Use a small epsilon for rcond to avoid warnings/errors on potentially ill-conditioned matrices
            x_mapping = np.linalg.lstsq(feature_matrix, screen_x, rcond=1e-10)[0]
            y_mapping = np.linalg.lstsq(feature_matrix, screen_y, rcond=1e-10)[0]
            self.calibration_matrix = (x_mapping, y_mapping)
            print("Calibration matrix computed.")
            self.calibration_complete = True # Set status only on success
        except np.linalg.LinAlgError as e:
            print(f"Error computing calibration matrix (LinAlgError): {e}. Calibration failed.")
            self.calibration_matrix = None
            self.calibration_complete = False
        except ValueError as e:
             print(f"Error computing calibration matrix (ValueError, likely input shape issue): {e}")
             print(f"Feature matrix shape: {feature_matrix.shape}, screen_x shape: {screen_x.shape}, screen_y shape: {screen_y.shape}")
             self.calibration_matrix = None
             self.calibration_complete = False


    # --- map_head_to_screen remains the same ---
    def map_head_to_screen(self, head_yaw, head_pitch):
        """Map head orientation to screen coordinates using calibration"""
        if not self.calibration_complete or self.calibration_matrix is None or self.reference_head_pose is None:
            # Fallback (shouldn't happen in normal operation after successful calibration)
            # Simple linear mapping around center if needed, though KF state persists
             return self.kf.x[0], self.kf.x[1] # Return current KF estimate? Or center?

        rel_yaw = head_yaw - self.reference_head_pose[2]
        rel_pitch = head_pitch - self.reference_head_pose[3]

        x_map, y_map = self.calibration_matrix
        features = np.array([
            rel_yaw, rel_pitch,
            rel_yaw**2, rel_pitch**2,
            rel_yaw*rel_pitch,
            1.0
        ])

        screen_x_ratio = np.dot(features, x_map)
        screen_y_ratio = np.dot(features, y_map)

        # Map to screen coordinates - NO CLAMPING HERE, Kalman/final step handles it
        screen_x = screen_x_ratio * self.screen_width
        screen_y = screen_y_ratio * self.screen_height

        return screen_x, screen_y

    # --- _draw_landmarks remains the same ---
    def _draw_landmarks(self, frame, landmarks):
        """Draw landmarks on the frame for visual feedback"""
        if not landmarks: return # Check if landmarks list is empty
        h, w = frame.shape[:2]
        # Use the stored state for color, not a fresh calculation
        left_color = (0, 255, 255) if self.left_eye_closed else (255, 0, 0) # Cyan if closed, Blue if open
        right_color = (0, 255, 255) if self.right_eye_closed else (255, 0, 0)

        # Draw eye landmarks used for EAR
        for idx in self.LEFT_EYE_V + self.LEFT_EYE_H:
             if idx < len(landmarks): cv2.circle(frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, left_color, -1)
        for idx in self.RIGHT_EYE_V + self.RIGHT_EYE_H:
             if idx < len(landmarks): cv2.circle(frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, right_color, -1)

        # Draw nose tip
        if 4 < len(landmarks):
            nose_tip = landmarks[4]
            cv2.circle(frame, (int(nose_tip.x * w), int(nose_tip.y * h)), 3, (0, 0, 255), -1) # Red


    def start_tracking(self):
        """Start the head tracking and cursor control"""
        cap = None
        preferred_indices = [0, 1, 2]
        other_indices = [i for i in range(3, 10) if i not in preferred_indices]
        
        # Try to open camera without platform-specific backend first
        for i in preferred_indices + other_indices:
            print(f"Trying camera index: {i}")
            cap = cv2.VideoCapture(i)  # Remove cv2.CAP_DSHOW which is Windows-specific
            if cap.isOpened():
                print(f"Camera {i} opened. Setting resolution...")
                # Set resolution - some Macs may not support arbitrary resolutions
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Verify resolution - Mac may use different resolution
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if actual_width > 0 and actual_height > 0:
                    print(f"Camera {i} resolution: {int(actual_width)}x{int(actual_height)}")
                    # Test reading a frame to ensure camera works
                    ret, test_frame = cap.read()
                    if ret:
                        break  # Successfully opened camera
                    else:
                        print(f"Camera {i} opened but cannot read frames. Trying next...")
                        cap.release()
                        cap = None
                else:
                    print(f"Camera {i} opened but failed to get resolution. Trying next...")
                    cap.release()
                    cap = None
            else:
                print(f"Failed to open camera index {i}.")
                if cap:
                    cap.release()
                cap = None

        if not cap or not cap.isOpened():
            print("Error: Could not open any suitable webcam.")
            return

        # Create window for main feed *before* calibration might use it
        cv2.namedWindow('Head Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Head Tracking', 640, 480)
        cv2.moveWindow('Head Tracking', 50, 50)

        # Run calibration
        if not self.run_calibration(cap):
            print("Calibration failed or aborted. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

        # --- Main Loop ---
        frame_count = 0
        fps = 0
        fps_update_rate = 15 # Update FPS display every N frames
        start_time_main = time.time()
        self.last_time = time.time() # Re-initialize for first dt calculation in the loop

        while self.running:
            # --- Calculate dt ---
            current_time = time.time()
            dt = current_time - self.last_time
            # Prevent zero or negative dt, and unusually large dt (e.g., after pause)
            if dt <= 1e-6:
                dt = 1/30.0 # Assume 30fps if dt is too small
            elif dt > 0.5: # If paused or major lag
                 dt = 1/30.0 # Reset to reasonable value to avoid filter jump
            self.last_time = current_time
            # -------------------------

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                time.sleep(0.1) # Wait a bit before retrying
                continue

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            # Make frame non-writeable to pass by reference for performance
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                results = self.face_mesh.process(rgb_frame)
            except Exception as e:
                 print(f"Error during face mesh processing: {e}")
                 frame.flags.writeable = True # Make writeable again before drawing/continuing
                 continue # Skip this frame if processing fails

            # Make frame writeable again for drawing overlays
            frame.flags.writeable = True

            left_ear, right_ear = 0.3, 0.3 # Default values if no face detected

            if results.multi_face_landmarks and self.calibration_complete:
                landmarks = results.multi_face_landmarks[0].landmark

                try:
                    # 1. Get Head Pose
                    _, _, head_yaw, head_pitch = self._get_head_pose(landmarks, frame)

                    # 2. Map to Screen Coordinates (Measurement for Kalman Filter)
                    measured_x, measured_y = self.map_head_to_screen(head_yaw, head_pitch)
                    measurement = np.array([[measured_x], [measured_y]])

                    # --- Kalman Filter Step ---
                    # Update State Transition Matrix F with current dt
                    self.kf.F[0, 2] = dt
                    self.kf.F[1, 3] = dt

                    # *** FIXED: Update Process Noise Q with dt ***
                    process_noise_std = 1.0 # TUNABLE: Std dev of acceleration noise (pixels/sec^2) - Adjust this!
                    # Calculate the 2x2 Q block for a single dimension (pos, vel)
                    # Represents noise introduced by acceleration:
                    # var = process_noise_std**2
                    # Q_block = [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * var
                    q_block_half = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_std**2)

                    # Create the 4x4 Q matrix assuming independence between x and y noise
                    self.kf.Q = np.zeros((4, 4))
                    # Block for x, vx
                    self.kf.Q[0,0] = q_block_half[0,0]
                    self.kf.Q[0,2] = q_block_half[0,1]
                    self.kf.Q[2,0] = q_block_half[1,0]
                    self.kf.Q[2,2] = q_block_half[1,1]
                    # Block for y, vy
                    self.kf.Q[1,1] = q_block_half[0,0]
                    self.kf.Q[1,3] = q_block_half[0,1]
                    self.kf.Q[3,1] = q_block_half[1,0]
                    self.kf.Q[3,3] = q_block_half[1,1]


                    # Predict next state
                    self.kf.predict()

                    # Update state based on measurement
                    self.kf.update(measurement)
                    # -----------------------------

                    # 3. Get Filtered State
                    filtered_x = self.kf.x[0]
                    filtered_y = self.kf.x[1]

                    # 4. Clamp Filtered Coordinates to Screen Bounds
                    smooth_x = max(0, min(self.screen_width - 1, filtered_x))
                    smooth_y = max(0, min(self.screen_height - 1, filtered_y))

                    # 5. Move Mouse Cursor *** FIXED duration=0.0 ***
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.0) # Instantaneous move

                    # 6. Detect Winks (and handle clicks/double clicks)
                    left_ear, right_ear = self._detect_winks(landmarks)

                    # 7. Visualize
                    self._draw_landmarks(frame, landmarks)
                    cv2.putText(frame, f"Cursor: ({int(smooth_x)}, {int(smooth_y)})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(frame, f"L_EAR: {left_ear:.2f}", (10, frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(frame, f"R_EAR: {right_ear:.2f}", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                except Exception as e:
                    import traceback
                    print(f"Error in main tracking loop: {e}")
                    traceback.print_exc()
                    cv2.putText(frame, "Processing Error", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif not self.calibration_complete:
                 # Should not happen if calibration succeeded, but handle defensively
                 cv2.putText(frame, "Calibration Required!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 time.sleep(0.1) # Prevent overly fast looping if stuck here
            else: # Face not detected (but calibration is complete)
                # Option 1: Let Kalman filter predict (coast)
                self.kf.predict()
                pred_x = max(0, min(self.screen_width - 1, self.kf.x[0])) # Clamp predicted pos
                pred_y = max(0, min(self.screen_height - 1, self.kf.x[1]))
                # pyautogui.moveTo(pred_x, pred_y, duration=0.0) # Uncomment to keep moving smoothly

                # Option 2: Freeze mouse (do nothing with pyautogui)

                cv2.putText(frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Reset wink states if face is lost to prevent accidental clicks on reappearance
                self.left_wink_counter = 0
                self.right_wink_counter = 0
                self.left_eye_closed = False
                self.right_eye_closed = False
                self.last_left_wink_detect_time = 0 # Reset double click state

            # --- Calculate and display FPS ---
            frame_count += 1
            if frame_count % fps_update_rate == 0: # Update FPS calculation less frequently
                 now = time.time()
                 duration = now - start_time_main
                 if duration > 0:
                     fps = frame_count / duration
                 # Reset counter and timer periodically to get rolling average FPS
                 # if frame_count > 1000: # Optional reset
                 #      frame_count = 0
                 #      start_time_main = time.time()

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "Q: Quit, C: Recalibrate", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Head Tracking', frame)

            # --- Handle User Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                print("Recalibrating...")
                # Calibration handles resetting KF state etc. on success
                if not self.run_calibration(cap):
                    print("Recalibration failed or aborted. Exiting.")
                    break
                else:
                    # Reset FPS counter after successful recalibration
                    frame_count = 0
                    start_time_main = time.time()
                    self.last_time = time.time() # Reset dt timer

        # --- Cleanup ---
        print("Releasing camera and closing windows.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = HeadGazeTracker()
    tracker.start_tracking()
    
