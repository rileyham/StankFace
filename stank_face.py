# Stank Face - Python Starter Code
# A facial gesture-controlled audio instrument

import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time

class StankFace:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize pygame for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Audio parameters
        self.base_frequency = 440.0  # A4
        self.volume = 0.5
        self.is_playing = False
        
        # Face landmark indices (MediaPipe specific)
        # These are key landmarks we'll track
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        
        # Gesture values (normalized 0-1)
        self.eyebrow_height = 0.5
        self.mouth_openness = 0.0
        self.mouth_width = 0.5
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def extract_gestures(self, landmarks):
        """Extract meaningful gestures from face landmarks"""
        if not landmarks:
            return
            
        # Convert landmarks to list for easier access
        landmark_points = landmarks.landmark
        
        # Calculate eye openness (simplified - using top/bottom distance)
        left_eye_openness = self.calculate_distance(
            landmark_points[self.LEFT_EYE_TOP], 
            landmark_points[self.LEFT_EYE_BOTTOM]
        )
        right_eye_openness = self.calculate_distance(
            landmark_points[self.RIGHT_EYE_TOP], 
            landmark_points[self.RIGHT_EYE_BOTTOM]
        )
        
        # Average eye openness (rough approximation of eyebrow position)
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
        self.eyebrow_height = min(1.0, max(0.0, avg_eye_openness * 20))  # Scale factor
        
        # Calculate mouth openness
        mouth_openness = self.calculate_distance(
            landmark_points[self.MOUTH_TOP],
            landmark_points[self.MOUTH_BOTTOM]
        )
        self.mouth_openness = min(1.0, max(0.0, mouth_openness * 50))  # Scale factor
        
        # Calculate mouth width (smile detection)
        mouth_width = self.calculate_distance(
            landmark_points[self.MOUTH_LEFT],
            landmark_points[self.MOUTH_RIGHT]
        )
        self.mouth_width = min(1.0, max(0.0, mouth_width * 20))  # Scale factor
    
    def generate_audio_sample(self, frequency, duration_ms, sample_rate=22050):
        """Generate a simple sine wave audio sample"""
        frames = int(duration_ms * sample_rate / 1000)
        arr = np.zeros((frames, 2))  # Stereo
        
        for i in range(frames):
            wave = np.sin(frequency * 2 * np.pi * i / sample_rate)
            arr[i][0] = wave * self.volume * 32767  # Left channel
            arr[i][1] = wave * self.volume * 32767  # Right channel
        
        return arr.astype(np.int16)
    
    def update_audio(self):
        """Update audio parameters based on facial gestures"""
        # Map gestures to audio parameters
        
        # Eyebrow height controls frequency (pitch bend)
        frequency_modifier = 0.5 + (self.eyebrow_height * 0.5)  # 0.5x to 1.0x
        current_frequency = self.base_frequency * frequency_modifier
        
        # Mouth openness controls volume
        self.volume = self.mouth_openness * 0.8  # Max volume of 0.8
        
        # Mouth width could control other parameters (filter, effects, etc.)
        # For now, we'll just print it
        
        # Only play sound if mouth is somewhat open
        should_play = self.mouth_openness > 0.1
        
        if should_play and not self.is_playing:
            # Start playing sound
            audio_sample = self.generate_audio_sample(current_frequency, 100)  # 100ms sample
            sound = pygame.sndarray.make_sound(audio_sample)
            sound.play(-1)  # Loop indefinitely
            self.is_playing = True
        elif not should_play and self.is_playing:
            # Stop playing sound
            pygame.mixer.stop()
            self.is_playing = False
        elif should_play and self.is_playing:
            # Update playing sound (regenerate with new parameters)
            pygame.mixer.stop()
            audio_sample = self.generate_audio_sample(current_frequency, 100)
            sound = pygame.sndarray.make_sound(audio_sample)
            sound.play(-1)
    
    def run(self):
        """Main application loop"""
        print("Stank Face is running! Press 'q' to quit.")
        print("Instructions:")
        print("- Raise eyebrows to increase pitch")
        print("- Open mouth to control volume")
        print("- Smile width affects... (future parameter)")
        
        while True:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            # Draw face landmarks and extract gestures
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract gestures from landmarks
                    self.extract_gestures(face_landmarks)
                    
                    # Draw landmarks on frame (optional - for debugging)
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        face_landmarks, 
                        self.mp_face_mesh.FACEMESH_CONTOURS
                    )
                    
                    # Update audio based on gestures
                    self.update_audio()
            
            # Display gesture values on screen
            cv2.putText(frame, f"Eyebrow: {self.eyebrow_height:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouth Open: {self.mouth_openness:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouth Width: {self.mouth_width:.2f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Stank Face', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    # Create and run Stank Face
    stank_face = StankFace()
    stank_face.run()