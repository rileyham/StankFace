# This file does face tracking using MediaPipe and OpenCV.
# modified from the sample code found on https://www.assemblyai.com/blog/mediapipe-for-dummies?ref=blog.roboflow.com
# adjusted to work with live video feed

import cv2 as cv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hollistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# Initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Set to False for processing video frames
    max_num_faces=1,  # Maximum number of faces to detect
    refine_landmarks=True,  # Refine landmarks for better accuracy
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5  # Minimum confidence for tracking
)

cam = cv.VideoCapture(0)  # Open the camera

if not cam.isOpened():
    raise Exception("Could not open camera")

while cam.isOpened():
    # read a frame from the camera
    success, frame = cam.read()

    # if the frame is not available, skip this iteration
    if not success:
        print("Camera Frame not available")
        continue

    # Convert the frame to RGB (required by MediaPipe)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame to detect face landmarks
    face_detected = face_mesh.process(frame)

    # Convert the frame back to BGR for display (required by OpenCV)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # If face landmarks are detected, draw them on the frame
    if face_detected.multi_face_landmarks:
        for face_landmarks in face_detected.multi_face_landmarks:

            # Draw the face mesh landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            
            # Draw the face features (eyes, mouth, brows)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
    cv.imshow("Face Mesh", frame)

    # Exit if 'q' is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()