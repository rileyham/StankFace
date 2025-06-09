# This file does face tracking using MediaPipe and OpenCV.
# modified from the sample code found on https://www.assemblyai.com/blog/mediapipe-for-dummies?ref=blog.roboflow.com
# adjusted to work with live video feed

import cv2 as cv
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Face landmark indices for specific features
UPPER_LIP = 13
LOWER_LIP = 14
TOP_OF_HEAD = 10
BOTTOM_OF_HEAD = 152
LEFT_EYE_CORNER = 133
RIGHT_EYE_CORNER = 362
TOP_OF_RIGHT_BROW = 296
TOP_OF_LEFT_BROW = 66

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

        landmark_points = face_detected.multi_face_landmarks[0].landmark
        for id, landmark in enumerate(landmark_points):
            # Get the landmark coordinates and display them
            if id == UPPER_LIP:
                upper_lip_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, upper_lip_location, 5, (200, 255, 0), -1)
            elif id == LOWER_LIP:
                lower_lip_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, lower_lip_location, 5, (200, 0, 255), -1)
            elif id == TOP_OF_HEAD:
                top_of_head_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, top_of_head_location, 5, (255, 0, 0), -1)
            elif id == BOTTOM_OF_HEAD:
                bottom_of_head_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, bottom_of_head_location, 5, (0, 255, 255), -1)
            elif id == LEFT_EYE_CORNER:
                LEFT_EYE_CORNER_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, LEFT_EYE_CORNER_location, 5, (255, 255, 0), -1)
            elif id == RIGHT_EYE_CORNER:
                RIGHT_EYE_CORNER_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, RIGHT_EYE_CORNER_location, 5, (255, 0, 255), -1)
            elif id == TOP_OF_LEFT_BROW:
                top_of_left_brow_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, top_of_left_brow_location, 5, (255, 255, 0), -1)
            elif id == TOP_OF_RIGHT_BROW:
                top_of_right_brow_location = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv.circle(frame, top_of_right_brow_location, 5, (255, 0, 255), -1)

        # calculate head, mouth openness, and brow raise
        head_x_scale = (top_of_head_location[0] - bottom_of_head_location[0])
        head_y_scale = (top_of_head_location[1] - bottom_of_head_location[1])
        head_scale = math.sqrt((head_x_scale ** 2) + (head_y_scale ** 2))  

        mouth_openness_x = upper_lip_location[0] - lower_lip_location[0]
        mouth_openness_y = upper_lip_location[1] - lower_lip_location[1]
        mouth_openness_raw = math.sqrt((mouth_openness_x ** 2) + (mouth_openness_y ** 2))

        brow_raise_x = ((top_of_left_brow_location[0] - LEFT_EYE_CORNER_location[0]) + (top_of_right_brow_location[0] - RIGHT_EYE_CORNER_location[0])) / 2
        brow_raise_y = ((top_of_left_brow_location[1] - LEFT_EYE_CORNER_location[1]) + (top_of_right_brow_location[1] - RIGHT_EYE_CORNER_location[1])) / 2
        brow_raise_raw = math.sqrt((brow_raise_x ** 2) + (brow_raise_y ** 2))

        # draw eye and brow average locations
        cv.circle(frame, (int((top_of_left_brow_location[0] + top_of_right_brow_location[0])/2), int((top_of_left_brow_location[1] + top_of_right_brow_location[1])/2)), 5, (200, 0, 0), -1)
        cv.circle(frame, (int((LEFT_EYE_CORNER_location[0] + RIGHT_EYE_CORNER_location[0])/2), int((LEFT_EYE_CORNER_location[1] + RIGHT_EYE_CORNER_location[1])/2)), 5, (200, 0, 0), -1)
    

        # Normalize mouth openness based on head scale
        if mouth_openness_raw == 0:
            normalized_mouth_openness = 0
        else:
            normalized_mouth_openness = 3 * (mouth_openness_raw / head_scale) # Scale factor of 3 for a roughly 0-1 range
            normalized_mouth_openness = min(max(normalized_mouth_openness, 0), 1)  # Clamp to [0, 1]
            normalized_mouth_openness = round(normalized_mouth_openness, 2)  # Round to 2 decimal places

        # Normalize brow raise based on head scale
        if brow_raise_raw == 0:
            normalized_brow_raise = 0
        else:
            normalized_brow_raise = (abs(brow_raise_raw / head_scale) * 25) - 3.5 # Scale factor of 25 for a roughly 0-1 range
            normalized_brow_raise = min(max(normalized_brow_raise, 0), 1) # Clamp to [0, 1]
            normalized_brow_raise = round(normalized_brow_raise, 2)  # Round to 2 decimal places

        # calculate head tilt
        head_tilt = (top_of_head_location[0] - bottom_of_head_location[0]) / 300  # Scale factor of 300 for a roughly 0-1 range
        head_tilt = min(max(head_tilt, -1), 1)  # Clamp to [-1, 1]
        head_tilt = (head_tilt / 2) + 0.5  # Normalize to [0, 1]
        head_tilt = round(head_tilt, 2)  # Round to 2 decimal places
    
        # display the frame with face mesh landmarks and variable text
        cv.putText(frame, "Press 'q' to exit", 
                (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (91, 95, 250), 2)
        cv.putText(frame, f"Normalized Mouth Open: {normalized_mouth_openness}",
                (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 250), 2)
        cv.putText(frame, f"Head Tilt: {head_tilt}",
                (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 250), 2)
        cv.putText(frame, f"Brow Raise Raw: {brow_raise_raw}",
                (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 250), 2)
        cv.putText(frame, f"Normalized Brow Raise: {normalized_brow_raise}",
                (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 250), 2)
        # cv.putText(frame, f"Mouth Open: {mouth_openness_raw}",
        #             (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 250), 2)
        # cv.putText(frame, f"Head Scale: {head_scale}",
        #             (10, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 250), 2)
    
    cv.imshow("Face Mesh", frame)

    # Exit if 'q' is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()