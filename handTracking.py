# sample hand tracking using MediaPipe and OpenCV
# courtesy of https://blog.roboflow.com/what-is-mediapipe/

import cv2 as cv 
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

# Initialize the hands model
hands = mp_hands.Hands(
    static_image_mode = False, # Set to False for processing video frames
    max_num_hands = 2, # Maximum number of hands to detect
    min_detection_confidence = 0.5, # Minimum confidence for detection
)

cam = cv.VideoCapture(0) # Open the camera

if not cam.isOpened():
    raise Exception("Could ]not open camera")

while cam.isOpened():
    # Read a frame from the camera
    success, frame = cam.read()

    # if the frame is not available, skip this iteration
    if not success:
        print("Camera Frame not available")
        continue

    # Convert the frame to RGB (required by MediaPipe)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame to detect hands
    hands_detected = hands.process(frame)

    # convert the frame back to BGR for display (required by OpenCV)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # If hands are detected, draw the landmarks and connections on the frame
    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
        
    # Display the frame with hand landmarks
    cv.imshow("Show Video", frame)

    # Exit if 'q' is pressed
    if cv.waitKey(20) & 0xff == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
