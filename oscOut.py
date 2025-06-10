from pythonosc import udp_client
import cv2 as cv
import numpy as np
import math
from faceTracking import *

# OSC setup
IP = "127.0.0.1"  # Localhost
PORT = 8000
client = udp_client.SimpleUDPClient(IP, PORT)

# Initialize camera and face mesh
face_mesh = init_face_mesh()
cam = open_camera()

print("Press 'q' to begin OSC transmission. Press 'q' again to stop.")

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Failed to read frame from camera")
        continue

    # Get face values
    mouth_open, head_tilt, brow_raise = get_face_values(frame, face_mesh)

    if mouth_open is not None:
        # Send OSC messages
        client.send_message("/mouthopen", float(mouth_open))
        client.send_message("/headtilt", float(head_tilt))
        client.send_message("/browraise", float(brow_raise))    

        print(f"Mouth Open: {mouth_open}, Head Tilt: {head_tilt}, Brow Raise: {brow_raise}")
    
    cv.putText(frame, "Press 'q' to Begin OSC Transmission. Press 'q' again to Stop", 
            (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (91, 95, 250), 2)
    cv.putText(frame, f"Mouth Open: {mouth_open}",
            (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 250), 2)
    cv.putText(frame, f"Head Tilt: {head_tilt}",
            (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 250), 2)
    cv.putText(frame, f"Brow Raise: {brow_raise}",
            (10, 160), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 250), 2)
    
    # Show the frame with overlay
    cv.imshow("OSC Face Tracking", frame)
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()