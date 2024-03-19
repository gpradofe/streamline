import cv2
import numpy as np
import time

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Set up the camera
camera = cv2.VideoCapture(0)  # Assumes the camera is connected to the first port (0)

# Initialize variables
people_count = 0
people_inside = 0
previous_detections = []

# Define the virtual fence coordinates (door area)
door_coordinates = [(200, 100), (400, 400)]  # Adjust coordinates according to your camera view

while True:
    # Read a frame from the camera
    ret, frame = camera.read()
    
    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    
    # Perform object detection
    net.setInput(blob)
    detections = net.forward()
    
    # Process the detections
    current_detections = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:  # Class ID 15 represents 'person'
                x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                current_detections.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Update people count based on detections crossing the virtual fence
    for detection in current_detections:
        x1, y1, x2, y2 = detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if door_coordinates[0][0] < center_x < door_coordinates[1][0] and \
           door_coordinates[0][1] < center_y < door_coordinates[1][1]:
            if detection not in previous_detections:
                # Person entered the room
                people_count += 1
                people_inside += 1
    
    for detection in previous_detections:
        if detection not in current_detections:
            # Person left the room
            people_count -= 1
            people_inside = max(0, people_inside - 1)
    
    previous_detections = current_detections
    
    # Display the frame with bounding boxes, people count, and people inside
    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"People Inside: {people_inside}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Delay to control the processing frame rate (optional)
    time.sleep(0.1)

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()