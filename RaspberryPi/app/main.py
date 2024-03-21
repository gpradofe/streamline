import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import logging
import time
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort  # Importing DeepSort


# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(level = logging.INFO, format = "[INFO] %(message)s")
logger = logging.getLogger(__name__)

model = YOLO('yolov8n.pt')


## Input Video
logger.info("Starting the video..")
cap = cv2.VideoCapture(1)

##for camera ip
# camera_ip = "Camera Url"
# logger.info("Starting the live stream..")
# cap = cv2.VideoCapture(camera_ip)
# time.sleep(1.0)



my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


#function for detect person coordinate
def get_person_coordinates(frame):
    """
    Extracts the coordinates of the person bounding boxes from the YOLO model predictions.

    Args:
        frame: Input frame for object detection.

    Returns:
        list: List of person bounding box coordinates in the format [x1, y1, x2, y2].
    """
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data.detach().cpu()
    px = pd.DataFrame(a).astype("float")

    list_corr = []
    for index, row in px.iterrows():
        x1 = row[0]
        y1 = row[1]
        x2 = row[2]
        y2 = row[3]
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list_corr.append([x1, y1, x2, y2])
    return list_corr


def people_counter():
    logger.info("Starting the video..")
    cap = cv2.VideoCapture(1)  # Assuming this is your desired video source

    # Initialize Deep SORT tracker
    tracker = DeepSort(max_age=30, nn_budget=70, override_track_class=None)

    fps = FPS().start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for consistent processing speed
        frame = cv2.resize(frame, (640, 360))

        # Detect people using YOLO model
        detections = get_person_coordinates(frame)  # Get detections as before

        # Format detections for Deep SORT (converting to Deep SORT expected format)
        formatted_detections = [([det[0], det[1], det[2] - det[0], det[3] - det[1]], 1.0, 'person') for det in detections]
        
        # Update Deep SORT tracker with detections
        tracks = tracker.update_tracks(formatted_detections, frame=frame)
        
        # Iterate over tracks and draw tracking info on frame
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue  # Skip unconfirmed or stale tracks
            
            bbox = track.to_ltrb()
            track_id = track.track_id
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, f"ID {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        # Display the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        fps.update()

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    people_counter()
    

## Apply threading also

# def start_people_counter():
#     t1 = threading.Thread(target=people_counter)
#     t1.start()


# if __name__ == "__main__":
#     start_people_counter()
