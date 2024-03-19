# People Counting with YOLOv4 on Raspberry Pi

This project implements a people counting system using YOLOv4 object detection on Raspberry Pi instances. The Raspberry Pis process video streams from connected cameras, detect people in the frames, and send the people count to a central server.

## Prerequisites

- Raspberry Pi with Docker installed
- Camera connected to the Raspberry Pi
- `yolov4.weights`, `yolov4.cfg`, and `coco.names` files (download from the official YOLO website)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/people-counting.git
   cd people-counting
   ```
