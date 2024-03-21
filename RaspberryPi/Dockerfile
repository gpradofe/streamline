FROM python:3.8-slim-buster

# Install OpenCV dependencies, OpenGL dependencies, and wget
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install opencv-python numpy

# Create a directory for the app
WORKDIR /app

# Download the MobileNet SSD model files
RUN wget -q https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -O MobileNetSSD_deploy.prototxt.txt && \
    wget -q https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -O MobileNetSSD_deploy.caffemodel

# Copy the script to the container
COPY app/main.py .

# Set the command to run the script
CMD ["python", "main.py"]