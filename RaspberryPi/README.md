# Person Detection with Virtual Fence

This project demonstrates how to perform person detection using a virtual fence to track the number of people entering and leaving a designated area. It utilizes the MobileNet SSD model for efficient person detection on a Raspberry Pi.

## Prerequisites

- Docker installed on your system
- Camera connected to your Raspberry Pi

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/person-detection.git
   cd person-detection
   ```

2. **Build the Docker image:**

   ```bash
   docker build -t person-detection .
   ```

3. **Run the Docker container:**
   ```bash
   docker run --device /dev/video0:/dev/video0 person-detection
   ```
   Make sure to replace `/dev/video0` with the appropriate device path for your camera if it differs.

The person detection script will start running inside the container, and you should see the output displayed in the console.
