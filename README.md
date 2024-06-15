# Headcount AI

Headcount AI is a computer vision project using OpenCV and deep learning to count people in a video stream. It employs a MobileNet SSD model for real-time object detection and a centroid tracking algorithm for counting and tracking individuals.

## Features

- **Real-time Object Detection**: Utilizes MobileNet SSD for detecting persons in video frames.
- **Centroid Tracking**: Tracks detected persons across frames using a centroid tracker.
- **Entry/Exit Counting**: Automatically counts people entering and exiting a defined area.
- **Performance Metrics**: Displays FPS, entry count, exit count, and current headcount.
- **Data Reset**: Option to reset counts and tracking state via a key press (`r` key).

## Requirements

- Python 3.x
- OpenCV
- NumPy
- imutils
- Caffe model files (`MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel`)

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Vatsalbirla317/headcount-ai.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

4. Use `q` to quit, `w` to pause/unpause the video, and `r` to reset the data.

