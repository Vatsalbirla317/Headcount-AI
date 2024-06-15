# Headcount-AI
HeadcountAI is a real-time people counting system using computer vision and a pre-trained MobileNetSSD model. It detects and tracks individuals, providing accurate entry and exit counts with real-time performance metrics. Features include lighting adjustment, pausing, and resetting counts. Ideal for surveillance and analytics.

## Features

- Real-time people detection and tracking
- Entry and exit counting
- Lighting adjustment for better detection accuracy
- Pausing and resuming the count
- Resetting the count data

## Requirements

- Python 3.6+
- OpenCV
- imutils
- numpy
- A webcam or video file for input

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/HeadcountAI.git
    cd HeadcountAI
    ```

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the MobileNetSSD model files and place them in the project directory:
    - `MobileNetSSD_deploy.prototxt`
    - `MobileNetSSD_deploy.caffemodel`

## Usage

1. Run the main script:
    ```bash
    python main.py
    ```

2. Use the following keys for control:
    - `q`: Quit the application
    - `w`: Pause/Resume the counting
    - `r`: Reset the count data

## File Structure

```
HeadcountAI/
├── centroidtracker.py
├── main.py
├── MobileNetSSD_deploy.prototxt
├── MobileNetSSD_deploy.caffemodel
├── requirements.txt
└── README.md
```
