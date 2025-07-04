# Basketball Tracker (Prototype)

This is a simple real-time basketball tracking tool built with Python, OpenCV, and YOLOv8. It detects a basketball using both AI and color filtering, shows a fading trail of its movement, and simulates a virtual hoop to count shots made.

## Features

- Tracks a basketball using your webcam
- AI-based detection (YOLOv8) with fallback to yellow color tracking
- Fading trajectory trail for ball movement
- Virtual hoop area to count "shots made"
- Mirror view for a more natural webcam experience

## Getting Started

### Requirements

- Python 3.8+
- `opencv-python`
- `numpy`
- `ultralytics`

### Install dependencies

```bash
pip install opencv-python numpy ultralytics
