# YOLOv5-Based Automated Sewer Pipe Anomaly Detection with Timestamped Logging

## Project Overview

Large sewage pipes are critical for transporting drainage waste. Regular inspection of these pipes is essential to identify defects that can obstruct sewage flow. Traditionally, small robots equipped with cameras capture inspection videos inside pipes, and certified inspectors manually review these videos to locate and classify defects.

This manual inspection process is costly, time-consuming, and prone to human error due to inspector fatigue. To overcome these challenges, this project automates sewer pipe inspection using deep learning techniques.

---

## Motivation

Current automated methods typically require converting inspection videos into images before detecting and classifying anomalies, which is inefficient and disconnected from the video timeline.

This project presents a streamlined, video-based anomaly detection approach using the YOLOv5 algorithm that:

- Detects anomalies directly from sewer inspection videos.
- Logs the timing (video timestamp) and class of each detected anomaly.
- Provides corresponding frames for visual verification.

This enables inspectors to quickly locate defects within videos based on the robot's speed and detection timestamps, facilitating faster and more accurate decision-making.

---

## Features

- **Real-time detection of sewer pipe anomalies** with bounding boxes using a custom-trained YOLOv5 model.
- **Video timestamped logging** of detected anomalies into a CSV file, capturing:
  - Timestamp (HH:MM:SS.mmm)
  - Anomaly class
  - Detection confidence
- Supports multiple target anomaly classes specified by the user.
- Saves frames where anomalies are detected for further inspection.
- Lightweight and faster training compared to other models like Faster R-CNN.

---

## Model Performance

- Trained on 160 images in approximately **12 minutes**.
- Achieved highest average validation accuracy of **0.78**.
- Precision: **0.40**
- Recall: **0.85**
- Faster training time and comparable accuracy compared to Faster R-CNN (which takes ~2 hours).

---

## Setup

### Clone and Install Dependencies

<pre>git clone https://github.com/Cvakapoor/yolov5
pip install -r yolov5/requirements.txt
cd yolov5</pre>
