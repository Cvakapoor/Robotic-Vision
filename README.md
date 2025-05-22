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

### Mount Google Drive (for Google Colab users)

<pre>from google.colab import drive
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/ /mydrive</pre>

---

## Dataset Preparation

- Format your dataset in YOLO format with images and labels.
- Organize data into training and validation splits.
- Create a dataset.yaml file specifying paths and classes.

---

## Training

Train the model using:

<pre>python train.py --img 384 --batch 4 --epochs 200 --data training/dataset.yaml --cfg training/yolov5s.yaml --weights '' --name yolov5s_detect --cache --device 0</pre>

---

## Inference and Logging

Run detection on videos/images with timestamped anomaly logs:

<pre>python detect_with_logs.py \
  --weights runs/train/yolov5s_detect/weights/best.pt \
  --source inference/video/block.mp4 \
  --output inference/video_output \
  --target_class Crack,Block,Biological-Object \
  --csv_output inference/anomaly_log.csv</pre>
  
- Outputs a CSV log with timestamps and anomaly details.
- Saves frames with detected anomalies for review.

---

## Output

The CSV detection log format:

| Timestamp  | Class            | Confidence |
|------------|------------------|------------|
| 00:00:05.123 | Crack            | 0.857      |
| 00:00:07.456 | Biological-Object| 0.921      |
| ...        | ...              | ...        |

Detected frames are saved in the output directory alongside the video frames.
