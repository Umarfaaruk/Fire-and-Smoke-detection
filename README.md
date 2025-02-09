# Fire and Smoke Detection Using YOLOv11

This repository contains code for detecting fire and smoke in images and videos using the YOLOv11 model. The dataset is stored in Google Drive, and the trained model alerts the user by sending an SMS when fire or smoke is detected.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Detection on Video](#detection-on-video)
- [Sending Alerts](#sending-alerts)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project uses the YOLO model to detect fire and smoke in images and videos. The code is designed to train the model using a dataset stored in Google Drive and then use the trained model to detect fire and smoke. When fire or smoke is detected, an SMS alert is sent to the user.

## Installation
Follow these steps to install the necessary dependencies and set up the project:

```bash
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clear cached files
!rm -rf ~/.cache    

# Display GPU configuration
!nvidia-smi

# Install required libraries
!pip install ultralytics matplotlib opencv-python-headless twilio

  
