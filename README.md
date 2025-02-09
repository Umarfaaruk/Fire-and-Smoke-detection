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

## Introduction
This project uses the YOLO model to detect fire and smoke in images and videos. The code is designed to train the model using a dataset stored in Google Drive and then use the trained model to detect fire and smoke. When fire or smoke is detected, an SMS alert is sent to the user.

## Installation
Follow these steps to install the necessary dependencies and set up the project:

1. Mount Google Drive.
2. Clear cached files.
3. Display GPU configuration.
4. Install required libraries.

Make sure you have the necessary permissions to access Google Drive and install the required libraries.

## Dataset
The dataset used for training and testing the model is stored in Google Drive. Ensure that you have your dataset organized with the necessary images and annotations. The `data.yaml` file should be included in the dataset directory, specifying the paths to the training, validation, and test sets.

Ensure the `data.yaml` file correctly points to the image and label directories.

## Training the Model
To train the YOLO model, use the provided code to load and train the model using the dataset from Google Drive. Adjust the parameters such as the number of epochs, batch size, image size, and optimizer as needed for your specific requirements.

### Training Parameters
- **Epochs**: Number of times the entire dataset is passed through the model.
- **Batch Size**: Number of samples processed before the model is updated.
- **Image Size**: Dimensions of the input images.
- **Optimizer**: Algorithm used to optimize the model's weights.
- **Learning Rate**: Step size for updating the model's weights.

## Testing the Model
After training, you can test the model using the provided code. Evaluate the model's performance on the test dataset and adjust the parameters to fine-tune the model's performance.

### Testing Parameters
- **Test Dataset**: Subset of the dataset used to evaluate the model's performance.
- **Metrics**: Precision, recall, F1 score, and accuracy used to evaluate the model.

## Detection on Video
Use the provided code to perform detection on a video and convert the video into frames. The annotated frames will be saved, and detections will be displayed.

### Video Processing
- **Video Path**: Path to the input video file.
- **Frame Extraction**: Convert video frames into images for detection.
- **Detection Results**: Annotated frames showing detected fire and smoke.

## Sending Alerts
When fire or smoke is detected, an SMS alert is sent to the user using the Twilio API. Ensure you have your Twilio account SID and authentication token.

### Twilio Setup
- **Account SID**: Unique identifier for your Twilio account.
- **Authentication Token**: Secret token for authenticating API requests.
- **Phone Numbers**: Twilio phone number and recipient phone number for sending SMS alerts.

## Results
Visualize the training results and metrics such as precision, recall, F1 score, and accuracy over the epochs. Use the provided code to plot these metrics.

### Visualization
- **Precision**: Ratio of true positive detections to the total detected positives.
- **Recall**: Ratio of true positive detections to the total actual positives.
- **F1 Score**: Harmonic mean of precision and recall.
- **Accuracy**: Overall correctness of the model's predictions.

## Contributing
Feel free to contribute to this project by submitting issues and pull requests. For major changes, please open an issue first to discuss what you would like to change. Contributions are always welcome!

### How to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push the branch to your fork.
5. Open a pull request to the main repository.
