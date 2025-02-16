from google.colab import drive
drive.mount('/content/drive')

!rm -rf ~/.cache
!nvidia-smi

#install the required libraries:
!pip install ultralytics
!pip install matplotlib
!pip install opencv-python-headless
!pip install twilio

#Import necessary modules:

from ultralytics import YOLO
import cv2
import time
from IPython.display import display, Image
import matplotlib.pyplot as plt
from twilio.rest import Client

#Load and prepare the dataset from Google Drive:

import os

dataset_dir = '/content/drive/MyDrive/dataset_directory'  # Change this to your dataset directory path
data_yaml_path = os.path.join(dataset_dir, 'data.yaml')

# Verify the dataset
print(f"Dataset directory: {dataset_dir}")
print(f"Data YAML path: {data_yaml_path}")

#Training the model:

model = YOLO('yolov8m.pt')

train_results = model.train(
    data=data_yaml_path,
    epochs=40,
    batch=16,
    imgsz=640,
    device='cuda',
    workers=2,
    optimizer='Adam',
    lr0=0.001,
    momentum=0.937,
    weight_decay=0.0005,
    cache=False
)

print(train_results)

#Retrain the model with corrected parameters:


train_results = model.train(
    data=data_yaml_path,
    epochs=100,
    batch=16,
    imgsz=640,
    device='cuda',
    workers=2,
    optimizer='Adam',
    lr0=0.001,
    momentum=0.937,
    weight_decay=0.0005,
    cache=False
)

print(train_results)


#Display training metrics over epochs:


metrics = train_results
precision = metrics['metrics/precision(B)']
recall = metrics['metrics/recall(B)']
f1_score = metrics['metrics/f1(B)']
accuracy = metrics.get('metrics/mAP50-95(B)', None)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
if accuracy is not None:
    print(f"Accuracy (mAP50-95): {accuracy:.2f}")
else:
    print("Accuracy (mAP50-95) not found in results.")

# Plot the metrics
epochs = range(1, len(precision) + 1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, precision, 'b', label='Precision')
plt.title('Precision over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, recall, 'r', label='Recall')
plt.title('Recall over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, f1_score, 'g', label='F1 Score')
plt.title('F1 Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(2, 2, 4)
if accuracy is not None:
    plt.plot(epochs, accuracy, 'm', label='Accuracy')
    plt.title('Accuracy (mAP50-95) over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (mAP50-95)')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Accuracy (mAP50-95) not found', horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.show()

#Verify the image:


model = YOLO('/content/runs/detect/train2/weights/best.pt')
results = model('/content/image.png', show=True) #path of the image 
for result in results:
    result.show()


#Detect smoke and fire from an image and convert video to frames:


video_path = '/content/video.mp4'  # Path of the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Perform detection on the current frame
        results = model(frame)
        annotated_frame = results[0].plot()
        # Check if fire or smoke is detected
        detections = results[0].pandas().xyxy
        if not detections.empty:
            # Save the annotated frame as an image
            image_path = '/content/fire.15.png'
            cv2.imwrite(image_path, annotated_frame)
            # Convert the annotated frame to a format suitable for IPython.display.Image
            _, encoded_img = cv2.imencode('.jpg', annotated_frame)
            display(Image(data=encoded_img.tobytes()))
        # Introduce a small delay to simulate video playback
        time.sleep(0.05)  # Adjust the delay as needed
    cap.release()

#Send SMS alert using Twilio:

account_sid = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'#replace with your twilio acct sid 
auth_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'# replace with your twilio token number
client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='+1***********'#replace with your twilio number 
  body='Alert!! Fire/Smoke is Detected',
  to='+91***********'#repalce with the client/user number 
)

print(message.sid)
print("Alert SMS was sent successfully.")
