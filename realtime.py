# Import necessary libraries
from ultralytics import YOLO
import cv2
import os

# use directly gta.pt file or Load your best.pt File here if you trained the model by train.py file .

model = YOLO("gta.pt")

# Initialize webcam for real-time detection
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam; replace with video file path if needed

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Create output directory for saving results
output_dir = "runs/detect"
os.makedirs(output_dir, exist_ok=True)

print("Press 'q' to exit the real-time detection.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform detection
    results = model(frame, conf=0.5)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
