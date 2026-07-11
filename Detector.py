import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def detect_objects_in_frame(frame):
    """
    Takes a raw camera frame (numpy array from OpenCV).
    Returns (annotated_frame, list_of_detected_labels)
    """
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()  # draws boxes on the frame

    labels = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        labels.append(model.names[class_id])

    return annotated_frame, labels