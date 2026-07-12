import streamlit as st
import cv2
from Detector import detect_objects_in_frame
from logic import get_storage_location
from database import init_db, log_entry
import time

init_db()

st.set_page_config(page_title="Smart Object Sorter", page_icon="📦")
st.title("📦 Smart Object Sorter — Live")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()
result_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    last_logged = None

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not access webcam.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame, labels = detect_objects_in_frame(frame)

        frame_placeholder.image(annotated_frame, channels="RGB")

        if labels:
            top_label = labels[0]
            location = get_storage_location(top_label)
            result_placeholder.success(f"Detected: **{top_label}** → Keep in: **{location}**")

            if top_label != last_logged:
                log_entry(top_label, location)
                last_logged = top_label
        else:
            result_placeholder.info("No object detected")

        time.sleep(0.05)  # small delay to avoid maxing out CPU

    cap.release()
else:
    st.write("Check the box above to start the live camera feed.")
