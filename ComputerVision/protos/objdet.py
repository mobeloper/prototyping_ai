import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Streamlit App Title
st.title("Object Detection and Tracking with OpenCV")

# Sidebar Options
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4"])
operation_option = st.sidebar.selectbox(
    "Select an Operation", ["Object Detection (Haar Cascade)", "Object Tracking (Kalman Filter)"]
)

# Function to Convert PIL Image to OpenCV
@st.cache
def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

# Function to Detect Objects Using Haar Cascade
def haar_cascade_detection(image):
    # Load Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image

# Function to Implement Simple Kalman Filter Tracking
def kalman_filter_tracking():
    st.write("Kalman Filter demo for tracking is a real-time operation typically demonstrated with live video feeds.")
    st.write("Please try the OpenCV implementation for a detailed example.")

# Main Application Logic
if uploaded_file is not None:
    if uploaded_file.name.endswith("mp4"):
        st.video(uploaded_file)
        st.write("Video support for Haar Cascade detection and tracking is limited in this demo.")
    else:
        image = load_image(uploaded_file)

        if operation_option == "Object Detection (Haar Cascade)":
            st.subheader("Object Detection Using Haar Cascade")
            result_image = haar_cascade_detection(image.copy())
            st.image(result_image, caption="Detected Objects", use_column_width=True)

        elif operation_option == "Object Tracking (Kalman Filter)":
            st.subheader("Object Tracking Using Kalman Filter")
            kalman_filter_tracking()

else:
    st.write("Please upload an image or video to get started.")

