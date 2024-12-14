import streamlit as st
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_objects(image, low_thresh, high_thresh, min_size):
    edges = cv2.Canny(image, low_thresh, high_thresh)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_size:
            detected_objects.append(cnt)
    return detected_objects

def draw_contours(image, contours):
    output_image = image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
    return output_image

def main():
    st.title("Object Counting with OpenCV")
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    default_image_path = "example.jpg"
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    else:
        image = Image.open(default_image_path).convert('RGB')
    image_array = np.array(image)
    st.image(image, caption="Input Image", use_container_width=True)
    blurred_image = preprocess_image(image_array)
    low_thresh = st.sidebar.slider("Low Threshold for Canny", 0, 255, 50)
    high_thresh = st.sidebar.slider("High Threshold for Canny", 0, 255, 150)
    min_size = st.sidebar.slider("Minimum Object Size", 10, 500, 100)
    contours = detect_objects(blurred_image, low_thresh, high_thresh, min_size)
    st.write(f"Number of Objects Detected: {len(contours)}")
    output_image = draw_contours(image_array, contours)
    st.image(output_image, caption="Detected Objects", use_container_width=True)

if __name__ == "__main__":
    main()
