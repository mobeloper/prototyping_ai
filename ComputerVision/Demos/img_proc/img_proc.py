import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Basic Image Filtering and Processing with OpenCV")
st.write("""
### Features:
- **Filters**: Apply custom filters using OpenCV's `cv2.filter2D`.
- **Processing Techniques**:
  - Grayscale Conversion
  - Thresholding
  - Edge Detection (Canny)
  - Blurring
""")

st.sidebar.title("Settings")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    default_image_path = "example.jpg"
    image = Image.open(default_image_path)

image_array = np.array(image)
st.image(image, caption="Uploaded Image", use_container_width=True)

if st.sidebar.checkbox("Convert to Grayscale"):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    st.image(image_array, caption="Grayscale Image", use_container_width=True, channels="GRAY")

if st.sidebar.checkbox("Apply Thresholding"):
    threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
    _, thresholded_image = cv2.threshold(image_array, threshold_value, 255, cv2.THRESH_BINARY)
    st.image(thresholded_image, caption="Thresholded Image", use_container_width=True, channels="GRAY")

if st.sidebar.checkbox("Apply Edge Detection (Canny)"):
    low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 50)
    high_threshold = st.sidebar.slider("High Threshold", 0, 255, 150)
    edges = cv2.Canny(image_array, low_threshold, high_threshold)
    st.image(edges, caption="Edge Detection (Canny)", use_container_width=True, channels="GRAY")

if st.sidebar.checkbox("Apply Blurring"):
    blur_kernel = st.sidebar.slider("Kernel Size", 1, 20, 5)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred_image = cv2.GaussianBlur(image_array, (blur_kernel, blur_kernel), 0)
    st.image(blurred_image, caption="Blurred Image", use_container_width=True)

if st.sidebar.checkbox("Apply Custom Filter"):
    st.sidebar.write("Custom Kernel Configuration")
    kernel_size = st.sidebar.slider("Kernel Size (NxN)", 3, 11, 3, step=2)
    kernel_values = []
    for i in range(kernel_size):
        row = st.sidebar.text_input(f"Row {i+1} values (comma-separated)", "1,0,-1")
        kernel_values.append([int(x) for x in row.split(",")])

    kernel = np.array(kernel_values, dtype=np.float32)
    custom_filtered_image = cv2.filter2D(image_array, -1, kernel)
    st.image(custom_filtered_image, caption="Custom Filter Applied", use_container_width=True)

