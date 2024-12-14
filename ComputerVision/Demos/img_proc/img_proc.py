

# pip install --upgrade streamlit
# streamlit version

#pip install streamlit opencv-python pillow


import streamlit as st
import cv2
import numpy as np
from PIL import Image



# Adjustable edge detection: Add functionality to toggle between different edge detection operators (Sobel, Laplacian, and Canny.) Include sliders for tuning Sobel's kernel size or Canny’s thresholds. Explore how different edge detection methods highlight image structures. Hint: Use cv2.Sobel and cv2.Laplacian for alternate edge detection methods.

# Add functionality to toggle between different types of blurring (Gaussian, median, and bilateral). Add sliders to adjust blurring strength. Explore how the blurring effects change as different methods are used. Hint: Use cv2.GaussianBlur, cv2.medianBlur, and cv2.bilateralFilter.



st.title("Basic Image Filtering and Processing with OpenCV")
st.write("""
### Features:
- **Filters**: Apply custom filters using OpenCV's `cv2.filter2D`.
- **Processing Techniques**:
  - Grayscale Conversion
  - Thresholding
  - Edge Detection (Canny)
  - Edge Detection (Sobel)
  - Edge Detection (Laplacian)
  - Blurring (Gaussian)
  - Blurring (Median)
  - Blurring (Bilateral)
""")

st.sidebar.title("Settings")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    default_image_path = "../ComputerVision/Demos/img_proc/example.jpg"
    
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

if st.sidebar.checkbox("Apply Edge Detection"):
    
    edge_method = st.sidebar.selectbox("Select Edge Detection Method", ["Canny", "Sobel", "Laplacian"])
    
    if edge_method == "Canny":
        low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 50)
        high_threshold = st.sidebar.slider("High Threshold", 0, 255, 150)
        edges = cv2.Canny(image_array, low_threshold, high_threshold)

    elif edge_method == "Sobel":
        # sobel_kernel = st.sidebar.slider("Sobel Kernel Size", 0, 1, 5)
        # if sobel_kernel % 2 == 0:   #Make odd number increments
        #     sobel_kernel += 1
        sobel_kernel = st.sidebar.slider("Sobel Kernel Size", 1, 12, 3, step=2)
        sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        edges = cv2.magnitude(sobel_x, sobel_y)

    elif edge_method == "Laplacian":
        laplacian_kernel = st.sidebar.slider("Laplacian Kernel Size", 1, 12, 3, step=2)
        laplacian = cv2.Laplacian(image_array, cv2.CV_64F, ksize=laplacian_kernel)
        #add a normalization step
        # laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.convertScaleAbs(laplacian)

    st.image(np.uint8(edges), caption=f"Applied Edge Detection [{edge_method}]", use_container_width=True, channels="GRAY")



if st.sidebar.checkbox("Apply Blurring"):
    
    blurring_type = st.sidebar.radio("Select blurring method:",["Gaussian","Median","Bilateral"])

    filter_size = st.sidebar.slider("Filter Size", 1, 47, 3, step=2)
    # blur_kernel = st.sidebar.slider("Filter Size", 1, 50, 5)
    # if blur_kernel % 2 == 0:
    #     blur_kernel += 1
    
    if blurring_type == "Gaussian":
        blurred_image = cv2.GaussianBlur(image_array, (filter_size, filter_size), 0)

    elif blurring_type == "Median":
        blurred_image = cv2.medianBlur(image_array, filter_size, 0)

    elif blurring_type == "Bilateral":
        sigmaSpace = 75
        blurred_image = cv2.bilateralFilter(image_array, filter_size, sigmaSpace, sigmaSpace)

    st.image(blurred_image, caption=f"Blurred Image [{blurring_type}]", use_container_width=True)
    # st.write("You selected:", blurring_type)
    


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

