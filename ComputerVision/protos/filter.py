import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Streamlit App Title
st.title("Image Processing and Filtering with OpenCV")

# Sidebar Options
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
filter_option = st.sidebar.selectbox("Select a Filter", ["Original", "Grayscale", "Blur", "Canny Edges", "Thresholding"])

# Function to Convert PIL Image to OpenCV
@st.cache
def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

# Process Image Based on Filter Option
if uploaded_file is not None:
    # Load Image
    image = load_image(uploaded_file)

    # Display Original Image
    st.subheader("Uploaded Image")
    st.image(image, caption="Original Image", use_column_width=True)

    processed_image = image.copy()

    # Apply Selected Filter
    if filter_option == "Grayscale":
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        st.image(processed_image, caption="Grayscale Image", use_column_width=True, channels="GRAY")

    elif filter_option == "Blur":
        ksize = st.sidebar.slider("Select Kernel Size for Blur", 3, 21, 5, step=2)
        processed_image = cv2.GaussianBlur(processed_image, (ksize, ksize), 0)
        st.image(processed_image, caption="Blurred Image", use_column_width=True)

    elif filter_option == "Canny Edges":
        threshold1 = st.sidebar.slider("Lower Threshold", 50, 150, 50)
        threshold2 = st.sidebar.slider("Upper Threshold", 100, 200, 150)
        processed_image = cv2.Canny(processed_image, threshold1, threshold2)
        st.image(processed_image, caption="Canny Edge Detection", use_column_width=True)

    elif filter_option == "Thresholding":
        thresh_val = st.sidebar.slider("Threshold Value", 0, 255, 127)
        _, processed_image = cv2.threshold(
            cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY), thresh_val, 255, cv2.THRESH_BINARY
        )
        st.image(processed_image, caption="Thresholded Image", use_column_width=True)

    else:
        st.image(processed_image, caption="Original Image", use_column_width=True)

else:
    st.write("Please upload an image to get started.")

