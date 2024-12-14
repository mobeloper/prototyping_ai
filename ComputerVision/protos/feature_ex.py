import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Streamlit App Title
st.title("Feature Extraction and Representation in Images")

# Sidebar Options
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
feature_option = st.sidebar.selectbox(
    "Select a Feature Extraction Technique", ["SIFT", "ORB", "Histogram"]
)

# Function to Convert PIL Image to OpenCV
@st.cache
def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

# Process Image Based on Feature Extraction Technique
if uploaded_file is not None:
    # Load Image
    image = load_image(uploaded_file)

    # Convert to Grayscale for Feature Extraction
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    st.subheader("Uploaded Image")
    st.image(image, caption="Original Image", use_column_width=True)

    if feature_option == "SIFT":
        # Initialize SIFT Detector
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        
        # Draw Keypoints on Image
        sift_image = cv2.drawKeypoints(
            image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        st.image(sift_image, caption="SIFT Keypoints", use_column_width=True)
        st.write(f"Number of Keypoints Detected: {len(keypoints)}")

    elif feature_option == "ORB":
        # Initialize ORB Detector
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        
        # Draw Keypoints on Image
        orb_image = cv2.drawKeypoints(
            image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        st.image(orb_image, caption="ORB Keypoints", use_column_width=True)
        st.write(f"Number of Keypoints Detected: {len(keypoints)}")

    elif feature_option == "Histogram":
        # Calculate Histogram
        color = ('b', 'g', 'r')
        st.subheader("Color Histograms")
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            st.line_chart(hist.flatten(), width=0, height=200)

        st.write("Histograms represent the distribution of pixel intensities for each color channel.")

else:
    st.write("Please upload an image to get started.")

