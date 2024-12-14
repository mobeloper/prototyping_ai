import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.title("Feature Extraction Demo: SIFT and Feature Visualization")
st.write("""

### Features:
- **Feature Extraction**: Detect keypoints and descriptors using SIFT (Scale-Invariant Feature Transform).
- **Feature Visualization**: Visualize extracted features by overlaying keypoints and creating feature density maps.
""")

st.sidebar.title("Settings")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

default_image_path = "example.jpg"
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
else:
    image = Image.open(default_image_path).convert('RGB')

image_array = np.array(image)

st.image(image, caption="Input Image", use_container_width=True)

# Convert to grayscale for feature extraction
gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

# SIFT
if st.sidebar.checkbox("Extract Features using SIFT"):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    keypoint_image = cv2.drawKeypoints(image_array, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(keypoint_image, caption=f"SIFT Keypoints ({len(keypoints)} detected)", use_container_width=True)

    # Feature density map
    feature_density = np.zeros_like(gray_image, dtype=np.float32)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        feature_density[y, x] += 1

    feature_density_normalized = cv2.normalize(feature_density, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    st.image(feature_density_normalized, caption="Feature Density Map", use_container_width=True, clamp=True)

    # Heatmap
    if descriptors is not None:
        st.write("### Visualizing Descriptors as Heatmap")
        descriptor_avg = np.mean(descriptors, axis=0)
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(descriptor_avg)), descriptor_avg)
        plt.title("Average Descriptor Values")
        plt.xlabel("Descriptor Dimensions")
        plt.ylabel("Average Value")
        st.pyplot(plt)

# Histogram
if st.sidebar.checkbox("Show Histogram Analysis"):
    st.write("### Histogram Analysis")

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist, color='black')
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    st.pyplot(plt)

    # Highlight regions with specific intensity ranges
    low_threshold = st.sidebar.slider("Low Intensity Threshold", 0, 255, 50)
    high_threshold = st.sidebar.slider("High Intensity Threshold", 0, 255, 200)

    highlighted_regions = cv2.inRange(gray_image, low_threshold, high_threshold)
    overlay = cv2.bitwise_and(image_array, image_array, mask=highlighted_regions)
    st.image(overlay, caption=f"Regions with Intensity between {low_threshold} and {high_threshold}", use_container_width=True)


