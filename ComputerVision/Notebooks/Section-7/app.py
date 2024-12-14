import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Load the pre-trained VGG16 model
model = models.vgg19(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the transformations to match the input format of VGG16
transform = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Crop to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Function to load image from URL or file
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to predict the class label
def predict(image):
    # Apply transformations
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)  # Get model predictions
    _, predicted_idx = torch.max(outputs, 1)  # Get the index of the max output
    return predicted_idx.item()

# Load class labels (ImageNet classes)
def load_labels():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    response = requests.get(url)
    return response.json()

# Streamlit UI
st.title("Image Classification with Deep Learning: Powered by CNN")

st.write("""
This application allows you to upload an image, which will then be classified using the VGG16 model pre-trained on ImageNet.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)  # Updated to use_container_width

    # Predict the image
    if st.button("Classify Image"):
        # Perform the prediction
        predicted_class_idx = predict(image)
        labels = load_labels()  # Load the ImageNet labels
        predicted_class = labels[str(predicted_class_idx)][1]  # Get the class label
        st.write(f"Predicted class: {predicted_class}")
