
# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade Pillow

# Ensure you have the required libraries installed using:
#pip install --upgrade torch torchvision matplotlib numpy Pillow


import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import __version__
except ImportError:
    from PIL import PILLOW_VERSION as __version__

def generate_saliency_map(model, image_tensor, target_class):
    image_tensor.requires_grad = True
    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    saliency = image_tensor.grad.abs().squeeze().detach().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency


def load_imagenet_classes():
    class_labels = []
    with open("imagenet_classes.txt", "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) > 1:  # Ensure there are at least two elements
                class_labels.append(parts[1])
            else:
                print(f"Skipping malformed line: {line.strip()}")
    return class_labels

def main():
    st.title("CNN Saliency Map Visualization")
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    default_image_path = "example.jpg"
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    else:
        image = Image.open(default_image_path).convert('RGB')
    st.image(image, caption="Input Image", use_container_width=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    model = models.resnet18(pretrained=True)
    model.eval()
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    class_labels = load_imagenet_classes()
    top5_prob, top5_classes = torch.topk(probabilities, 5)
    st.write("Top 5 Predictions:")
    human_readable_predictions = []
    for i in range(5):
        label = class_labels[top5_classes[i].item()].replace("_", " ")
        prob = top5_prob[i].item()
        human_readable_predictions.append(f"{label}: {prob:.2f}")
        st.write(f"{label}: {prob:.2f}")
    selected_class = st.sidebar.selectbox("Select Class for Saliency Map", human_readable_predictions)
    target_class_label = selected_class.split(":")[0].strip()
    target_class = class_labels.index(target_class_label.replace(" ", "_"))
    saliency_map = generate_saliency_map(model, image_tensor, target_class)
    plt.imshow(saliency_map, cmap='hot')
    plt.axis('off')
    st.pyplot(plt)

if __name__ == "__main__":
    main()

