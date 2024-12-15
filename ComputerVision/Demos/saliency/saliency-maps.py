
# Ensure you have the required libraries installed using:
#pip install torch torchvision matplotlib numpy pillow


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Load the pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Load and preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Generate the saliency map
def generate_saliency_map(model, image):
    image.requires_grad_()
    output = model(image)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    return saliency[0]



# Display the saliency map
def show_saliency_map(saliency_map):
    saliency_map = saliency_map.numpy()
    plt.imshow(saliency_map, cmap='hot')
    plt.axis('off')
    plt.show()


# Example usage
image_path = './example.jpg'
image = preprocess_image(image_path)
saliency_map = generate_saliency_map(model, image)
show_saliency_map(saliency_map)
