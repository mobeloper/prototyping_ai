	import torch
import torchvision.models as models
import torch.nn.functional as F
 
def generate_saliency_map(image):
    """
    Generates a saliency map using the ResNet18 model.
 
    Parameters:
    - image: torch.Tensor
        The input image for which the saliency map needs to be generated.
 
    Returns:
    - torch.Tensor:
        The saliency map generated by the ResNet18 model.
 
    Raises:
    - TypeError:
        Raises an error if the input image is not a torch.Tensor.
    """
 
    # Checking if the input image is a torch.Tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image should be a torch.Tensor.")
 
    # Loading the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    model.eval()
 
    # Disabling gradient computation for the model
    for param in model.parameters():
        param.requires_grad = False
 
    # Forward pass through the model to get the output logits
    logits = model(image)
 
    # Calculating the gradients of the logits with respect to the input image
    gradients = torch.autograd.grad(outputs=logits, inputs=image, grad_outputs=torch.ones_like(logits),
                                    create_graph=True, retain_graph=True)[0]
 
    # Computing the saliency map by taking the absolute values of the gradients
    saliency_map = torch.abs(gradients)
 
    # Normalizing the saliency map
    saliency_map = F.normalize(saliency_map, p=1, dim=1)
 
    return saliency_map
 
# Unit tests for generate_saliency_map function.
 
import unittest
 
class TestGenerateSaliencyMap(unittest.TestCase):
 
    # Tests for positive cases
    def test_generate_saliency_map_with_valid_input(self):
        """
        Tests the generation of saliency map with a valid input image.
        """
        image = torch.randn(1, 3, 224, 224)
        saliency_map = generate_saliency_map(image)
        self.assertEqual(saliency_map.shape, (1, 3, 224, 224))
 
    # Tests for negative cases
    def test_generate_saliency_map_with_invalid_input(self):
        """
        Tests if TypeError is raised when the input image is not a torch.Tensor.
        """
        with self.assertRaises(TypeError):
            generate_saliency_map("image.jpg")
 
    def test_generate_saliency_map_with_empty_input(self):
        """
        Tests if ValueError is raised when the input image is empty.
        """
        with self.assertRaises(ValueError):
            generate_saliency_map(torch.empty(0))
 
    # Tests for edge cases
    def test_generate_saliency_map_with_single_pixel_image(self):
        """
        Tests the generation of saliency map with a single pixel image.
        """
        image = torch.randn(1, 3, 1, 1)
        saliency_map = generate_saliency_map(image)
        self.assertEqual(saliency_map.shape, (1, 3, 1, 1))
 
    def test_generate_saliency_map_with_large_image(self):
        """
        Tests the generation of saliency map with a large input image.
        """
        image = torch.randn(1, 3, 1000, 1000)
        saliency_map = generate_saliency_map(image)
        self.assertEqual(saliency_map.shape, (1, 3, 1000, 1000))
 
# Example usage of generate_saliency_map function:
 
# Example 1: Generating saliency map for an image
image = torch.randn(1, 3, 224, 224)
saliency_map = generate_saliency_map(image)
print(f"Saliency map shape: {saliency_map.shape}")
 
# Example 2: Generating saliency map with invalid input (should raise an error)
try:
    saliency_map = generate_saliency_map("image.jpg")
    print(f"Saliency map shape: {saliency_map.shape}")
except TypeError as e:
    print(f"Error while generating saliency map: {e}")