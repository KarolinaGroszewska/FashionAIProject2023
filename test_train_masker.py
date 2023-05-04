import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2


class TrainMasker:
    def __init__(self):
        # Define the ResNet50 model pretrained on ImageNet
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
        self.model.eval()

        # Define the mean and standard deviation of the ImageNet dataset
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

    def get_binary_mask(self, image_name, foreground, plot=False):
        # Load the image using OpenCV
        image = cv2.imread(image_name)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a PyTorch tensor
        tensor = transforms.ToTensor()(image)

        # Normalize the image using the mean and standard deviation of the ImageNet dataset
        tensor = (tensor - self.mean) / self.std

        # Add a batch dimension to the tensor
        tensor = tensor.unsqueeze(0)

        # Move the tensor to the GPU if available
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            self.model.cuda()

        # Get the feature map from the last convolutional layer of the model
        with torch.no_grad():
            features = self.model.conv1(tensor)
            features = self.model.bn1(features)
            features = self.model.relu(features)
            features = self.model.maxpool(features)

            features = self.model.layer1(features)
            features = self.model.layer2(features)
            features = self.model.layer3(features)
            features = self.model.layer4(features)

        # Get the height and width of the feature map
        height, width = features.shape[2:]

        # Create a mask initialized to zeros
        mask = np.zeros((height, width))

        # Set the foreground pixel to one
        mask[foreground[1], foreground[0]] = 1

        # Compute the logits from the feature map
        logits = self.model.fc(features.mean([2, 3]))

        # Compute the gradients of the logits with respect to the feature map
        gradients = torch.autograd.grad(logits[0, foreground[2]], features)[0][0]

        # Compute the weight of each pixel
        weights = gradients.abs().mean(dim=0)

        # Normalize the weights
        weights = (weights - weights.min()) / (weights.max() - weights.min())

        # Apply the weights to the mask
        mask = weights.cpu().numpy() * mask

        # Normalize the mask
        mask = (mask - mask.min()) / (mask.max() - mask.min())

        # Set the threshold to 0.5
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Convert the mask to uint8
        mask = (mask * 255).astype(np.uint8)

        # Apply morphological opening to the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

