import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for image classification
    """

    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        # Convolutional layers: progressively increase number of filters (3 -> 32 -> 64 -> 128)
        # 3x3 kernels with padding=1 maintain spatial dimensions before pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling reduces spatial dimensions by half
        # Fully connected layers: flatten feature maps and classify
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 128 channels * 4x4 spatial resolution
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def create_resnet18_cifar(num_classes: int = 100) -> nn.Module:
    """ResNet18 adapted for 32x32 inputs (no initial downsampling)."""
    model = models.resnet18(num_classes=num_classes)
    # Replace the initial conv and remove maxpool for CIFAR-sized images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def create_resnet34_cifar(num_classes: int = 100) -> nn.Module:
    """ResNet34 adapted for 32x32 inputs (no initial downsampling)."""
    model = models.resnet34(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def create_model(num_classes: int, device, model_name: str = "resnet18_cifar"):
    """Create and initialize the model by name. Backward-compatible signature."""
    if model_name == "resnet18_cifar":
        model = create_resnet18_cifar(num_classes=num_classes)
    elif model_name == "resnet34_cifar":
        model = create_resnet34_cifar(num_classes=num_classes)
    elif model_name == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    else:
        # default to SimpleCNN if unknown
        model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return model
