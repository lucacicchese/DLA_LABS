import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define CIFAR-10 transform (ResNet expects 224x224 inputs)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Get a batch of 2 images
images, labels = next(iter(dataloader))
input1 = images[0].unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]
input2 = images[1].unsqueeze(0).to(device)

# Load two independent ResNet18 models (not pretrained)
model1 = resnet18(pretrained=False).to(device)
model2 = resnet18(pretrained=False).to(device)

# Replace classification head with Identity to get feature embeddings
model1.fc = nn.Identity()
model2.fc = nn.Identity()

# Set both models to evaluation mode
model1.eval()
model2.eval()

# Forward pass through each model
with torch.no_grad():
    features1 = model1(input1)  # Shape: [1, 512]
    features2 = model2(input2)

# Compute cosine similarity
cos = nn.CosineSimilarity(dim=1)
similarity = cos(features1, features2)

print(f"Cosine similarity between features: {similarity.item():.4f}")
