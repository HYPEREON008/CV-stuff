import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import models, transforms, datasets
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader

# Custom loader to handle UnidentifiedImageError
def custom_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img
    except (UnidentifiedImageError, OSError) as e:
        print(f"Warning: Could not load image {path}. Error: {e}. Returning a black image placeholder.")
        # Return a black image of expected size (e.g., 224x224 RGB) as a placeholder
        # This assumes the target image size after transforms is 224x224.
        # If transforms include resizing, this might need adjustment or a more complex collate_fn.
        return Image.new('RGB', (224, 224), (0, 0, 0))

# No Augmentation for Validation/Test
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_dir = "C:\\Users\\SANMITRA\\The Multiverse\\Programming Stuff\\jupyter NB\\CV basics\\data\\data"
test_data = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=val_transforms, loader=custom_pil_loader)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Test data size: {len(test_data)}")

## load training curve data
train_losses = np.load("train_losses.npy").tolist()
train_accuracies = np.load("train_accuracies.npy").tolist()
val_losses = np.load("val_losses.npy").tolist()
val_accuracies = np.load("val_accuracies.npy").tolist()

os.chdir("C:\\Users\\SANMITRA\\GitHubRepos\\CV-stuff\\day3-assignment")

## Load best model for evaluation
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final layer (same as training)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# Move model to device
model = model.to(device)

# Evaluation mode
model.eval()

print(test_loader.dataset.root)
print(test_loader.dataset.samples[:2])

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100*correct/total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()

ax2.plot(train_accuracies, label="Train Accuracy")
ax2.plot(val_accuracies, label="Val Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy %")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# Confusion Matrix
cmatrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cmatrix, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

