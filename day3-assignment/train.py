# ! pip install kaggle

# %%writefile kaggle.json
# {
#   "username": "sanmitragoswami",
#   "key": ""
# }

# !mkdir -p ~/.kaggle
# !mv kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# !kaggle datasets list ## test if kaggle loaded
# !kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset
# !unzip microsoft-catsvsdogs-dataset.zip
# !ls

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# Data Augmentation for Training
# Compose means apply these transforms in order one after the other
train_transforms = transforms.Compose([
    transforms.Resize(256), # shorter side to 256 pixels
    transforms.CenterCrop(224), # randomly crops 224*224 patch from this
    transforms.RandomHorizontalFlip(), # randomly flips img laterally with 0.5 prob
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

    transforms.ToTensor(), # converts img to pytorch tensor output shape becomes (c,h,w)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 # image net normalisation
])

# No Augmentation for Validation/Test
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# import os

# base = "data"
# for split in ["train", "val", "test"]:
#     for cls in ["Cat", "Dog"]:
#         os.makedirs(f"{base}/{split}/{cls}", exist_ok=True)
# import random
# import shutil

# def split_class(class_name):
#     src = f"PetImages/{class_name}"
#     files = os.listdir(src)
#     random.shuffle(files)

#     n = len(files)
#     train_end = int(0.8 * n)
#     val_end = int(0.9 * n)

#     splits = {
#         "train": files[:train_end],
#         "val": files[train_end:val_end],
#         "test": files[val_end:]
#     }

#     for split, split_files in splits.items():
#         for f in split_files:
#             shutil.copy(
#                 os.path.join(src, f),
#                 f"data/{split}/{class_name}/{f}"
#             )

# split_class("Cat")
# split_class("Dog")

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


base_dir = "C:\\Users\\SANMITRA\\The Multiverse\\Programming Stuff\\jupyter NB\\CV basics\\data\\data"
# Load Datasets
train_data = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=train_transforms, loader=custom_pil_loader)
val_data = datasets.ImageFolder(os.path.join(base_dir, 'val'), transform=val_transforms, loader=custom_pil_loader)

# Create Data Loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully connected layer with a new one for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# Move to GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Using device: {device}")
print(f"Training only Final Layer with {model.fc.in_features} output to 2")

# Loss and Optimiser
criterion = nn.CrossEntropyLoss()
## cross entropy loss is a loss function used for multi-class classification
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
## creates an adam optimizer, that updates only model's fully connected params

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training Tracking
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0

n_epochs = 10
for epoch in range(n_epochs):
    # Training Phase
    model.train()
    running_losses = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Fwd
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Bck
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Statistics
        running_losses += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_losses/len(train_loader)
    train_accuracy = 100*correct/total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation Phase
    model.eval()
    val_running_losses = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_losses += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_losses/len(val_loader)
    val_accuracy = 100*val_correct/val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    ## update learning rate
    scheduler.step(val_loss)
    ## save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved best model with val_accuracy: {best_val_accuracy:.2f}%")

    print(f"Epoch: [{epoch+1}/{n_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    print()

np.save("train_losses.npy", train_losses)
np.save("val_losses.npy", val_losses)
np.save("val_accuracies.npy", val_accuracies)
np.save("train_accuracies.npy", train_accuracies)

