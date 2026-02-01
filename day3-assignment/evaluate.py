import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

## Load best model for evaluation
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild ResNet18
model = models.resnet18(weights=None)

# Replace final layer (same as training)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# Move model to device
model = model.to(device)

# Evaluation mode
model.eval()

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize(14, 5))

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
# plt.save_fig("training_curves.png")
plt.show()

# Confusion Matrix
cmatrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cmatrix, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
# plt.save_fig("confusion_matrix.png")
plt.show()

