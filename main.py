import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from multiprocessing import freeze_support

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
DATA_DIR = "dataset"  # ✅ Make sure path is correct
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dynamic transform dataset wrapper
class DynamicTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, apply_dynamic=True):
        self.dataset = dataset
        self.transform = transform
        self.apply_dynamic = apply_dynamic

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.apply_dynamic and self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

# Load base dataset
print("Loading dataset...")
base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
print(f"Total images in full dataset: {len(base_dataset)}")
print(f"Classes: {base_dataset.classes}")
print(f"Number of classes: {len(base_dataset.classes)}")

# Split dataset
train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size
indices = torch.randperm(len(base_dataset), generator=torch.Generator().manual_seed(42))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_subset = Subset(base_dataset, train_indices)
val_subset = Subset(base_dataset, val_indices)

train_dataset = DynamicTransformDataset(train_subset, transform=train_transform, apply_dynamic=True)
val_dataset = DynamicTransformDataset(val_subset, transform=val_transform, apply_dynamic=True)


print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Load pre-trained model
print("Initializing model...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # Freeze backbone

# Replace classifier
num_classes = len(base_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE)

# Training loop
def train():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return epoch_loss, accuracy

# Validation loop
def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return epoch_loss, accuracy

if __name__ == "__main__":
    freeze_support()
    print("Starting training...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        train_loss, train_acc = train()
        val_loss, val_acc = validate()
        print(f"Epoch [{epoch+1}/{EPOCHS}]\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Save model
    torch.save(model.state_dict(), "crop_disease_model.pth")
    print("Model saved as crop_disease_model.pth")