import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the KITTI Dataset class
class KITTI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for file in os.listdir(root_dir + '/camera/image_2'):
            self.images.append(root_dir + '/camera/image_2/' + file)
            self.labels.append(root_dir + '/label_2/' + file.split('.')[0] + '.txt')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        label = self.read_label(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

    def read_label(self, label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        label = []
        for line in lines:
            values = line.split()
            label.append([float(values[4]), float(values[5]), float(values[6]), float(values[7])])
        return torch.tensor(label)

# Define data augmentation transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(0.5, 1.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# Load the pre-trained SSD model
from torchvision.models import detection
model = detection.ssd300_vgg16(pretrained=True)

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Add a custom classification head
num_classes = 10
model.classification_head = nn.Sequential(
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, num_classes)
)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the dataset and data loader
dataset = KITTI_Dataset(root_dir='path/to/kitti/dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set up the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model
for epoch in range(10):
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    total_correct = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / len(dataset)
    print(f'Accuracy: {accuracy:.4f}')

# Visualize the detection results
def visualize_detections(image, detections):
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for detection in detections:
        x, y, w, h = detection['bbox']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, detection['class'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detections', image)
    cv2.waitKey(0)

# Test the model on a sample image
image = cv2.imread('path/to/sample/image.jpg')
image = transform(image)
image = image.unsqueeze(0).to(device)
outputs = model(image)
detections = []
for output in outputs:
    for detection in output:
        if detection['score'] > 0.5:
            detections.append(detection)
visualize_detections(image, detections)