import os
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image


seed = 42
torch.manual_seed(seed)


transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.CenterCrop(250),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset_path = "dataset"
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'), transform=transform)

# # Split dataset into train and test sets (80% train, 20% test)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# print('train', len(train_dataset))
# print('val', len(val_dataset))

