import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


seed = 42
torch.manual_seed(seed)

transform = transforms.Compose([
    transforms.Resize((480, 480)),
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

# img, label = train_dataset[0]

# print('su', img.shape, label)
# print("Follwing classes are there : \n", dataset.classes)

# def show_batch(dl):
#     for images, labels in dl:
#         fig, ax = plt.subplots(figsize=(16, 12))
#         ax.set_xticks([])  # Hide x-axis ticks
#         ax.set_yticks([])  # Hide y-axis ticks
#         ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))  # Create and display grid of images
#         plt.show()  # Ensure the plot is shown
#         break
        
# show_batch(train_loader)

# # Print class names
# class_names = dataset.classes
# print("Class Names:", class_names)

# # Example: Fetch one batch from training set
# images, labels = next(iter(train_loader))
# print(f"Train Batch shape: {images.shape}, Labels: {labels}")

# # Example: Fetch one batch from test set
# images, labels = next(iter(test_loader))
# print(f"Test Batch shape: {images.shape}, Labels: {labels}")