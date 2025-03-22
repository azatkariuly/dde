import os
import shutil
import random

random.seed(42)

# t_count = 0
# for filename in os.listdir(os.path.join('dataset_split', 'train', 'normal')):
#     if filename.endswith('.jpg'):
#         t_count += 1
        
# v_count = 0
# for filename in os.listdir(os.path.join('dataset_split', 'val', 'normal')):
#     if filename.endswith('.jpg'):
#         v_count += 1
        
# print(t_count, v_count)

# Define your dataset directories
dataset_dir = 'dataset'
train_dir = 'dataset_split/train'
val_dir = 'dataset_split/val'

# Create directories for train and val sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'anomaly'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'normal'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'anomaly'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'normal'), exist_ok=True)

# Define a function to copy images from source to destination
def copy_images(source_dir, subfolder, split_ratio=0.8):
    # List all image files in the source directory
    images = [img for img in os.listdir(source_dir) if img.endswith(('jpg'))]

    # Shuffle the list of images
    random.shuffle(images)

    # Calculate the split index
    split_index = int(len(images) * split_ratio)

    # Split the images into training and validation sets
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy the images to the corresponding directories
    for img in train_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join('dataset_split', 'train', os.path.basename(subfolder), img))
    for img in val_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join('dataset_split', 'val', os.path.basename(subfolder), img))

# Apply the function for both 'anomaly' and 'normal' folders
copy_images(os.path.join(dataset_dir, 'anomaly'), 'anomaly', split_ratio=0.8)
copy_images(os.path.join(dataset_dir, 'normal'), 'normal', split_ratio=0.8)

print("Dataset split completed!")
