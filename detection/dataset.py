import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

dataset_path = 'dataset'
pipeline = ['train', 'val']
class_dict = {0: 'anomaly', 1: 'normal'}

train_images, val_images = [], []

for class_id in class_dict.keys():
    train_path = os.path.join(dataset_path, 'train', class_dict[class_id])
    val_path = os.path.join(dataset_path, 'val', class_dict[class_id])
    
    train_images += [[image, train_path] for image in os.listdir(train_path) if image.endswith('.jpg')]
    val_images += [[image, val_path] for image in os.listdir(val_path) if image.endswith('.jpg')]

train_annot, val_annot = [], []

for image in train_images:
    train_annot += [image[0].replace(".jpg", ".txt")]
    
for image in val_images:
    val_annot += [image[0].replace(".jpg", ".txt")]
    
#########
train_images = pd.DataFrame(train_images, columns=['images', 'file_path'])
train_annot = pd.Series(train_annot, name='annots')
train_df = pd.concat([train_images, train_annot], axis=1)
train_df = pd.DataFrame(train_df)    

class CustomImageDataset(Dataset):
    def __init__(self, annotations=train_df, S=7, B=2, C=2, transform=None):
        self.annotations = annotations
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.annotations.iloc[idx, 1], self.annotations.iloc[idx, 2])
        img_path = os.path.join(self.annotations.iloc[idx, 1], self.annotations.iloc[idx, 0])
        boxes = []
        
        with open(label_path, 'r') as file:
            for line in file:
                values = line.strip().split()
                if len(values) == 5:
                    idx, x, y, w, h = map(float, values)
                    
                    boxes.append([int(idx), x, y, w, h])
                    
        boxes = torch.tensor(boxes)
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform:
            image, boxes = self.transform(image, boxes)
            
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 4:8] = box_coordinates
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix

        