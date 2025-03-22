import os
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Detection
from dataset import CustomImageDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)
from loss import DetectionLoss

seed = 42
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 5
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "weights.pt"
    
dataset_path = 'dataset'
pipeline = ['train', 'val']
class_dict = {0: 'anomaly', 1: 'normal'}

# files_dir = 'data/train'
# test_dir = 'data/test'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():

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
    
    val_images = pd.DataFrame(val_images, columns=['images', 'file_path'])
    val_annot = pd.Series(val_annot, name='annots')
    val_df = pd.concat([val_images, val_annot], axis=1)
    val_df = pd.DataFrame(val_df)        
        
        

    model = Detection(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    loss_fn = DetectionLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = CustomImageDataset(
        transform=transform,
        annotations=train_df
    )

    val_dataset = CustomImageDataset(
        transform=transform,
        annotations=val_df
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)
        
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        
        scheduler.step(mean_avg_prec)
    
    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
    }

    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
    
if __name__ == "__main__":
    main()