import os
import torch
import torch.optim as optim
import cv2
from model import CNNModel
import torchvision.transforms as transforms

data_path = 'dataset/val'

total_anomaly, total_anomaly_correct = 0, 0
total_normal, total_normal_correct = 0, 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

params_model = {
        "shape_in": (3, 480, 480), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2,
        "nbits": 8,
}

model = CNNModel(params_model)

model.load_state_dict(torch.load("weights_best.pt"))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

for image in os.listdir(os.path.join(data_path, 'anomaly')):
    if image.endswith('.jpg'):
        frame = cv2.imread(os.path.join(data_path, 'anomaly', image))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        input_tensor = transform(frame)
        input_tensor = input_tensor.unsqueeze(0)
        
        out = model(input_tensor)
        
        pred = out.argmax(dim=1, keepdim=True)
        
        print('out', out, pred)
            
        # total_anomaly += 1
        
        # if class_idx == 0:
        #     total_anomaly_correct += 1
            

print(total_anomaly, total_anomaly_correct)
            

