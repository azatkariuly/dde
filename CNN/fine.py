import cv2
import os
import torch
import time
from model import CNNModel
from model_quantized import CNNModel_Q
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_anomaly, total_anomaly_correct = 0, 0
total_normal, total_normal_correct = 0, 0

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
        "nbits": 2,
}

model = CNNModel_Q(params_model)

model.load_state_dict(torch.load("weights_best.pt", map_location=torch.device(device)), strict=False)
print('model', model)


data_path = 'dataset/val'
for image in os.listdir(os.path.join(data_path, 'normal')):
    if image.endswith('.jpg'):
        frame = cv2.imread(os.path.join(data_path, 'normal', image))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        input_tensor = transform(frame)
        input_tensor = input_tensor.unsqueeze(0)
        
        out = model(input_tensor)
        
        pred = out.argmax(dim=1, keepdim=True)
        
        print('out', out[0], pred[0])
        
        total_anomaly += 1
        
        if pred[0] == 0:
            total_anomaly_correct += 1
            

print(total_anomaly, total_anomaly_correct)
            