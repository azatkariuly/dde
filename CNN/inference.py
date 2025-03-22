import cv2
import os
import torch
import time
# from model import CNNModel
from model_quantized import CNNModel_Q
from torchvision import transforms

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

model = CNNModel_Q(params_model)

model.load_state_dict(torch.load("weights_best_480.pt"))

model_int8 = torch.quantization.quantize_dynamic(
    model,  # Original model
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8  # Quantization type
)


# Path to input video
video_path = "video/normal/cropped_normal_event_01.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    input_tensor = transform(frame)
    input_tensor = input_tensor.unsqueeze(0)
    
    
    start = time.time()
    out = model(input_tensor)
    end = time.time()
    
    print('took', end-start)
    
    pred = out.argmax(dim=1, keepdim=True)
    
    print('out', frame_count, out, pred)
    
    frame_count += 1

cap.release()
