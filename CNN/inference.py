import cv2
import os
import torch
import time
from model import CNNModel
from model_q import CNNModel_LSQ
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 480)),
    transforms.CenterCrop(250),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = CNNModel()
model_lsq = CNNModel_LSQ(nbits=2)

model.load_state_dict(torch.load("weights_best.pt", map_location=torch.device('cpu')))
model_lsq.load_state_dict(torch.load("src/lsq/weights_best_lsq_2b.pt", map_location=torch.device('cpu')))

# Path to input video
video_path = "../videos/test_data/event_2.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

class_dict = [['anomaly', (255, 0, 0)], ['normal', (0, 255, 0)]]

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    (h, w) = frame.shape[:2]

    input_tensor = transform(frame)
    input_tensor = input_tensor.unsqueeze(0)
    
    
    start = time.time()
    out = model_lsq(input_tensor)
    end = time.time()
    
    print('took', end-start)
    
    pred = out.argmax(dim=1, keepdim=True)
    
    # print('out', frame_count, out, pred)
    
    # frame_count += 1
    
    d = class_dict[pred[0]]
    
    (text_w, text_h), baseline = cv2.getTextSize(d[0], font, font_scale, thickness)
    x = w - text_w - 10
    y = text_h + 10
    cv2.putText(frame, d[0], (x, y), font, font_scale, d[1], thickness)
    
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
