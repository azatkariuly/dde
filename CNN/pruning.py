import os
import torch
import torch.nn as nn
import torch.sparse
import time
import torch.nn.utils.prune as prune
from PIL import Image
from model import CNNModel
import torchvision.transforms as transforms

data_path = 'dataset/val'

total_anomaly, total_anomaly_correct = 0, 0
total_normal, total_normal_correct = 0, 0

params_model = {
        "shape_in": (3, 480, 480), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2,
        # "nbits": 8,
}

transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = CNNModel(params_model)

# define computation hardware approach (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load("weights_best.pt", map_location=torch.device('cpu')))

# Apply Pruning
def apply_pruning(model, amount=0.5):
    """ Apply L1 unstructured pruning to convolutional and fully connected layers. """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            
def remove_pruning(model):
    """ Remove pruning masks, keeping only the pruned weights. """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
            
def convert_to_sparse(model):
    """ Convert the model's weights to sparse format. """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Convert weight tensor to sparse
            weight_data = module.weight.data
            sparse_weight = weight_data.to_sparse()
            
            # Wrap the sparse tensor in a Parameter to keep it trainable
            module.weight = torch.nn.Parameter(sparse_weight)
    return model

# tensor_normal = torch.randn(32, 3, 480, 480)
# print('sss', model.conv1.weight.to_sparse())

# print('suka',model.conv1.weight)
apply_pruning(model, amount=0.3)

# # Convert to sparse model
# model = convert_to_sparse(model)

# torch.save(model.state_dict(), "sparse_model.pth")
# print("Model saved successfully.")

# print('suka',model.conv1.weight)
# remove_pruning(model)
# # print('suka',model.conv1.weight)

for label, folder in enumerate(['anomaly', 'normal']):
    for image in os.listdir(os.path.join(data_path, folder)):
        if image.endswith('.jpg'):
            frame = Image.open(os.path.join(data_path, folder, image))
            
            input_tensor = transform(frame)
            input_tensor = input_tensor.unsqueeze(0)
            
            out = model(input_tensor)
            pred = out.argmax(dim=1, keepdim=True)
            
            if folder == 'anomaly':
                total_anomaly += 1
                if pred[0] == 0:
                    total_anomaly_correct += 1
            else:  # 'normal' folder
                total_normal += 1
                if pred[0] == 1:
                    total_normal_correct += 1
                    
print('res', total_anomaly, total_anomaly_correct, total_normal, total_normal_correct)


# # time_array = []
# # for i in range(10):
# #     start = time.time()
# #     out = model(tensor_normal)
# #     end = time.time()
    
# #     time_array += [end-start]
# # print('ss', sum(time_array)/len(time_array))

