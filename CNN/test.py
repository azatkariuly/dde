import torch
import torch.quantization
from model import CNNModel
import os

params_model = {
    "shape_in": (3, 480, 480),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2,
}
model_fp32 = CNNModel(params_model)
model_fp32.load_state_dict(torch.load("weights_best.pt"))

# Convert the model to eval mode before quantization
model_fp32.eval()

# Apply static quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # Original model
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8  # Quantization type
)

print(model_int8.state_dict())

# Save the INT8 model
# torch.save(model_int8.state_dict(), "weights_q8.pt")

# Get INT8 model size
# int8_size = os.path.getsize("weights_q8.pt") / 1024  # in KB
# print(f"INT8 Model Size: {int8_size:.2f} KB")
