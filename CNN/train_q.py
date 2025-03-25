import torch
import time
from model import CNNModel
import torch.quantization as quant

model = CNNModel()
model.load_state_dict(torch.load("weights_best.pt", map_location=torch.device('cpu')))
# Model preparation for quantization
model.eval()

# Apply 8-bit quantization
model_quantized = quant.quantize_dynamic(
    model,
    dtype=torch.qint8
)

# Save the quantized model weights
torch.save(model_quantized.state_dict(), 'quantized_model.pt')

avg_time = []

dummy_input = torch.randn(256, 3, 250, 250)

for i in range(10):
    start = time.time()
    out = model_quantized(dummy_input)
    end = time.time()
    
    avg_time += [end-start]
    
print('average', sum(avg_time)/len(avg_time))
