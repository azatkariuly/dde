import time
import torch
from model import CNNModel
from torch.quantization import quantize_dynamic

# # Load your trained model
# model = CNNModel()
# model.load_state_dict(torch.load("weights_best.pt", map_location=torch.device('cpu')))
# model.eval()

# # Apply dynamic quantization
# quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# # Save quantized model
# torch.save(quantized_model.state_dict(), "quantized_model.pt")

# # Check model size reduction
# print("Original Model Size:", sum(p.numel() for p in model.parameters()))
# print("Quantized Model Size:", sum(p.numel() for p in quantized_model.parameters()))

import torch
import torch.nn as nn

class FastQuantizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quantization_scale = {}  # Scale factors
        self.quantized_weights = {}   # 4-bit weight storage

        self.quantize_model()  # Quantize weights

    def quantize_tensor(self, tensor):
        """Vectorized quantization: Convert tensor to 4-bit integer representation."""
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / 15  # 4-bit has 16 levels
        quantized = torch.round((tensor - min_val) / scale).clamp(0, 15) - 8  # Shift to range [-8, 7]

        return quantized.to(torch.int8), scale, min_val

    def dequantize_tensor(self, quantized, scale, min_val):
        """Vectorized dequantization: Convert back to floating point."""
        return quantized.float() * scale + min_val

    def quantize_model(self):
        """Apply 4-bit quantization to all linear layers."""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                quantized, scale, min_val = self.quantize_tensor(param.data)
                self.quantized_weights[name] = quantized
                self.quantization_scale[name] = (scale, min_val)
                param.data = self.dequantize_tensor(quantized, scale, min_val)  # Use quantized weights

    def fast_linear(self, x, weight_name):
        quantized_weight = self.quantized_weights[weight_name]
        scale, min_val = self.quantization_scale[weight_name]

        # Convert input tensor to INT8 for faster computations
        x_int8 = (x - x.min()) / ((x.max() - x.min()) / 127)
        x_int8 = torch.round(x_int8).to(torch.int8)

        # Matrix multiplication (SIMD-like acceleration with int8)
        output = torch.mm(x_int8.float(), quantized_weight.float())
        output = output * scale + min_val

        return output

    def forward(self, x):
        """Override forward method to use fast quantized layers."""
        i = 0
        for name, module in self.model.named_modules():
            if i == 0:
                i += 1
                continue
            if isinstance(module, nn.Linear) and name + ".weight" in self.quantized_weights:
                x = self.fast_linear(x, name + ".weight")  # Apply fast INT8 matmul
            else:
                x = module(x)
        return x

    def save_quantized_weights(self, file_path="fast_quantized_weights.pth"):
        """Save quantized weights for fast loading."""
        torch.save({"quantized_weights": self.quantized_weights, "scales": self.quantization_scale}, file_path)

    def load_quantized_weights(self, file_path="fast_quantized_weights.pth"):
        """Load quantized weights and dequantize them."""
        checkpoint = torch.load(file_path)
        self.quantized_weights = checkpoint["quantized_weights"]
        self.quantization_scale = checkpoint["scales"]

class QuantizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quantization_scale = {}  # Store scale factors for each layer
        self.quantized_weights = {}   # Store 4-bit quantized weights

        self.quantize_model()  # Apply quantization

    def quantize_tensor(self, tensor):
        """Quantizes a tensor to 4-bit integers."""
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / 15  # 4-bit: 2^4 = 16 levels (0-15)
        quantized = torch.round((tensor - min_val) / scale).clamp(0, 15) - 8  # Center at 0

        return quantized.to(torch.int8), scale, min_val

    def dequantize_tensor(self, quantized, scale, min_val):
        """Dequantizes a 4-bit tensor back to floating point."""
        return quantized.float() * scale + min_val

    def quantize_model(self):
        """Quantizes all linear layers in the model."""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                quantized, scale, min_val = self.quantize_tensor(param.data)
                self.quantized_weights[name] = quantized
                self.quantization_scale[name] = (scale, min_val)
                param.data = self.dequantize_tensor(quantized, scale, min_val)  # Store in FP32 form

    def forward(self, x):
        return self.model(x)

    def save_quantized_weights(self, file_path="quantized_weights.pth"):
        """Save quantized weights and scaling factors."""
        torch.save({"quantized_weights": self.quantized_weights, "scales": self.quantization_scale}, file_path)

    def load_quantized_weights(self, file_path="quantized_weights.pth"):
        """Load quantized weights and dequantize them."""
        checkpoint = torch.load(file_path)
        self.quantized_weights = checkpoint["quantized_weights"]
        self.quantization_scale = checkpoint["scales"]

        for name, param in self.model.named_parameters():
            if name in self.quantized_weights:
                scale, min_val = self.quantization_scale[name]
                param.data = self.dequantize_tensor(self.quantized_weights[name], scale, min_val)  # Restore FP32

# Load pre-trained model
original_model = CNNModel()

fast_quant_model = FastQuantizedModel(original_model)

# Save & Load quantized weights
fast_quant_model.save_quantized_weights("resnet18_4bit_fast.pth")
fast_quant_model.load_quantized_weights("resnet18_4bit_fast.pth")

# Benchmark Inference Speed
dummy_input = torch.randn(256, 3, 250, 250)

# # Measure time for original model
start = time.time()
_ = original_model(dummy_input)
end = time.time()
print(f"Original Model Inference Time: {end - start:.6f} sec")

# Measure time for quantized model
start = time.time()
_ = fast_quant_model(dummy_input)
end = time.time()
print(f"Quantized Model Inference Time: {end - start:.6f} sec")
# original_model.eval()

# # Quantize model
# quantized_model = QuantizedModel(original_model)

# # Save quantized weights
# quantized_model.save_quantized_weights("resnet18_4bit.pth")

# # Load quantized model
# quantized_model.load_quantized_weights("resnet18_4bit.pth")

# # Test inference
# dummy_input = torch.randn(128, 3, 250, 250)

# for i in range(10):
#     start = time.time()
#     output = original_model(dummy_input)
#     end = time.time()
#     print('time', end-start)
# print(output.shape)

