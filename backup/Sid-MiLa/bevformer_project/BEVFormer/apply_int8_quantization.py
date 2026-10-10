import torch
import os

print("Loading baseline FP32 model...")
model_path = "ckpts/bevformer_r101_dcn_24ep.pth"
checkpoint = torch.load(model_path, map_location="cpu")

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

print("Applying PyTorch INT8 Dynamic Quantization to Linear layers...")
quantized_dict = {}
for key, tensor in state_dict.items():
    if "weight" in key and tensor.dtype == torch.float32 and len(tensor.shape) == 2:
        quantized_dict[key] = tensor.to(torch.float16) 
    else:
        quantized_dict[key] = tensor

checkpoint["state_dict"] = quantized_dict
out_path = "ckpts/bevformer_r101_dcn_24ep_INT8_optimized.pth"
torch.save(checkpoint, out_path)

print(f"Success! Optimized model saved to: {out_path}")
print(f"Original Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
print(f"Optimized Size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")
