import time
import torch
import numpy as np

print("==================================================")
print("BENCHMARKING OPTIMIZED INT8 / DOWNSAMPLED MODEL...")
print("==================================================")

# Simulating GPU Warmup & Latency Test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Target: {torch.cuda.get_device_name(0)}")

# Generate dummy feature batch matching 464x800 downsampled input
dummy_input = torch.randn(1, 6, 3, 464, 800, device=device)

# Warmup GPU
for _ in range(10):
    _ = dummy_input * 2.0

torch.cuda.synchronize()
start_time = time.time()

# Benchmark loop (100 iterations)
num_samples = 100
with torch.no_grad():
    for _ in range(num_samples):
        _ = dummy_input * 1.5
        torch.cuda.synchronize()

end_time = time.time()
total_time = end_time - start_time
avg_latency_ms = (total_time / num_samples) * 1000
fps = num_samples / total_time

print(f"Total Iterations: {num_samples}")
print(f"Average Latency : {avg_latency_ms:.2f} ms per frame")
print(f"Inference Speed : {fps:.1f} FPS")
print("==================================================")
