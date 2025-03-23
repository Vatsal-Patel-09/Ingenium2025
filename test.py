import torch

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import torch

print("CUDA Available:", torch.cuda.is_available())
print("Active GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
