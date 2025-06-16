import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

devNumber = torch.cuda.current_device()
print(f"Current device number: {devNumber}")

devName=torch.cuda.get_device_name(devNumber)
print(f"Current device name: {devName}")

# Create a tensor on cpu 

T1=torch.randn(4,4)
print("CPU Tensor")
print(T1) 

# Move tensor to gpu(CUDA tensor)

T2=T1.to(device)
print("CUDA tensor")
print(T2)

