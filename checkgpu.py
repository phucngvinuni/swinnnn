import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch was built with CUDA version: {torch.version.cuda}")
    print(f"Current GPU model: {torch.cuda.get_device_name(0)}")