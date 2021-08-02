import torch

model = torch.load('checkpoints/best.pth')
print(model.keys())