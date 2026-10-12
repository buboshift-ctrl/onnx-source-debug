import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)      # 이 라인이 살아남는지가 관건
        return self.relu(x)

torch.onnx.export(M(), (torch.randn(1,3,32,32),), "m.onnx", dynamo=True)
