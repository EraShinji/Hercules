import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 5, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm0(self.conv0(x)))
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = F.gelu(self.conv6(x))
        return x