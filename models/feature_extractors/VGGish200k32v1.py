import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x



class VGGish200k32(nn.Module):
    def __init__(self):
        super(VGGish200k32, self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.cnn1 = DepthwiseSeparableConv(1, 8, kernel_size=3, padding=1)        
        self.cnn2 = DepthwiseSeparableConv(8, 16, kernel_size=3, padding=1)
        self.cnn3 = DepthwiseSeparableConv(16, 32, kernel_size=3, padding=1)        
        self.cnn4 = DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1)
        self.cnn5 = DepthwiseSeparableConv(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.pool(x)
        
        x = F.relu(self.cnn2(x))
        x = self.pool(x)
        
        x = F.relu(self.cnn3(x))
        x = self.pool(x)
        
        x = F.relu(self.cnn4(x))
        x = self.pool(x)
        
        x = F.relu(self.cnn5(x))
        x = self.pool(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x