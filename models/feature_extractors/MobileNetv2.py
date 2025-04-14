import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion_rate: int,
        use_relu6=False,
    ):
        super(InvertedResidualBlock, self).__init__()
        self.expansion_rate = expansion_rate
        self.stride = stride
        self.hidden_dim = int(self.expansion_rate * in_channels)
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.input_conv = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=self.hidden_dim,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(self.hidden_dim)

        self.dw_conv = nn.Conv2d(
            kernel_size=3,
            groups=self.hidden_dim,
            in_channels=self.hidden_dim,
            stride=1,
            padding=1,
            out_channels=self.hidden_dim,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(self.hidden_dim)

        self.pw_conv = nn.Conv2d(
            kernel_size=1, in_channels=self.hidden_dim, out_channels=self.hidden_dim
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU6(inplace=True) if use_relu6 else nn.ReLU(inplace=True)

    def forward(self, x):
        input_copy = x

        x = self.input_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.relu(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)

        if self.use_residual:
            x = x + input_copy

        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, activation="relu"
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = (
            nn.ReLU6(inplace=True) if activation == "relu" else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvertedResidualBlock(nn.Module):
    """Bloque residual invertido con expansión y depthwise separable convolution"""

    def __init__(
        self, in_channels, out_channels, expansion_factor, stride, activation="relu"
    ):
        super().__init__()

        hidden_dim = in_channels * expansion_factor
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand = (
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            if expansion_factor > 1
            else None
        )
        self.bn1 = nn.BatchNorm2d(hidden_dim) if expansion_factor > 1 else None

        self.depthwise_separable = DepthwiseSeparableConv(
            hidden_dim, out_channels, stride=stride
        )

    def forward(self, x):
        identity = x
        if self.expand:
            x = F.relu6(self.bn1(self.expand(x)), inplace=True)
        x = self.depthwise_separable(x)
        if self.use_residual:
            x += identity  # Residual connection
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.Sequential(
            InvertedResidualBlock(32, 16, expansion_factor=1, stride=1),  # Bloque 1
            InvertedResidualBlock(16, 24, expansion_factor=6, stride=2),  # Bloque 2
            InvertedResidualBlock(24, 32, expansion_factor=6, stride=2),  # Bloque 3
            InvertedResidualBlock(32, 64, expansion_factor=6, stride=2),  # Bloque 4
            InvertedResidualBlock(64, 96, expansion_factor=6, stride=1),  # Bloque 5
            InvertedResidualBlock(96, 160, expansion_factor=6, stride=2),  # Bloque 6
            InvertedResidualBlock(160, 320, expansion_factor=6, stride=1),  # Bloque 7
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)  # Global Average Pooling
        x = self.classifier(x)
        return x


# Crear modelo y probar salida
model = MobileNetV2(num_classes=10)
x = torch.randn(1, 3, 224, 224)  # Imagen de entrada
output = model(x)
print("Output shape:", output.shape)  # (1, 10)
