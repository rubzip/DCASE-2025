import torch
import torch.nn as nn
import torch.nn.functional as F


class Hardswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(DepthwiseSeparableConv, self).__init__()
        self.expand = nn.Conv2d(
            in_channels, in_channels * expansion_factor, kernel_size=1
        )
        self.expand_bn = nn.BatchNorm2d(in_channels * expansion_factor)
        self.expand_activation = Hardswish()

        self.depthwise = nn.Conv2d(
            in_channels * expansion_factor,
            in_channels * expansion_factor,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels * expansion_factor,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels * expansion_factor)
        self.depthwise_activation = Hardswish()

        self.project = nn.Conv2d(
            in_channels * expansion_factor, out_channels, kernel_size=1
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Expansión -> Depthwise -> Proyección
        x = self.expand_activation(self.expand_bn(self.expand(x)))
        x = self.depthwise_activation(self.depthwise_bn(self.depthwise(x)))
        x = self.project_bn(self.project(x))
        return x


# Bloque principal del modelo MobileNetV3
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3, self).__init__()

        # Capa inicial
        self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.initial_bn = nn.BatchNorm2d(16)
        self.initial_activation = Hardswish()

        # Secuencia de bloques Depthwise Separable
        self.features = nn.Sequential(
            DepthwiseSeparableConv(16, 24, stride=2, expansion_factor=2),
            DepthwiseSeparableConv(24, 24, stride=1, expansion_factor=2),
            DepthwiseSeparableConv(24, 40, stride=2, expansion_factor=3),
            DepthwiseSeparableConv(40, 40, stride=1, expansion_factor=3),
            DepthwiseSeparableConv(40, 80, stride=2, expansion_factor=4),
            DepthwiseSeparableConv(80, 80, stride=1, expansion_factor=4),
            DepthwiseSeparableConv(80, 112, stride=1, expansion_factor=6),
            DepthwiseSeparableConv(112, 112, stride=1, expansion_factor=6),
            DepthwiseSeparableConv(112, 160, stride=2, expansion_factor=6),
            DepthwiseSeparableConv(160, 160, stride=1, expansion_factor=6),
        )

        self.conv_1x1 = nn.Conv2d(160, 960, kernel_size=1)
        self.conv_1x1_bn = nn.BatchNorm2d(960)
        self.conv_1x1_activation = Hardswish()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        x = self.initial_activation(self.initial_bn(self.initial_conv(x)))
        x = self.features(x)
        x = self.conv_1x1_activation(self.conv_1x1_bn(self.conv_1x1(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Inicializar el modelo
model = MobileNetV3(num_classes=1000)

# Imprimir la arquitectura
print(model)
