import torch
import torch.nn as nn
import torch.nn.functional as F


# Activación Hardswish (utilizada en MobileNetV3)
class Hardswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


# Bloque Depthwise Separable Convolutions
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(DepthwiseSeparableConv, self).__init__()
        # Expansión: capa convolucional 1x1
        self.expand = nn.Conv2d(
            in_channels, in_channels * expansion_factor, kernel_size=1
        )
        self.expand_bn = nn.BatchNorm2d(in_channels * expansion_factor)
        self.expand_activation = Hardswish()

        # Depthwise: convolución depthwise
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

        # Proyección: capa convolucional 1x1 para reducir la dimensión
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


# Bloque Bottleneck de MobileNetV4 (similar al V2, con mejor optimización)
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(Bottleneck, self).__init__()
        self.depthwise_sep_conv = DepthwiseSeparableConv(
            in_channels, out_channels, stride, expansion_factor
        )

        # Si el número de canales de entrada no es igual al número de canales de salida, añadimos una convolución de proyección.
        self.shortcut = (
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Realizar la convolución y sumar el shortcut
        out = self.depthwise_sep_conv(x)
        return F.relu(out + self.shortcut_bn(self.shortcut(x)))


# MobileNetV4 Especulativo
class MobileNetV4(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV4, self).__init__()

        # Capa inicial de convolución
        self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.initial_bn = nn.BatchNorm2d(16)
        self.initial_activation = Hardswish()

        # Secuencia de bloques Bottleneck y Depthwise Separable
        self.features = nn.Sequential(
            Bottleneck(16, 24, stride=2, expansion_factor=2),
            Bottleneck(24, 24, stride=1, expansion_factor=2),
            Bottleneck(24, 40, stride=2, expansion_factor=3),
            Bottleneck(40, 40, stride=1, expansion_factor=3),
            Bottleneck(40, 80, stride=2, expansion_factor=4),
            Bottleneck(80, 80, stride=1, expansion_factor=4),
            Bottleneck(80, 112, stride=1, expansion_factor=6),
            Bottleneck(112, 112, stride=1, expansion_factor=6),
            Bottleneck(112, 160, stride=2, expansion_factor=6),
            Bottleneck(160, 160, stride=1, expansion_factor=6),
        )

        # Capa de convolución 1x1 para la proyección final
        self.conv_1x1 = nn.Conv2d(160, 960, kernel_size=1)
        self.conv_1x1_bn = nn.BatchNorm2d(960)
        self.conv_1x1_activation = Hardswish()

        # Capa global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Capa de clasificación
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        x = self.initial_activation(self.initial_bn(self.initial_conv(x)))
        x = self.features(x)
        x = self.conv_1x1_activation(self.conv_1x1_bn(self.conv_1x1(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Crear una instancia del modelo
model = MobileNetV4(num_classes=1000)

# Imprimir la arquitectura del modelo
print(model)
