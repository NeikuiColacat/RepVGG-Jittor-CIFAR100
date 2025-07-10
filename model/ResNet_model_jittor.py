import jittor as jt
import jittor.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(out_channels)
            )

    def execute(self, x):
        return nn.relu(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64 // 2

        self.conv1 = nn.Sequential(
            nn.Conv(3, 64 // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm(64 // 2),
            nn.ReLU()
        )
        self.conv2_x = self._make_layer(block, 64 // 2, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128 // 2, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 // 2, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 // 2, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 // 2, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def execute(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])