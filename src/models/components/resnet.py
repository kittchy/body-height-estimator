import torch
import torch.nn as nn
from .senet import SENet

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        hid_channels,
        use_senet=False,
        ratio=16,
        stride=1,
        downsample=None,
    ):
        super(BasicBlock, self).__init__()
        out_channels = hid_channels * self.expansion
        self.conv1 = conv3x3(in_channels, hid_channels, stride)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(hid_channels, hid_channels)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.downsample = downsample

        if use_senet:
            self.senet = SENet(out_channels, ratio)
        else:
            self.senet = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # SENet
        if self.senet is not None:
            out = self.senet(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        layers,
        num_classes=1,
        input_channel_num=4,  # depth, pose, mask, original image
        use_senet=False,
        ratio=16,
    ):
        super(ResNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.input_channel_num = input_channel_num

        self.layers = layers
        self.in_channels = 64 + self.input_channel_num
        self.use_senet = use_senet
        self.ratio = ratio

        self.conv1 = nn.Conv2d(
            in_channels=self.input_channel_num,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block = BasicBlock
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.conv2 = self.get_layers(block, 64, self.layers[0])
        self.avg_pool3 = nn.AdaptiveAvgPool2d(2)
        self.conv3 = self.get_layers(block, 128, self.layers[1], stride=2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(2)
        self.conv4 = self.get_layers(block, 256, self.layers[2], stride=2)
        self.avg_pool5 = nn.AdaptiveAvgPool2d(2)
        self.conv5 = self.get_layers(block, 512, self.layers[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        torch.nn.init.kaiming_normal(self.fc.weight)
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, hid_channels, n_layers, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != hid_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, hid_channels * block.expansion, stride),
                nn.BatchNorm2d(hid_channels * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels,
                hid_channels,
                self.use_senet,
                self.ratio,
                stride,
                downsample,
            )
        )
        self.in_channels = hid_channels * block.expansion + self.input_channel_num

        for _ in range(1, n_layers):
            layers.append(
                block(self.in_channels, hid_channels, self.use_senet, self.ratio)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Example tensor shape based on resnet101
        """
        original_x = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxPool(x)

        origial_x = self.avg_pool2(original_x)
        x = torch.cat((x, origial_x), 1)
        x = self.conv2(x)
        origial_x = self.avg_pool3(original_x)
        x = torch.cat((x, origial_x), 1)
        x = self.conv3(x)
        origial_x = self.avg_pool4(original_x)
        x = torch.cat((x, origial_x), 1)
        x = self.conv4(x)
        origial_x = self.avg_pool5(original_x)
        x = torch.cat((x, origial_x), 1)
        x = self.conv5(x)

        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    return ResNet(False, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(False, [3, 4, 6, 3], **kwargs)
