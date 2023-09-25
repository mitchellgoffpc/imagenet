import torch
import torch.nn as nn
import torch.nn.functional as F


def Downsample(in_channels, out_channels, stride):
    if in_channels == out_channels:
        return nn.Identity()
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * expansion)
        self.downsample = Downsample(in_channels, out_channels * expansion, stride)

    def forward(self, x):
        y = self.bn1(self.conv1(x)).relu()
        y = self.bn2(self.conv2(y))
        return (y + self.downsample(x)).relu()

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        # NOTE: This uses the ResNet V1.5 architecture, so downsampling is done in the 3x3 convs instead of the 1x1s
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        self.downsample = Downsample(in_channels, out_channels * expansion, stride)

    def forward(self, x):
        y = self.bn1(self.conv1(x)).relu()
        y = self.bn2(self.conv2(y)).relu()
        y = self.bn3(self.conv3(y))
        return (y + self.downsample(x)).relu()


class ResNet(nn.Module):
    STRIDES = [1, 2, 2, 2]
    CHANNELS = [64, 128, 256, 512]
    CONFIGS = {
        18: (BasicBlock, 1, [2, 2, 2, 2]),
        34: (BasicBlock, 1, [3, 4, 6, 3]),
        50: (Bottleneck, 4, [3, 4, 6, 3]),
        101: (Bottleneck, 4, [3, 4, 23, 3]),
        152: (Bottleneck, 4, [3, 8, 36, 3])}
    
    def __init__(self, size:int):
        super().__init__()
        assert size in ResNet.CONFIGS, f"Invalid size ({size}), choices are {list(ResNet.CONFIGS.keys())}"
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        layers = []
        in_channels = 64
        Block, expansion, blocks_per_layer = ResNet.CONFIGS[size]
        for out_channels, stride, num_blocks in zip(ResNet.CHANNELS, ResNet.STRIDES, blocks_per_layer):
            blocks = [Block(in_channels, out_channels, stride, expansion)]
            blocks += [Block(out_channels * expansion, out_channels, 1, expansion) for _ in range(1, num_blocks)]
            in_channels = out_channels * expansion
            layers.append(nn.Sequential(*blocks))

        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(512 * expansion, 1000)

    def forward(self, x):
        x = self.bn1(self.conv1(x)).relu()
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = block(x)
        return self.fc(x.mean((2, 3)))


if __name__ == "__main__":
    import torchvision

    for size in ResNet.CONFIGS:
        PretrainedResNet = getattr(torchvision.models, f'resnet{size}')
        PretrainedResNetWeights = getattr(torchvision.models, f'ResNet{size}_Weights').DEFAULT
        pretrained_resnet = PretrainedResNet(weights=PretrainedResNetWeights)
        custom_resnet = ResNet(size)
        custom_resnet.load_state_dict(pretrained_resnet.state_dict())

        print(f"Testing ResNet{size}...")
        x = torch.randn(1, 3, 224, 224)
        a = pretrained_resnet(x)
        b = custom_resnet(x)
        torch.testing.assert_close(a, b)
        print("Looks good!")
