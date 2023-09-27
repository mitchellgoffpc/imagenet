import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

BN_EPSILON = 1e-5
BN_MOMENTUM = 0.1

def round_channels(channels, multiplier):
    divisor = 8
    channels *= multiplier
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels: # prevent rounding by more than 10%
        new_channels += divisor
    return int(new_channels)

def round_repeats(repeats, multiplier):
    return int(math.ceil(multiplier * repeats))


def ConvNorm(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(OrderedDict({
        'conv': nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        'bn': nn.BatchNorm2d(out_channels, eps=BN_EPSILON, momentum=BN_MOMENTUM)}))

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, squeezed_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1)

    def forward(self, x):
        y = x.mean((2, 3), keepdims=True)
        y = F.silu(self.fc1(y))
        y = F.sigmoid(self.fc2(y))
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.res = stride == 1 and in_channels == out_channels
        self.expand = ConvNorm(in_channels, expanded_channels, kernel_size=1, bias=False) if expand_ratio != 1 else None
        self.depthwise = ConvNorm(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                  padding=(kernel_size-1)//2, groups=expanded_channels, bias=False)
        self.squeeze_excite = SqueezeExcite(expanded_channels, int(in_channels * se_ratio))
        self.project = ConvNorm(expanded_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = F.silu(self.expand(x)) if self.expand else x
        y = F.silu(self.depthwise(y))
        y = self.squeeze_excite(y)
        y = self.project(y)
        return x + y if self.res else y


class EfficientNet(nn.Module):
    BLOCKS = [
        (1, 16, 3, 1, 1),
        (2, 24, 3, 2, 6),
        (2, 40, 5, 2, 6),
        (3, 80, 3, 2, 6),
        (3, 112, 5, 1, 6),
        (4, 192, 5, 2, 6),
        (1, 320, 3, 1, 6)]

    CONFIGS = {
        0: (1.0, 1.0),
        1: (1.0, 1.1),
        2: (1.1, 1.2),
        3: (1.2, 1.4),
        4: (1.4, 1.8),
        5: (1.6, 2.2),
        6: (1.8, 2.6),
        7: (2.0, 3.1)}

    def __init__(self, size:int):
        super().__init__()
        assert size in EfficientNet.CONFIGS, f"Invalid size ({size}), choices are {list(EfficientNet.CONFIGS.keys())}"
        width_multiplier, depth_multiplier = EfficientNet.CONFIGS[size]
        layer_channels = [32] + [c for _,c,_,_,_ in EfficientNet.BLOCKS] + [1280]
        stem_channels, *block_channels, head_channels = [round_channels(c, width_multiplier) for c in layer_channels]
        self.conv_stem = ConvNorm(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)

        layers = []
        in_channels = stem_channels
        for out_channels, (num_repeats, _, kernel_size, stride, expand_ratio) in zip(block_channels, EfficientNet.BLOCKS):
            blocks = [MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, 0.25)]
            blocks += [MBConvBlock(out_channels, out_channels, kernel_size, 1, expand_ratio, 0.25) for _ in range(1, round_repeats(num_repeats, depth_multiplier))]
            in_channels = out_channels
            layers.append(nn.Sequential(*blocks))

        self.layers = nn.Sequential(*layers)
        self.conv_head = ConvNorm(in_channels, head_channels, kernel_size=1, bias=False)
        self.fc = nn.Linear(head_channels, 1000)

    def forward(self, x):
        x = F.silu(self.conv_stem(x))
        for block in self.layers:
            x = block(x)
        x = F.silu(self.conv_head(x))
        x = self.fc(x.mean((2, 3)))
        return x


if __name__ == "__main__":
    import re, torchvision

    for size in range(8):
        if size >= 5:  # lukemelas's models (B5, B6, B7) use non-standard epsilon/momentum parameters
            BN_EPSILON = 1e-3
            BN_MOMENTUM = 0.01

        PretrainedEfficientNet = getattr(torchvision.models, f'efficientnet_b{size}')
        PretrainedEfficientNetWeights = getattr(torchvision.models, f'EfficientNet_B{size}_Weights').DEFAULT
        pretrained_efficientnet = PretrainedEfficientNet(weights=PretrainedEfficientNetWeights).eval()

        replacements = {
            r'^features\.0\.': r'conv_stem.',
            r'^features\.8\.': r'conv_head.',
            r'^features\.([0-9]+)\.': lambda m: f"layers.{int(m.groups()[0]) - 1}.",
            r'^layers\.0\.([0-9]+)\.block\.([0-9]+)\.': lambda m: f"layers.0.{m.groups()[0]}.block.{int(m.groups()[1]) + 1}.",
            r'\.block\.0\.': r'.expand.',
            r'\.block\.1\.': r'.depthwise.',
            r'\.block\.2\.': r'.squeeze_excite.',
            r'\.block\.3\.': r'.project.',
            r'^classifier\.1\.': r'fc.',
            r'\.0\.([a-z_]+)$': r'.conv.\1',
            r'\.1\.([a-z_]+)$': r'.bn.\1',
        }

        state_dict = pretrained_efficientnet.state_dict()
        for r,s in replacements.items():
            state_dict = {re.sub(r, s, k): v for k,v in state_dict.items()}

        custom_efficientnet = EfficientNet(size).eval()
        custom_efficientnet.load_state_dict(state_dict)

        print(f"Testing EfficientNet-B{size}...")
        x = torch.randn(1, 3, 224, 224)
        a = pretrained_efficientnet(x)
        b = custom_efficientnet(x)
        torch.testing.assert_close(a, b)
        print("Looks good!")
