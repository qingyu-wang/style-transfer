"""

Reference Paper:

    Perceptual Losses for Real-Time Style Transfer and Super-Resolution
    Perceptual Losses for Real-Time Style Transfer and Super-Resolution: Supplementary Material

Reference Repo:

    https://github.com/jcjohnson/fast-neural-style
    https://github.com/jcjohnson/fast-neural-style/issues/63

Note 1:

    last conv block should use "Sigmoid" ("TanH" in paper) as activation function instead of "ReLU"
    because we need to conver the output value to [0, 1] like an image
    after normalization, the color range of the output are same as the target (ground truth)

Note 2:

    because the last layer of model is "Sigmoid" ("TanH" in paper), the value in 0 and 1 are hard to reach
    so there exist sort of color shift, which could be reduced by hitogram matching while input image is the reference

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ConvBlockSigmoid(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlockSigmoid, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x_res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_res
        return x


class DeconvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class SuperResolutionX4(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(SuperResolutionX4, self).__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4) # (W, H) -> (W, H)
        self.res1 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.res2 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.res3 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.res4 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.deconv1 = DeconvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # (W, H) -> (2W, 2H)
        self.deconv2 = DeconvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # (W, H) -> (2W, 2H)
        self.conv2 = ConvBlockSigmoid(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4) # (W, H) -> (W, H)

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)   # ( 3,  72,  72) -> (64,  72,  72)
        x = self.res1(x)    # (64,  72,  72) -> (64,  72,  72)
        x = self.res2(x)    # (64,  72,  72) -> (64,  72,  72)
        x = self.res3(x)    # (64,  72,  72) -> (64,  72,  72)
        x = self.res4(x)    # (64,  72,  72) -> (64,  72,  72)
        x = self.deconv1(x) # (64,  72,  72) -> (64, 144, 144)
        x = self.deconv2(x) # (64, 144, 144) -> (64, 288, 288)
        x = self.conv2(x)   # (64, 288, 288) -> ( 3, 288, 288)
        x = x - self.mean
        x = x / self.std
        return x


class SuperResolutionX8(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(SuperResolutionX8, self).__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4) # (W, H) -> (W, H)
        self.res1 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.res2 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.res3 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.res4 = ResidualBlock(channels=64) # (W, H) -> (W, H)
        self.deconv1 = DeconvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # (W, H) -> (2W, 2H)
        self.deconv2 = DeconvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # (W, H) -> (2W, 2H)
        self.deconv3 = DeconvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # (W, H) -> (2W, 2H)
        self.conv2 = ConvBlockSigmoid(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4) # (W, H) -> (W, H)

        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)   # ( 3,  36,  36) -> (64,  36,  36)
        x = self.res1(x)    # (64,  36,  36) -> (64,  36,  36)
        x = self.res2(x)    # (64,  36,  36) -> (64,  36,  36)
        x = self.res3(x)    # (64,  36,  36) -> (64,  36,  36)
        x = self.res4(x)    # (64,  36,  36) -> (64,  36,  36)
        x = self.deconv1(x) # (64,  36,  36) -> (64,  72,  72)
        x = self.deconv2(x) # (64,  72,  72) -> (64, 144, 144)
        x = self.deconv3(x) # (64, 144, 144) -> (64, 288, 288)
        x = self.conv2(x)   # (64, 288, 288) -> ( 3, 288, 288)
        x = x - self.mean
        x = x / self.std
        return x


class VGG16Features(torch.nn.Module):

    """VGG16Features

    models.vgg16().features

    conv1_1 (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu1_1 (1): ReLU(inplace=True)
    conv1_2 (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu1_2 (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv2_1 (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu2_1 (6): ReLU(inplace=True)
    conv2_2 (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu2_2 (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv3_1 (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu3_1 (11): ReLU(inplace=True)
    conv3_2 (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu3_2 (13): ReLU(inplace=True)
    conv3_3 (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu3_3 (15): ReLU(inplace=True)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv4_1 (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu4_1 (18): ReLU(inplace=True)
    conv4_2 (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu4_2 (20): ReLU(inplace=True)
    conv4_3 (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu4_3 (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv5_1 (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu5_1 (25): ReLU(inplace=True)
    conv5_2 (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu5_2 (27): ReLU(inplace=True)
    conv5_3 (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu5_3 (29): ReLU(inplace=True)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    """

    def __init__(self, pretrained=True, state_dict=None):
        super(VGG16Features, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        if state_dict:
            vgg16.load_state_dict(state_dict)

        self.relu1_2 = torch.nn.Sequential()
        for x in range(0, 4):
            self.relu1_2.add_module(str(x), vgg16.features[x])

        self.relu2_2 = torch.nn.Sequential()
        for x in range(4, 9):
            self.relu2_2.add_module(str(x), vgg16.features[x])

        self.relu3_3 = torch.nn.Sequential()
        for x in range(9, 16):
            self.relu3_3.add_module(str(x), vgg16.features[x])

        self.relu4_3 = torch.nn.Sequential()
        for x in range(16, 23):
            self.relu4_3.add_module(str(x), vgg16.features[x])

        self.relu5_3 = torch.nn.Sequential()
        for x in range(23, 30):
            self.relu5_3.add_module(str(x), vgg16.features[x])

    def forward(self, x):
        x_relu1_2 = self.relu1_2(x)
        x_relu2_2 = self.relu2_2(x_relu1_2)
        x_relu3_3 = self.relu3_3(x_relu2_2)
        x_relu4_3 = self.relu4_3(x_relu3_3)
        x_relu5_3 = self.relu5_3(x_relu4_3)
        out = {
            "relu1_2": x_relu1_2,
            "relu2_2": x_relu2_2,
            "relu3_3": x_relu3_3,
            "relu4_3": x_relu4_3,
            "relu5_3": x_relu5_3
        }
        return out


class VGG19Features(torch.nn.Module):

    """VGG19Features

    models.vgg19().features

    conv1_1 (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu1_1 (1): ReLU(inplace=True)
    conv1_2 (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu1_2 (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv2_1 (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu2_1 (6): ReLU(inplace=True)
    conv2_2 (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu2_2 (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv3_1 (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu3_1 (11): ReLU(inplace=True)
    conv3_2 (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu3_2 (13): ReLU(inplace=True)
    conv3_3 (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu3_3 (15): ReLU(inplace=True)
    conv3_4 (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu3_4 (17): ReLU(inplace=True)
            (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv4_1 (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu4_1 (20): ReLU(inplace=True)
    conv4_2 (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu4_2 (22): ReLU(inplace=True)
    conv4_3 (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu4_3 (24): ReLU(inplace=True)
    conv4_4 (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu4_4 (26): ReLU(inplace=True)
            (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    conv5_1 (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu5_1 (29): ReLU(inplace=True)
    conv5_2 (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu5_2 (31): ReLU(inplace=True)
    conv5_3 (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu5_3 (33): ReLU(inplace=True)
    conv5_4 (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    relu5_4 (35): ReLU(inplace=True)
            (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    """

    def __init__(self, pretrained=True, state_dict=None):
        super(VGG19Features, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        if state_dict:
            vgg19.load_state_dict(state_dict)

        self.relu1_2 = torch.nn.Sequential()
        for x in range(0, 4):
            self.relu1_2.add_module(str(x), vgg19.features[x])

        self.relu2_2 = torch.nn.Sequential()
        for x in range(4, 9):
            self.relu2_2.add_module(str(x), vgg19.features[x])

        self.relu3_4 = torch.nn.Sequential()
        for x in range(9, 18):
            self.relu3_4.add_module(str(x), vgg19.features[x])

        self.relu4_4 = torch.nn.Sequential()
        for x in range(18, 27):
            self.relu4_4.add_module(str(x), vgg19.features[x])

        self.relu5_4 = torch.nn.Sequential()
        for x in range(27, 36):
            self.relu5_4.add_module(str(x), vgg19.features[x])

    def forward(self, x):
        x_relu1_2 = self.relu1_2(x)
        x_relu2_2 = self.relu2_2(x_relu1_2)
        x_relu3_4 = self.relu3_4(x_relu2_2)
        x_relu4_4 = self.relu4_4(x_relu3_4)
        x_relu5_4 = self.relu5_4(x_relu4_4)
        out = {
            "relu1_2": x_relu1_2,
            "relu2_2": x_relu2_2,
            "relu3_4": x_relu3_4,
            "relu4_4": x_relu4_4,
            "relu5_4": x_relu5_4
        }
        return out


if __name__ == "__main__":

    """

python -m model

    """

    # SuperResolutionX4
    print("=" *80)
    print("SuperResolutionX4")

    inputs = torch.rand((1, 3, 72, 72))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    inputs = (inputs - mean) / std
    print()
    print(inputs.shape)

    model = SuperResolutionX4(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(x):
        x = model.conv1(x)   # ( 3,  72,  72) -> (64,  72,  72)
        print(x.shape)
        x = model.res1(x)    # (64,  72,  72) -> (64,  72,  72)
        print(x.shape)
        x = model.res2(x)    # (64,  72,  72) -> (64,  72,  72)
        print(x.shape)
        x = model.res3(x)    # (64,  72,  72) -> (64,  72,  72)
        print(x.shape)
        x = model.res4(x)    # (64,  72,  72) -> (64,  72,  72)
        print(x.shape)
        x = model.deconv1(x) # (64,  72,  72) -> (64, 144, 144)
        print(x.shape)
        x = model.deconv2(x) # (64, 144, 144) -> (64, 288, 288)
        print(x.shape)
        x = model.conv2(x)   # (64, 288, 288) -> ( 3, 288, 288)
        print(x.shape)
        x = x - model.mean
        x = x / model.std
        return x
    model.forward = forward

    outputs = model(inputs)
    print(outputs.shape)

    print()
    print("inputs:  [%.3f, %.3f]" % (
        inputs.detach().cpu().numpy().min(),
        inputs.detach().cpu().numpy().max()
    ))
    print("outputs: [%.3f, %.3f]" % (
        outputs.detach().cpu().numpy().min(),
        outputs.detach().cpu().numpy().max()
    ))

    # SuperResolutionX8
    print("=" *80)
    print("SuperResolutionX8")

    inputs = torch.rand((1, 3, 36, 36))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    inputs = (inputs - mean) / std
    print()
    print(inputs.shape)

    print()
    print("inputs:  [%.3f, %.3f]" % (
        inputs.detach().cpu().numpy().min(),
        inputs.detach().cpu().numpy().max()
    ))
    print("outputs: [%.3f, %.3f]" % (
        outputs.detach().cpu().numpy().min(),
        outputs.detach().cpu().numpy().max()
    ))

    model = SuperResolutionX8()
    def forward(x):
        x = model.conv1(x)   # ( 3,  36,  36) -> (64,  36,  36)
        print(x.shape)
        x = model.res1(x)    # (64,  36,  36) -> (64,  36,  36)
        print(x.shape)
        x = model.res2(x)    # (64,  36,  36) -> (64,  36,  36)
        print(x.shape)
        x = model.res3(x)    # (64,  36,  36) -> (64,  36,  36)
        print(x.shape)
        x = model.res4(x)    # (64,  36,  36) -> (64,  36,  36)
        print(x.shape)
        x = model.deconv1(x) # (64,  36,  36) -> (64,  72,  72)
        print(x.shape)
        x = model.deconv2(x) # (64,  72,  72) -> (64, 144, 144)
        print(x.shape)
        x = model.deconv3(x) # (64, 144, 144) -> (64, 288, 288)
        print(x.shape)
        x = model.conv2(x)   # (64, 288, 288) -> ( 3, 288, 288)
        x = x - model.mean
        x = x / model.std
        return x
    model.forward = forward

    outputs = model(inputs)
    print(outputs.shape)

    print()
    print("inputs:  [%.3f, %.3f]" % (
        inputs.detach().cpu().numpy().min(),
        inputs.detach().cpu().numpy().max()
    ))
    print("outputs: [%.3f, %.3f]" % (
        outputs.detach().cpu().numpy().min(),
        outputs.detach().cpu().numpy().max()
    ))
