"""

Paper:
    We use reflection padding in both encoder and decoder to avoid border artifacts.


Code:
    https://github.com/xunhuang1995/AdaIN-style/blob/master/models/download_models.sh#L2-L5
        # The VGG-19 network is obtained by:
        # 1. converting vgg_normalised.caffemodel to .t7 using loadcaffe
        # 2. inserting a convolutional module at the beginning to preprocess the image (no need in pytorch)
        # 3. replacing zero-padding with reflection-padding

    https://github.com/xunhuang1995/AdaIN-style/blob/master/lib/ArtisticStyleLossCriterion.lua

    https://github.com/xunhuang1995/AdaIN-style/blob/master/lib/AdaptiveInstanceNormalization.lua

Model Structure:
    ./src/model_structure/vgg_normalised.t7.png

"""

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):

    """Encoder

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

    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        vgg19 = models.vgg19(pretrained=True)

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.net = torch.nn.Sequential()
        for layer_idx, layer in enumerate(vgg19.features):
            if layer_idx == 21: # stop at relu4_1
                break
            if isinstance(layer, nn.Conv2d): # replacing zero-padding with reflection-padding
                layer.padding_mode = "reflect"
            self.net.add_module(str(layer_idx), layer)

        self.feature_info = {
            1:  "relu1_1",
            6:  "relu2_1",
            11: "relu3_1",
            20: "relu4_1",
        }

    def forward(self, x, intermediate=False):
        if intermediate:
            out = {}
            x = self.pad(x)
            for layer_idx, layer in enumerate(self.net):
                x = layer(x)
                if layer_idx in self.feature_info:
                    name = self.feature_info[layer_idx]
                    out[name] = x
        else:
            x = self.pad(x)
            for layer_idx, layer in enumerate(self.net):
                x = layer(x)
            out = x
        return out


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, eps=1e-05):
        super(AdaptiveInstanceNorm, self).__init__()
        self.register_buffer("eps", torch.tensor(eps))

    def calc_mean_std(self, tensor):
        _N, _C, _H, _W = tensor.shape
        var = tensor.view(_N, _C, -1).var(dim=2) + self.eps # channel-wise
        std = var.sqrt().view(_N, _C, 1, 1)
        mean = tensor.view(_N, _C, -1).mean(dim=2).view(_N, _C, 1, 1)
        return mean, std

    def forward(self, content_features, style_features):
        assert content_features.shape == style_features.shape, "content_features.shape: %s, style_features.shape: %s" % (
            content_features.shape, style_features.shape
        )
        # Calculate Mean & Standard Deviation
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        # Calculate Adaptive Feature
        adaptive_features = (content_features - content_mean.expand(content_features.shape)) / content_std.expand(content_features.shape)
        adaptive_features = adaptive_features * style_std.expand(style_features.shape) + style_mean.expand(style_features.shape)
        return adaptive_features


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="reflect")
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = self.net(x)
        return out


class AdaptiveInstanceNormModel(nn.Module):

    def __init__(self, encoder_pretrained=True, eps=1e-05, decoder_state_dict=None):
        super(AdaptiveInstanceNormModel, self).__init__()
        self.encoder = Encoder(pretrained=encoder_pretrained).eval()
        self.ada_in = AdaptiveInstanceNorm(eps=eps).eval()
        self.decoder = Decoder().train()
        
        # Freeze Parameters
        # https://pytorch.org/docs/stable/notes/autograd.html
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.ada_in.parameters():
            param.requires_grad = False

    def forward(self, content_inputs, style_inputs):
        if self.training:
            # Context-manager that disabled gradient calculation.
            # https://pytorch.org/docs/stable/generated/torch.no_grad.html
            with torch.no_grad():
                content_features = self.encoder(content_inputs)
                style_features = self.encoder(style_inputs, intermediate=True)
                adaptive_features = self.ada_in(content_features, style_features["relu4_1"])
            outputs = self.decoder(adaptive_features)
            output_features = self.encoder(outputs, intermediate=True)
            return style_features, adaptive_features, output_features
        else:
            # Context-manager that disabled gradient calculation.
            # https://pytorch.org/docs/stable/generated/torch.no_grad.html
            with torch.no_grad():
                content_features = self.encoder(content_inputs)
                style_features = self.encoder(style_inputs, intermediate=True)
                adaptive_features = self.ada_in(content_features, style_features["relu4_1"])
                outputs = self.decoder(adaptive_features)
            return outputs


if __name__ == "__main__":

    """

python -m model

    """

    # Init Inputs
    content_inputs = torch.rand((4, 3, 224, 224))
    style_inputs = torch.rand((4, 3, 224, 224))

    # Init Encoder
    encoder = Encoder()
    def encoder_forward(x):
        out = {}
        for layer_idx, layer in enumerate(encoder.net):
            x = layer(x)
            if layer_idx in encoder.feature_info:
                name = encoder.feature_info[layer_idx]
                out[name] = x
                print("    [OUTPUT] %02d %s" % (layer_idx, layer))
            else:
                print("    [      ] %02d %s" % (layer_idx, layer))
        return out
    encoder.forward = encoder_forward

    # Init AdaptiveInstanceNorm
    ada_in = AdaptiveInstanceNorm()

    # Init Decoder
    decoder = Decoder()
    def decoder_forward(x):
        for layer_idx, layer in enumerate(decoder.net):
            x = layer(x)
            if layer_idx == (len(decoder.net) - 1):
                print("    [OUTPUT] %02d %s" % (layer_idx, layer))
            else:
                print("    [      ] %02d %s" % (layer_idx, layer))
        return x
    decoder.forward = decoder_forward

    # Forward Encoder (Content Inputs)
    print("=" *80)
    print("Forward Encoder (Content Inputs)")
    content_features = encoder(content_inputs)
    for k, v in content_features.items():
        print("content_features[\"%s\"].shape = %s" % (k, v.shape))

    # Forward Encoder (Style Inputs)
    print("=" *80)
    print("Forward Encoder (Style Inputs)")
    style_features = encoder(style_inputs)
    for k, v in style_features.items():
        print("style_features[\"%s\"].shape = %s" % (k, v.shape))

    # Forward AdaptiveInstanceNorm
    print("=" *80)
    print("Forward AdaptiveInstanceNorm")
    adaptive_features = ada_in(content_features["relu4_1"], style_features["relu4_1"])
    print("adaptive_features.shape = %s" % adaptive_features.shape.__str__())

    # Forward Decoder
    print("=" *80)
    print("Forward Decoder")
    adaptive_outputs = decoder(adaptive_features)
    print("adaptive_outputs.shape = %s" % adaptive_outputs.shape.__str__())

    # Init AdaptiveInstanceNormModel
    print("=" *80)
    print("Init AdaptiveInstanceNormModel")
    model = AdaptiveInstanceNormModel()

    # Forward AdaptiveInstanceNormModel in Train Mode
    print("=" *80)
    print("Forward AdaptiveInstanceNormModel in Train Mode")
    model.train()
    style_features, adaptive_features, output_features = model(content_inputs, style_inputs)
    for k, v in style_features.items():
        print("style_features[\"%s\"].shape = %s" % (k, v.shape))
    print("adaptive_features.shape = %s" % adaptive_features.shape.__str__())
    for k, v in output_features.items():
        print("output_features[\"%s\"].shape = %s" % (k, v.shape))

    # Forward AdaptiveInstanceNormModel in Eval Mode
    print("=" *80)
    print("Forward AdaptiveInstanceNormModel in Eval Mode")
    model.eval()
    outputs = model(content_inputs, style_inputs)
    print("outputs.shape = %s" % outputs.shape.__str__())
