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

    def __init__(self, pretrained_path):
        super(Encoder, self).__init__()
        vgg19 = models.vgg19()
        state_dict = torch.load(pretrained_path, map_location="cpu")
        if "caffe" in pretrained_path:
            state_dict["classifier.0.weight"] = state_dict.pop("classifier.1.weight")
            state_dict["classifier.0.bias"] = state_dict.pop("classifier.1.bias")
            state_dict["classifier.3.weight"] = state_dict.pop("classifier.4.weight")
            state_dict["classifier.3.bias"] = state_dict.pop("classifier.4.bias")
        vgg19.load_state_dict(state_dict)

        self.net = torch.nn.Sequential()
        for layer_idx, layer in enumerate(vgg19.features):
            if layer_idx == 30: # stop at relu5_1
                break
            self.net.add_module(str(layer_idx), layer)

        for param in self.parameters():
            param.requires_grad = False

        self.feature_info = {
            "content": {
                22: "relu4_2"
            },
            "style": {
                1:  "relu1_1",
                6:  "relu2_1",
                11: "relu3_1",
                20: "relu4_1",
                29: "relu5_1"
            }
        }

        # self.feature_info = {
        #     "content": {
        #         21: "conv4_2"
        #     },
        #     "style": {
        #         0:  "conv1_1",
        #         5:  "conv2_1",
        #         10: "conv3_1",
        #         19: "conv4_1",
        #         28: "conv5_1"
        #     }
        # }

    def forward(self, x):
        out = {"content": {}, "style": {}}
        for layer_idx, layer in enumerate(self.net):
            x = layer(x)
            if layer_idx in self.feature_info["content"]:
                out["content"][self.feature_info["content"][layer_idx]] = x
            if layer_idx in self.feature_info["style"]:
                out["style"][self.feature_info["style"][layer_idx]] = x
        return out


if __name__ == "__main__":

    """

python -m model

    """

    # Init Inputs
    inputs = torch.rand((4, 3, 224, 224))

    # Init Encoder
    encoder = Encoder()

    # Forward Encoder
    print("=" *80)
    print("Forward Encoder")
    features = encoder(inputs)
    for k, v in features["content"].items():
        print("features[\"content\"][\"%s\"].shape = %s" % (k, v.shape))
    for k, v in features["style"].items():
        print("features[\"style\"][\"%s\"].shape = %s" % (k, v.shape))
