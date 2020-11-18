"""
A Neural Algorithm of Artistic Style
Image Style Transfer Using Convolutional Neural Networks



Reference:
    [PyTorch]       https://github.com/rrmina/neural-style-pytorch

Other Reference:
    [Torch]         https://github.com/jcjohnson/neural-style
    [TensorFlow]    https://github.com/anishathalye/neural-style
    [PyTorch]       https://github.com/ProGamerGov/neural-style-pt
    [PyTorch]       https://pytorch.org/tutorials/advanced/neural_style_tutorial.html



PRETRAINED MODELS

1 Caffe 2 PyTorch
    https://github.com/jcjohnson/pytorch-vgg
    vgg16: https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth # 0-255
    vgg19: https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth # 0-255

2 Origin PyTorch
    https://pytorch.org/docs/stable/torchvision/models.html
        RGB
        range of [0, 1]
        normalized mean = [0.485, 0.456, 0.406]
        normalized std = [0.229, 0.224, 0.225]
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    vgg16: https://download.pytorch.org/models/vgg16-397923af.pth # 0-1
    vgg19: https://download.pytorch.org/models/vgg19-dcbb9e9d.pth # [-1, -1]

Loss Functions

1. All loss functions are basically Mean of the Squared Errors (MSE)

2. Total Variation(TV) Loss
    The total variation (TV) loss encourages spatial smoothness in the generated image.
    It was not used by Gatys et al in their CVPR paper but it can sometimes improve the results;
    For more details and explanation see Mahendran and Vedaldi "Understanding Deep Image Representations by Inverting Them" CVPR 2015.
    https://en.wikipedia.org/wiki/Total_variation_denoising



Size

1. Origin Paper
    resized the style image to the same size as the content image

2. PyTorch Repo
    initialize target image to the same size as the content image
    do not change style image but normalize the gram matrix (devided by height and width) to keep same magnitude

"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torchvision import models, transforms
from tqdm import tqdm


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(4)
torch.manual_seed(4)
np.random.seed(0)


def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.show()
    return


def image2tensor(image, max_size=512):
    _H, _W, _C = image.shape
    new_size = tuple([int((float(max_size) / max(_H, _W))*x) for x in [_H, _W]])

    transformer = transforms.Compose([
        transforms.ToPILImage(),                                                # cv2 -> PIL
        transforms.Resize(new_size),                                            # resize (long side equal to max_size, keep ratio)
        transforms.ToTensor(),                                                  # PIL [0, 255] (H x W x C) -> tensor [0.0, 1.0] (C x H x W)
        transforms.Lambda(lambda x: x*255),                                     # [0.0, 1.0] -> [0.0, 255.0]
        transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1, 1, 1])    # reduce mean
    ])

    tensor = transformer(image).unsqueeze(dim=0)
    return tensor


def tensor2image(tensor):
    transformer = transforms.Compose([
        transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1, 1, 1]),    # add mean
        transforms.Lambda(lambda x: x.clamp(0, 255))                                # limit range [0, 255]
    ])
    image = transformer(tensor.detach().squeeze().cpu()).numpy().transpose(1, 2, 0).astype(np.uint8)
    return image


def initial_target_tensor(content_tensor, init_image="random"):
    B, C, H, W = content_tensor.shape
    if (init_image=="random"):
        #tensor = torch.randn(C, H, W).mul(torch.std(content_tensor.clone().cpu())/255).unsqueeze(0)
        tensor = torch.randn(C, H, W).mul(0.001).unsqueeze(0)
    else:
        tensor = content_tensor.clone().detach()
    return tensor


def transfer_color(src, dst):
    """
    transfer dst color style to src

    src control content
    dst control color style
    """
    _H, _W, _C = src.shape
    dst = cv2.resize(dst, dsize=(_W, _H), interpolation=cv2.INTER_CUBIC)

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)            #1 Extract the Destination"s luminance
    dst_yiq = cv2.cvtColor(dst, cv2.COLOR_BGR2YCrCb)            #2 Convert the Source from BGR to YIQ/YCbCr
    dst_yiq[..., 0] = src_gray                                  #3 Combine Destination"s luminance and Source"s IQ/CbCr
    dst_transfer = cv2.cvtColor(dst_yiq, cv2.COLOR_YCrCb2BGR)   #4 Convert new image from YIQ back to BGR
    return dst_transfer



def calc_gram_matrix(tensor):
    _B, _C, _H, _W = tensor.shape
    assert _B == 1
    x = tensor.view(_C, _H * _W)
    gram_matrix = torch.mm(x, x.t())
    return gram_matrix


def calc_vgg19_features(vgg19_features_model, inputs):
    """
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
    content_layers = {
        "22" : "relu4_2"
    }
    style_layers = {
        "3":   "relu1_2",
        "8":   "relu2_2",
        "17" : "relu3_3",
        "26" : "relu4_3",
        "35" : "relu5_3"
    }

    output_features = {
        "content": {name: None for idx, name in content_layers.items()},
        "style": {name: None for idx, name in style_layers.items()}
    }

    x = inputs
    for idx, layer in vgg19_features_model._modules.items():
        x = layer(x)

        if idx in content_layers:
            name = content_layers[idx]
            output_features["content"][name] = x

        if idx in style_layers:
            name = style_layers[idx]
            output_features["style"][name] = x

    return output_features


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output_feature, content_feature):
        loss = self.criterion(output_feature, content_feature)
        return loss


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def calc_gram_matrice(self, features, normalize=True):
        _B, _C, _H, _W = features.shape
        x = features.view(_B, _C, _H * _W)
        gram_matrice = torch.bmm(x, x.permute(0, 2, 1))
        if normalize:
            gram_matrice = gram_matrice / (_C * _H * _W)
        return gram_matrice

    def forward(self, output_features, style_feature):
        loss = self.mse_loss(
            self.calc_gram_matrice(output_features),
            self.calc_gram_matrice(style_feature)
        )
        return loss


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, outputs):
        x = outputs[:, :, 1:, :] - outputs[:, :, :-1, :]
        y = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
        loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
        return loss


def main(content_image_paths, style_image_paths, save_dir):

    content_layers = {
        "relu4_2": 1.0
    }
    style_layers = {
        "relu1_2": 0.2,
        "relu2_2": 0.2,
        "relu3_3": 0.2,
        "relu4_3": 0.2,
        "relu5_3": 0.2
    }

    vgg19_model = models.vgg19(pretrained=True)
    vgg19_model.load_state_dict(torch.load("C:/Users/Administrator/.cache/torch/checkpoints/vgg19-d01eb7cb-caffe.pth"), strict=False)
    vgg19_features_model = vgg19_model.features
    vgg19_features_model.eval()

    content_criterion = ContentLoss()
    style_criterion = StyleLoss()
    tv_criterion = TVLoss()

    preserve_color = False

    for content_image_path, style_image_path in zip(content_image_paths, style_image_paths):

        save_name = "%s_&_%s" % (
            os.path.basename(content_image_path).split(".")[0],
            os.path.basename(style_image_path).split(".")[0]
        )

        print("[name] %s" % save_name)

        content_image = cv2.imdecode(np.fromfile(content_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        style_image = cv2.imdecode(np.fromfile(style_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # show_image(content_image)
        # show_image(style_image)

        content_tensor = image2tensor(content_image)
        style_tensor = image2tensor(style_image)
        target_tensor = initial_target_tensor(content_tensor).requires_grad_()

        # optimizer = optim.LBFGS([target_tensor])
        optimizer = optim.Adam([target_tensor], lr=10)

        # NOTE
        # 如果这里不设置no_grad，单独计算其中一个 content_loss / style_loss 都需要设置 retain_graph=True，才能更新参数
        # 因为不设置no_grad， content_features / style_features 和 target_features 都包含梯度，需要动态图来进行反向传播
        # 所以即使只根据一个 content_loss / style_loss，更新梯度，都需要 retain_graph=True，进行两次回传
        with torch.no_grad():
            content_features = calc_vgg19_features(vgg19_features_model, content_tensor)
            style_features = calc_vgg19_features(vgg19_features_model, style_tensor)

        num_iters = 500
        num_iters_show = 100
        with tqdm(total=num_iters) as pbar:
            for iter_idx in range(num_iters):

                target_features = calc_vgg19_features(vgg19_features_model, target_tensor)

                content_loss = 0
                for name, weight in content_layers.items():
                    content_loss += weight * content_criterion(target_features["content"][name], content_features["content"][name])

                style_loss = 0
                for name, weight in style_layers.items():
                    style_loss += weight * style_criterion(target_features["style"][name], style_features["style"][name])

                tv_loss = tv_criterion(target_tensor)

                content_loss = 5e+0 * content_loss
                style_loss = 1e+2 * style_loss
                tv_loss = 1e-3 * tv_loss
                loss = content_loss + style_loss + tv_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                # loss.backward(retain_graph=True)
                optimizer.step()

                if (iter_idx % num_iters_show == 0) or (iter_idx == num_iters):
                    if preserve_color:
                        target_image = transfer_color(tensor2image(target_tensor), tensor2image(content_tensor))
                    else:
                        target_image = tensor2image(target_tensor)
                    # show_image(target_image)
                    cv2.imencode(".jpg", target_image)[1].tofile("%s/%s_iter_%03d.png" % (save_dir, save_name, iter_idx))

                pbar.set_postfix(
                    loss = "%.2E" % loss.item(),
                    loss_c = "%.2E" % content_loss.item(),
                    loss_s = "%.2E" % style_loss.item(),
                    loss_tv = "%.2E" % tv_loss.item(),
                )
                pbar.update(1)

        if preserve_color:
            target_image = transfer_color(tensor2image(target_tensor), tensor2image(content_tensor))
        else:
            target_image = tensor2image(target_tensor)
        # show_image(target_image)
        cv2.imencode(".jpg", target_image)[1].tofile("%s/%s.png" % (save_dir, save_name))
    return



if __name__ == "__main__":


    """

python -m test

iter: 000, loss : 157324.89 (content: 27023.36, style: 130299.84, TV: 1.69)
iter: 100, loss : 16799.85 (content: 3295.63, style: 2168.28, TV: 11335.95)
iter: 200, loss : 14553.54 (content: 2823.83, style: 1463.73, TV: 10265.98)
iter: 300, loss : 13994.63 (content: 2645.79, style: 1355.07, TV: 9993.77)
iter: 400, loss : 13440.50 (content: 2521.04, style: 1301.34, TV: 9618.12)

    """

    content_image_paths = [
        "./src/images/content/sailboat.jpg",
        "./src/images/content/cornell.jpg",
        "./src/images/content/lenna.jpg",
        "./src/images/content/brad_pitt.jpg",
        "./src/images/content/bridge.jpg"
    ]
    style_image_paths = [
        "./src/images/style/sketch.png",
        "./src/images/style/woman_with_hat_matisse.png",
        "./src/images/style/picasso_seated_nude_hr.png",
        "./src/images/style/picasso_self_portrait.png",
        "./src/images/style/la_muse.png"
    ]
    save_dir="./tmp/images"


    main(
        content_image_paths=content_image_paths,
        style_image_paths=style_image_paths,
        save_dir=save_dir
    )
