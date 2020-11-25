"""
A Neural Algorithm of Artistic Style (Image Style Transfer Using Convolutional Neural Networks)


REFERENCE

    [PyTorch]       https://github.com/rrmina/neural-style-pytorch
    [Torch]         https://github.com/jcjohnson/neural-style
    [TensorFlow]    https://github.com/anishathalye/neural-style
    [PyTorch]       https://github.com/ProGamerGov/neural-style-pt
    [PyTorch]       https://pytorch.org/tutorials/advanced/neural_style_tutorial.html


PRETRAINED MODELS

    1 Caffe 2 PyTorch
        https://github.com/jcjohnson/pytorch-vgg

        vgg16: https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth
        vgg19: https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth

        # mode="BGR"
        # color=[0, 255]
        # mean=[103.939, 116.779, 123.68]
        # std=[1, 1, 1]

    2 Origin PyTorch
        https://pytorch.org/docs/stable/torchvision/models.html
        https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

        vgg16: https://download.pytorch.org/models/vgg16-397923af.pth
        vgg19: https://download.pytorch.org/models/vgg19-dcbb9e9d.pth

        # mode="RGB"
        # color=[0, 1]
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225]

        Note: This model can not converge well


LOSS FUNCTIONS

    1. All loss functions are basically Mean of the Squared Errors (MSE)

    2. Total Variation(TV) Loss
        The total variation (TV) loss encourages spatial smoothness in the generated image.
        It was not used by Gatys et al in their CVPR paper but it can sometimes improve the results;
        For more details and explanation see Mahendran and Vedaldi "Understanding Deep Image Representations by Inverting Them" CVPR 2015.
        https://en.wikipedia.org/wiki/Total_variation_denoising


OPTIMIZER

    LBFGS
        https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS


INPUT SIZE

    1. Origin Paper
        resized the style image to the same size as the content image

    2. PyTorch Repo
        initialize target image to the same size as the content image
        do not change style image but normalize the gram matrix (devided by height and width) to keep same magnitude

"""

import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from tqdm import tqdm

from criterion import ContentLoss, calc_gram_matrice, StyleLoss, TVLoss
from data import PreProcessor, PostProcessor, transfer_color
from model import Encoder


torch.backends.cudnn.benchmark = True


def main(args):

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        print("mkdir => %s" % args.save_dir)

    encoder = Encoder(args.pretrained_path).cuda().eval()

    assert len(args.size) in [1, 2]

    pre_processor = PreProcessor(
        ctype=args.ctype, div=args.div, mean=args.mean, std=args.std,
        size=tuple(args.size) if len(args.size) == 2 else args.size[0],
        init_mode=args.init_mode
    )
    post_processor = PostProcessor(
        ctype=args.ctype, div=args.div, mean=args.mean, std=args.std
    )

    content_criterion = ContentLoss()
    style_criterion = StyleLoss()
    tv_criterion = TVLoss()

    for content_image_path, style_image_path in zip(args.content_image_paths, args.style_image_paths):

        save_name = "%s_&_%s" % (
            os.path.basename(content_image_path).split(".")[0],
            os.path.basename(style_image_path).split(".")[0]
        )
        print("[name] %s" % save_name)

        shutil.copy(content_image_path, "%s/%s-c.jpg" % (args.save_dir, save_name))
        shutil.copy(style_image_path, "%s/%s-s.jpg" % (args.save_dir, save_name))

        content_input, style_input, transfer_input = pre_processor(content_image_path, style_image_path)

        content_inputs = content_input.unsqueeze(0).cuda(non_blocking=True)
        style_inputs = style_input.unsqueeze(0).cuda(non_blocking=True)
        transfer_inputs = transfer_input.unsqueeze(0).cuda(non_blocking=True).requires_grad_()

        transfer_image = post_processor(transfer_inputs.detach().cpu().squeeze())
        transfer_image.save("%s/%s-t-init.jpg" % (args.save_dir, save_name))

        if args.optim == "Adam":
            optimizer = optim.Adam([transfer_inputs], lr=args.lr)
        else:
            optimizer = optim.LBFGS([transfer_inputs])

        with torch.no_grad():
            content_features = encoder(content_inputs)["content"]
            style_features = encoder(style_inputs)["style"]
            style_gram_matrice = {}
            for layer, features in style_features.items():
                style_gram_matrice[layer] = calc_gram_matrice(features)

        with tqdm(total=args.num_iters) as pbar:
            for iter_idx in range(args.num_iters):

                def closure():
                    transfer_features = encoder(transfer_inputs)
                    transfer_content_features = transfer_features["content"]
                    transfer_style_gram_matrice = {}
                    for layer, features in transfer_features["style"].items():
                        transfer_style_gram_matrice[layer] = calc_gram_matrice(features)

                    content_loss = 0
                    for layer in content_features:
                        content_loss += content_criterion(transfer_content_features[layer], content_features[layer])

                    style_loss = 0
                    for layer in style_gram_matrice:
                        style_loss += style_criterion(transfer_style_gram_matrice[layer], style_gram_matrice[layer])

                    tv_loss = tv_criterion(transfer_inputs)

                    content_loss = args.content_weight * content_loss
                    style_loss = args.style_weight * style_loss
                    tv_loss = args.tv_weight * tv_loss

                    loss = content_loss + style_loss + tv_loss

                    pbar.set_postfix(
                        loss = "%.2E" % loss.item(),
                        loss_c = "%.2E" % content_loss.item(),
                        loss_s = "%.2E" % style_loss.item(),
                        loss_tv = "%.2E" % tv_loss.item(),
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    return loss

                optimizer.step(closure)

                if (iter_idx + 1) % args.num_iters_save == 0 or (iter_idx + 1) == args.num_iters:
                    transfer_image = post_processor(transfer_inputs.detach().cpu().squeeze())
                    transfer_image.save("%s/%s-t-origin-iter_%03d.jpg" % (args.save_dir, save_name, iter_idx+1))
                    if args.preserve_color:
                        content_image = post_processor(content_input)
                        transfer_image_preserved = transfer_color(transfer_image, content_image)
                        transfer_image_preserved.save("%s/%s-t-preserved-iter_%03d.jpg" % (args.save_dir, save_name, iter_idx+1))

                pbar.update(1)

        transfer_image = post_processor(transfer_inputs.detach().cpu().squeeze())
        transfer_image.save("%s/%s-t-origin-final.jpg" % (args.save_dir, save_name))
        if args.preserve_color:
            content_image = post_processor(content_input)
            transfer_image_preserved = transfer_color(transfer_image, content_image)
            transfer_image_preserved.save("%s/%s-t-preserved-final.jpg" % (args.save_dir, save_name))
    return


if __name__ == "__main__":


    """

python -m main

    """

    import argparse

    # NOTE Caffe Model & Adam
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--content_image_paths", type=str, nargs="+", default=[
    #     "./src/images/content/sailboat.jpg",
    #     "./src/images/content/cornell.jpg",
    #     "./src/images/content/lenna.jpg",
    #     "./src/images/content/brad_pitt.jpg",
    #     "./src/images/content/golden_bridge.jpg",
    #     "./src/images/content/janelle_monae.jpg",
    # ])
    # parser.add_argument("--style_image_paths", type=str, nargs="+", default=[
    #     "./src/images/style/sketch.jpg",
    #     "./src/images/style/woman_with_hat_matisse.jpg",
    #     "./src/images/style/picasso_seated_nude_hr.jpg",
    #     "./src/images/style/picasso_self_portrait.jpg",
    #     "./src/images/style/la_muse.jpg",
    #     "./src/images/style/starry_night.jpg",
    # ])
    # parser.add_argument("--pretrained_path", type=str, default="./src/models/vgg19-d01eb7cb-caffe.pth")
    # parser.add_argument("--ctype", type=str, default="BGR")
    # parser.add_argument("--div", type=float, default=1)
    # parser.add_argument("--mean", type=float, nargs="+", default=[103.939, 116.779, 123.68])
    # parser.add_argument("--std", type=float, nargs="+", default=[1, 1, 1])
    # parser.add_argument("--size", type=float, nargs="+", default=[512, 512])
    # parser.add_argument("--init_mode", type=str, default="random")
    # parser.add_argument("--optim", type=str, default="Adam")
    # parser.add_argument("--lr", type=float, default=10)
    # parser.add_argument("--num_iters", type=int, default=500)
    # parser.add_argument("--num_iters_save", type=int, default=100)
    # parser.add_argument("--content_weight", type=float, default=5)
    # parser.add_argument("--style_weight", type=float, default=20)
    # parser.add_argument("--tv_weight", type=float, default=0.001)
    # parser.add_argument("--save_dir", type=str, default="./tmp/images_caffe_adam")
    # parser.add_argument("--preserve_color", action="store_true")

    # NOTE Caffe Model & LBFGS
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--content_image_paths", type=str, nargs="+", default=[
    #     "./src/images/content/sailboat.jpg",
    #     "./src/images/content/cornell.jpg",
    #     "./src/images/content/lenna.jpg",
    #     "./src/images/content/brad_pitt.jpg",
    #     "./src/images/content/golden_bridge.jpg",
    #     "./src/images/content/janelle_monae.jpg",
    # ])
    # parser.add_argument("--style_image_paths", type=str, nargs="+", default=[
    #     "./src/images/style/sketch.jpg",
    #     "./src/images/style/woman_with_hat_matisse.jpg",
    #     "./src/images/style/picasso_seated_nude_hr.jpg",
    #     "./src/images/style/picasso_self_portrait.jpg",
    #     "./src/images/style/la_muse.jpg",
    #     "./src/images/style/starry_night.jpg",
    # ])
    # parser.add_argument("--pretrained_path", type=str, default="./src/models/vgg19-d01eb7cb-caffe.pth")
    # parser.add_argument("--ctype", type=str, default="BGR")
    # parser.add_argument("--div", type=float, default=1)
    # parser.add_argument("--mean", type=float, nargs="+", default=[103.939, 116.779, 123.68])
    # parser.add_argument("--std", type=float, nargs="+", default=[1, 1, 1])
    # parser.add_argument("--size", type=float, nargs="+", default=[512, 512])
    # parser.add_argument("--optim", type=str, default="LBFGS")
    # parser.add_argument("--init_mode", type=str, default="random")
    # parser.add_argument("--lr", type=float, default=None)
    # parser.add_argument("--num_iters", type=int, default=25)
    # parser.add_argument("--num_iters_save", type=int, default=5)
    # parser.add_argument("--content_weight", type=float, default=5)
    # parser.add_argument("--style_weight", type=float, default=20)
    # parser.add_argument("--tv_weight", type=float, default=0.001)
    # parser.add_argument("--save_dir", type=str, default="./tmp/images_caffe_lbfgs")
    # parser.add_argument("--preserve_color", action="store_true")

    # NOTE Caffe Model & LBFGS (Face)
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image_paths", type=str, nargs="+", default=[
        "./tmp/test_c.jpg"
    ])
    parser.add_argument("--style_image_paths", type=str, nargs="+", default=[
        "./tmp/test_s.jpg",
    ])
    parser.add_argument("--pretrained_path", type=str, default="./src/models/vgg19-d01eb7cb-caffe.pth")
    parser.add_argument("--ctype", type=str, default="BGR")
    parser.add_argument("--div", type=float, default=1)
    parser.add_argument("--mean", type=float, nargs="+", default=[103.939, 116.779, 123.68])
    parser.add_argument("--std", type=float, nargs="+", default=[1, 1, 1])
    parser.add_argument("--size", type=float, nargs="+", default=[1024, 1024])
    parser.add_argument("--optim", type=str, default="LBFGS")
    parser.add_argument("--init_mode", type=str, default="content")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_iters", type=int, default=600)
    parser.add_argument("--num_iters_save", type=int, default=5)
    parser.add_argument("--content_weight", type=float, default=5)
    parser.add_argument("--style_weight", type=float, default=10000)
    parser.add_argument("--tv_weight", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default="./tmp/test")
    parser.add_argument("--preserve_color", action="store_true")

    # NOTE PyToch Model & Adam [Not Converge Well]
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--content_image_paths", type=str, nargs="+", default=[
    #     "./src/images/content/sailboat.jpg",
    #     "./src/images/content/cornell.jpg",
    #     "./src/images/content/lenna.jpg",
    #     "./src/images/content/brad_pitt.jpg",
    #     "./src/images/content/golden_bridge.jpg",
    #     "./src/images/content/janelle_monae.jpg",
    # ])
    # parser.add_argument("--style_image_paths", type=str, nargs="+", default=[
    #     "./src/images/style/sketch.jpg",
    #     "./src/images/style/woman_with_hat_matisse.jpg",
    #     "./src/images/style/picasso_seated_nude_hr.jpg",
    #     "./src/images/style/picasso_self_portrait.jpg",
    #     "./src/images/style/la_muse.jpg",
    #     "./src/images/style/starry_night.jpg",
    # ])
    # parser.add_argument("--pretrained_path", type=str, default="./src/models/vgg19-dcbb9e9d-pytorch.pth")
    # parser.add_argument("--ctype", type=str, default="RGB")
    # parser.add_argument("--div", type=float, default=255)
    # parser.add_argument("--mean", type=float, nargs="+", default=[0.485, 0.456, 0.406])
    # parser.add_argument("--std", type=float, nargs="+", default=[0.229, 0.224, 0.225])
    # parser.add_argument("--size", type=float, nargs="+", default=[512, 512])
    # parser.add_argument("--optim", type=str, default="Adam")
    # parser.add_argument("--lr", type=float, default=10)
    # parser.add_argument("--num_iters", type=int, default=500)
    # parser.add_argument("--num_iters_save", type=int, default=100)
    # parser.add_argument("--content_weight", type=float, default=5)
    # parser.add_argument("--style_weight", type=float, default=20)
    # parser.add_argument("--tv_weight", type=float, default=0.001)
    # parser.add_argument("--save_dir", type=str, default="./tmp/images_pytorch_adam")

    args = parser.parse_args()

    main(args)
