"""

Reference Paper:

    Perceptual Losses for Real-Time Style Transfer and Super-Resolution
    Perceptual Losses for Real-Time Style Transfer and Super-Resolution: Supplementary Material

Reference Repo:

    https://github.com/jcjohnson/fast-neural-style
    https://github.com/jcjohnson/fast-neural-style/issues/63

Note 1:

    we need to preprocess input images normalized in the same way,
    because the torchvision pretrained models which are used for
    extracting image feature are base on this setting.

    https://pytorch.org/docs/stable/torchvision/models.html
        type:  RGB
        range: [0, 1]
        mean:  [0.485, 0.456, 0.406]
        std:   [0.229, 0.224, 0.225]

"""

import glob
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm

from criterion import PixelLoss, ContentLoss, StyleLoss, TVLoss
from dataset import SuperResolutionDataset, PostProcessor
from model import SuperResolutionX4, VGG16Features
from utils import AverageMeter


def main(method, log_dir):

    """
    1w 张 coco 训练数据
    4 batch 20w 轮迭代 => 80 epoch
    """

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
        print("mkdir => %s" % log_dir)

    log_model_dir = "%s/models" % log_dir
    if not os.path.isdir(log_model_dir):
        os.mkdir(log_model_dir)
        print("mkdir => %s" % log_model_dir)

    log_image_dir = "%s/images" % log_dir
    if not os.path.isdir(log_image_dir):
        os.mkdir(log_image_dir)
        print("mkdir => %s" % log_image_dir)

    torch.backends.cudnn.benchmark = True

    train_image_paths = glob.glob("D:/Downloads/coco/style_transfer/train/*")
    train_dataset = SuperResolutionDataset(image_paths=train_image_paths, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], large_size=(288, 288), small_size=(72, 72))
    train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    num_train_data = len(train_dataset)
    num_train_iters = len(train_dataloader)
    print("Init Train Data (data: %d, iters: %d)" % (num_train_data, num_train_iters))

    valid_image_paths = sorted(glob.glob("./src/images/*"))
    valid_dataset = SuperResolutionDataset(image_paths=valid_image_paths, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], large_size=(288, 288), small_size=(72, 72))
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
    num_valid_data = len(valid_dataset)
    num_valid_iters = len(valid_dataloader)
    print("Init Valid Data (data: %d, iters: %d)" % (num_valid_data, num_valid_iters))

    post_processor = PostProcessor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = SuperResolutionX4(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.cuda()
    print("Init Model")

    feature_model = VGG16Features(pretrained=True)
    feature_model.cuda()
    feature_model.eval()
    print("Init Feature Model")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    pixel_criterion = PixelLoss()
    content_criterion = ContentLoss()
    style_criterion = StyleLoss()
    tv_criterion = TVLoss()

    num_epochs = 80
    for epoch_idx in range(num_epochs):
        # Train
        model.train()

        pixel_loss_meter = AverageMeter()
        content_loss_meter = AverageMeter()
        style_loss_meter = AverageMeter()
        tv_loss_meter = AverageMeter()
        loss_meter = AverageMeter()

        with tqdm(desc="train epoch %02d" % epoch_idx, total=num_train_iters) as pbar:
            for inputs, targets in train_dataloader:

                # Forward
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                outputs = model(inputs)

                # print("inputs:  [%.3f, %.3f]" % (
                #     inputs.detach().cpu().numpy().min(),
                #     inputs.detach().cpu().numpy().max()
                # ))
                # print("targets: [%.3f, %.3f]" % (
                #     targets.detach().cpu().numpy().min(),
                #     targets.detach().cpu().numpy().max()
                # ))
                # print("outputs: [%.3f, %.3f]" % (
                #     outputs.detach().cpu().numpy().min(),
                #     outputs.detach().cpu().numpy().max()
                # ))

                # Feature
                with torch.no_grad():
                    target_features = feature_model(targets)
                output_features = feature_model(outputs)

                if method == 1:
                    # Loss
                    content_loss = content_criterion(output_features["relu2_2"], target_features["relu2_2"])
                    pixel_loss = pixel_criterion(outputs, targets)
                    tv_loss = tv_criterion(outputs)
                    # Loss Weight
                    content_loss = 1 * content_loss
                    pixel_loss = 1e-2 * pixel_loss
                    tv_loss = 1e-6 * tv_loss
                    loss = content_loss + pixel_loss + tv_loss
                    # Record
                    content_loss_meter.update(content_loss.item())
                    pixel_loss_meter.update(pixel_loss.item())
                    tv_loss_meter.update(tv_loss.item())
                    loss_meter.update(loss.item())

                elif method == 2:
                    # Loss
                    content_loss = 0
                    content_loss += 0.5 * content_criterion(output_features["relu1_2"], target_features["relu1_2"])
                    content_loss += 0.5 * content_criterion(output_features["relu2_2"], target_features["relu2_2"])
                    pixel_loss = pixel_criterion(outputs, targets)
                    style_loss = 0
                    style_loss += 0.2 * style_criterion(output_features["relu1_2"], target_features["relu1_2"])
                    style_loss += 0.2 * style_criterion(output_features["relu2_2"], target_features["relu2_2"])
                    style_loss += 0.2 * style_criterion(output_features["relu3_3"], target_features["relu3_3"])
                    style_loss += 0.2 * style_criterion(output_features["relu4_3"], target_features["relu4_3"])
                    style_loss += 0.2 * style_criterion(output_features["relu5_3"], target_features["relu5_3"])
                    tv_loss = tv_criterion(outputs)
                    # Loss Weight
                    content_loss = 1 * content_loss
                    pixel_loss = 1e-2 * pixel_loss
                    style_loss = 1e2 * style_loss
                    tv_loss = 1e-6 * tv_loss
                    loss = content_loss + pixel_loss + style_loss + tv_loss
                    # Record
                    content_loss_meter.update(content_loss.item())
                    pixel_loss_meter.update(pixel_loss.item())
                    style_loss_meter.update(style_loss.item())
                    tv_loss_meter.update(tv_loss.item())
                    loss_meter.update(loss.item())

                else:
                    raise ValueError("unknown method %s" % method)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Info
                pbar.set_postfix(
                    loss="%.2E" % loss_meter.avg,
                    loss_p="%.2E" % pixel_loss_meter.avg,
                    loss_c="%.2E" % content_loss_meter.avg,
                    loss_s="%.2E" % style_loss_meter.avg,
                    loss_t="%.2E" % tv_loss_meter.avg,
                    lr="%.2E" % optimizer.param_groups[0]["lr"]
                )
                pbar.update(1)

        # Validate
        log_image_epoch_dir = "%s/epoch_%02d" % (log_image_dir, epoch_idx)
        if not os.path.isdir(log_image_epoch_dir):
            os.mkdir(log_image_epoch_dir)
            print("mkdir => %s" % log_image_epoch_dir)

        model.eval()
        with tqdm(desc="infer", total=num_valid_iters) as pbar:
            data_idx = 0
            for inputs, targets in valid_dataloader:

                # Forward
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                with torch.no_grad():
                    outputs = model(inputs)

                # print("inputs:  [%.3f, %.3f]" % (
                #     inputs.detach().cpu().numpy().min(),
                #     inputs.detach().cpu().numpy().max()
                # ))
                # print("targets: [%.3f, %.3f]" % (
                #     targets.detach().cpu().numpy().min(),
                #     targets.detach().cpu().numpy().max()
                # ))
                # print("outputs: [%.3f, %.3f]" % (
                #     outputs.detach().cpu().numpy().min(),
                #     outputs.detach().cpu().numpy().max()
                # ))

                for input_, target, output in zip(inputs, targets, outputs):

                    image_input = post_processor(input_.detach().cpu())
                    image_target = post_processor(target.detach().cpu())
                    image_output = post_processor(output.detach().cpu())
                    image_output_final = Image.fromarray(
                        match_histograms(
                            image=np.array(image_output),
                            reference=np.array(image_input.resize((288, 288), Image.BICUBIC)),
                            multichannel=True
                        )
                    )

                    image_input.save("%s/image_%02d_input.jpg" % (log_image_epoch_dir, data_idx))
                    image_target.save("%s/image_%02d_target.jpg" % (log_image_epoch_dir, data_idx))
                    image_output.save("%s/image_%02d_output.jpg" % (log_image_epoch_dir, data_idx))
                    image_output_final.save("%s/image_%02d_output_final.jpg" % (log_image_epoch_dir, data_idx))

                    data_idx += 1
                # Info
                pbar.update(1)

        # Save Epoch Model
        save_path = "%s/epoch_%02d.pth" % (log_model_dir, epoch_idx)
        torch.save(model.state_dict(), save_path)
        print("save model => %s" % os.path.abspath(save_path))
        # Save Last Model
        save_path = "%s/last.pth" % log_model_dir
        torch.save(model.state_dict(), save_path)
        print("save model => %s" % os.path.abspath(save_path))
    return


if __name__ == "__main__":

    """

python -m train_beta

    """

    main(method=1, log_dir="./log/train_beta_method_1")
