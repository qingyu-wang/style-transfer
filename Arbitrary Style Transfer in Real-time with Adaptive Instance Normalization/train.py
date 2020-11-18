import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from tqdm import tqdm

from criterion import ContentLoss, StyleLoss
from dataset import AdaptiveInstanceNormDataset, PreProcessor, PostProcessor
from lr_scheduler import LRScheduler
from model import AdaptiveInstanceNormModel
from utils import AverageMeter


Image.MAX_IMAGE_PIXELS = None           # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def main(args):

    torch.backends.cudnn.benchmark = True

    # Init Path
    print("Init Path")
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
        print("mkdir => %s" % args.log_dir)

    log_model_dir = "%s/models" % args.log_dir
    if not os.path.isdir(log_model_dir):
        os.mkdir(log_model_dir)
        print("mkdir => %s" % log_model_dir)

    log_image_dir = "%s/images" % args.log_dir
    if not os.path.isdir(log_image_dir):
        os.mkdir(log_image_dir)
        print("mkdir => %s" % log_image_dir)

    log_write_dir = "%s/writes" % args.log_dir
    if not os.path.isdir(log_image_dir):
        os.mkdir(log_image_dir)
        print("mkdir => %s" % log_image_dir)

    # Init Writer
    print("Init Writer")
    writer = SummaryWriter(log_dir=str(log_write_dir))

    # Init Model
    print("Init Model")
    model = AdaptiveInstanceNormModel().cuda().train()

    # Init Train Data
    print("Init Train Data")
    train_content_image_paths = []
    for train_content_image_pattern in sorted(args.train_content_image_patterns):
        temp_image_paths = sorted(glob.glob(train_content_image_pattern))
        print("    train content <= [%d] %s" % (len(temp_image_paths), train_content_image_pattern))
        train_content_image_paths.extend(temp_image_paths)
    train_style_image_paths = []
    for train_style_image_pattern in sorted(args.train_style_image_patterns):
        temp_image_paths = sorted(glob.glob(train_style_image_pattern))
        print("    train style   <= [%d] %s" % (len(temp_image_paths), train_style_image_pattern))
        train_style_image_paths.extend(temp_image_paths)
    train_dataset = AdaptiveInstanceNormDataset(
        content_image_paths=train_content_image_paths,
        style_image_paths=train_style_image_paths
    )
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print("    content:    %d" % len(train_content_image_paths))
    print("    style:      %d" % len(train_style_image_paths))
    print("    dataset:    %d" % len(train_dataset))
    print("    dataloader: %d" % len(train_dataloader))

    # Init Valid Data
    print("Init Valid Data")
    valid_content_image_paths = []
    for valid_content_image_pattern in sorted(args.valid_content_image_patterns):
        temp_image_paths = sorted(glob.glob(valid_content_image_pattern))
        print("    valid content <= [%d] %s" % (len(temp_image_paths), valid_content_image_pattern))
        valid_content_image_paths.extend(temp_image_paths)
    valid_style_image_paths = []
    for valid_style_image_pattern in sorted(args.valid_style_image_patterns):
        temp_image_paths = sorted(glob.glob(valid_style_image_pattern))
        print("    valid style   <= [%d] %s" % (len(temp_image_paths), valid_style_image_pattern))
        valid_style_image_paths.extend(temp_image_paths)
    num_valid_images = min(len(valid_content_image_paths), len(valid_style_image_paths))
    valid_content_image_paths = valid_content_image_paths[:num_valid_images]
    valid_style_image_paths = valid_style_image_paths[:num_valid_images]
    print("    content:    %d" % len(valid_content_image_paths))
    print("    style:      %d" % len(valid_style_image_paths))

    # Init Processor
    pre_processor = PreProcessor()
    post_processor = PostProcessor()

    # Init Optim
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr)

    # Init LRScheduler
    lr_scheduler = LRScheduler(optimizer, args.lr, args.lr_decay)

    # Init Criterion
    content_criterion = ContentLoss()
    style_criterion = StyleLoss()

    # Init Meter
    content_loss_meter = AverageMeter()
    style_loss_meter = AverageMeter()
    loss_meter = AverageMeter()

    # Loop
    iter_cnt = 0
    with tqdm(total=args.max_iters) as pbar:
        while True:
            # Train
            for content_inputs, style_inputs in train_dataloader:
                lr_scheduler(iter_cnt)

                content_inputs = content_inputs.cuda(non_blocking=True)
                style_inputs = style_inputs.cuda(non_blocking=True)

                style_features, adaptive_features, output_features = model(content_inputs, style_inputs)

                # print()
                # print("content_inputs                [%8.3f, %8.3f]" % (content_inputs.min().item(), content_inputs.max().item()))
                # print("style_inputs                  [%8.3f, %8.3f]" % (style_inputs.min().item(), style_inputs.max().item()))
                # print("style_features[\"relu4_1\"]     [%8.3f, %8.3f]" % (style_features["relu4_1"].min().item(), style_features["relu4_1"].max().item()))
                # print("style_features[\"relu3_1\"]     [%8.3f, %8.3f]" % (style_features["relu3_1"].min().item(), style_features["relu3_1"].max().item()))
                # print("adaptive_features             [%8.3f, %8.3f]" % (adaptive_features.min().item(), adaptive_features.max().item()))
                # print("output_features[\"relu4_1\"]    [%8.3f, %8.3f]" % (output_features["relu4_1"].min().item(), output_features["relu4_1"].max().item()))
                # print("output_features[\"relu3_1\"]    [%8.3f, %8.3f]" % (output_features["relu3_1"].min().item(), output_features["relu3_1"].max().item()))

                content_loss = content_criterion(output_features["relu4_1"], adaptive_features)
                # print("content_loss                  [%8.3f]" % content_loss.item())
                content_loss *= args.content_weight
                # print("content_loss                  [%8.3f]" % content_loss.item())

                style_loss = 0
                for key in output_features.keys():
                    temp_style_loss = style_criterion(output_features[key], style_features[key])
                    # print("style_loss[\"%s\"]         [%8.3f]" % (key, temp_style_loss.item()))
                    style_loss += temp_style_loss
                style_loss *= args.style_weight
                # print("style_loss                    [%8.3f]" % style_loss.item())

                # import pdb
                # pdb.set_trace()

                # content_inputs                [  -2.118,    2.640]
                # style_inputs                  [  -2.118,    2.640]
                # style_features["relu4_1"]     [   0.000,  117.341]
                # style_features["relu3_1"]     [   0.000,   65.178]
                # adaptive_features             [  -4.662,  143.250]
                # output_features["relu4_1"]    [   0.000, 4810.681]
                # output_features["relu3_1"]    [   0.000, 6744.249]
                # content_loss                  [5049.859]
                # content_loss                  [5049.859]
                # style_loss["relu1_1"]         [1049.200]
                # style_loss["relu2_1"]         [4035.273]
                # style_loss["relu3_1"]         [10710.754]
                # style_loss["relu4_1"]         [4859.628]
                # style_loss                    [206548.562]

                loss = content_loss + style_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(getattr(model.encoder.net, "0").weight.data[0])
                # print(getattr(model.decoder.net, "0").weight.data[0])

                content_loss_meter.update(content_loss)
                style_loss_meter.update(style_loss)
                loss_meter.update(loss)

                writer.add_scalar("loss_content", content_loss.item(), iter_cnt)
                writer.add_scalar("loss_style", style_loss.item(), iter_cnt)
                writer.add_scalar("lr", lr_scheduler.lr, iter_cnt)

                if (iter_cnt+1) % args.save_iters == 0 or (iter_cnt+1) == args.max_iters:

                    # Valid
                    model.eval()
                    for image_idx, (content_image_path, style_image_path) in enumerate(zip(valid_content_image_paths, valid_style_image_paths)):
                        content_input, style_input = pre_processor(content_image_path, style_image_path)
                        output = model(
                            content_input.unsqueeze(0).cuda(non_blocking=True),
                            style_input.unsqueeze(0).cuda(non_blocking=True)
                        ).detach().cpu().squeeze()
                        output_image = post_processor(output)
                        output_image.save("%s/image_%02d_iter_%06d.jpg" % (log_image_dir, image_idx+1, iter_cnt+1))
                    model.train()

                    # Save
                    torch.save(model.decoder.state_dict(), "%s/decoder_iter_%06d.pth" % (log_model_dir, iter_cnt+1))

                pbar.set_postfix(
                    loss_content="%.2e (avg: %.2e)" % (content_loss_meter.val, content_loss_meter.avg),
                    loss_style="%.2e (avg: %.2e)" % (style_loss_meter.val, style_loss_meter.avg),
                    lr="%.2e" % lr_scheduler.lr
                )
                pbar.update(1)

                iter_cnt += 1
                if iter_cnt == args.max_iters:
                    break
            if iter_cnt == args.max_iters:
                break
    writer.close()
    return


if __name__ == "__main__":

    """

python -m train

tensorboard --logdir logs

    """

    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument("--train_content_image_patterns", type=str, nargs="+", default=[
        "D:/Downloads/coco/train2017/*"
        # "D:/Downloads/coco/temp_style_transfer/*"
    ])
    parser.add_argument("--train_style_image_patterns", type=str, nargs="+", default=[
        "D:/Downloads/wikiart/train/*"
        # "D:/Downloads/wikiart/temp_style_transfer/*"
    ])
    parser.add_argument("--valid_content_image_patterns", type=str, nargs="+", default=[
        "./src/images/content/*"
    ])
    parser.add_argument("--valid_style_image_patterns", type=str, nargs="+", default=[
        "./src/images/style/*"
    ])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--max_iters", type=int, default=40000)
    parser.add_argument("--save_iters", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)

    parser.add_argument("--content_weight", type=float, default=5)
    parser.add_argument("--style_weight", type=float, default=25)

    parser.add_argument("--log_dir", default="./logs/train_20201117_3")
    args = parser.parse_args()

    main(args)
