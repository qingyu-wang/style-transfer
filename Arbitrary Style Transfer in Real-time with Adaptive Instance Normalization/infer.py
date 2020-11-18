import argparse
import glob
import os

import torch

from PIL import Image, ImageFile
from tqdm import tqdm

from dataset import PreProcessor, PostProcessor
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

    log_image_dir = "%s/images" % args.log_dir
    if not os.path.isdir(log_image_dir):
        os.mkdir(log_image_dir)
        print("mkdir => %s" % log_image_dir)

    # Init Model
    print("Init Model")
    model = AdaptiveInstanceNormModel()
    model.decoder.load_state_dict(torch.load(args.decoder_state_dict_path, map_location="cpu"))
    model.eval()

    # Init Data
    print("Init Data")
    assert len(args.content_image_paths) == len(args.style_image_paths)
    print("    content:    %d" % len(args.content_image_paths))
    print("    style:      %d" % len(args.style_image_paths))

    # Init Processor
    pre_processor = PreProcessor()
    post_processor = PostProcessor()

    # Loop
    with tqdm(total=len(args.content_image_paths)) as pbar:
        # Infer
        for image_idx, (content_image_path, style_image_path) in enumerate(zip(args.content_image_paths, args.style_image_paths)):
            content_input, style_input = pre_processor(content_image_path, style_image_path)
            output = model(
                content_input.unsqueeze(0),
                style_input.unsqueeze(0)
            ).detach().cpu().squeeze()
            output_image = post_processor(output)
            output_image.save("%s/image_%02d.jpg" % (log_image_dir, image_idx+1))
            pbar.update(1)
    return


if __name__ == "__main__":

    """

python -m infer

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image_paths", type=str, nargs="+", default=[
        "./src/images/content/sailboat.jpg",
        "./src/images/content/cornell.jpg",
        "./src/images/content/lenna.jpg",
        "./src/images/content/brad_pitt.jpg",
        "./src/images/content/bridge.jpg"
    ])
    parser.add_argument("--style_image_paths", type=str, nargs="+", default=[
        "./src/images/style/sketch.png",
        "./src/images/style/woman_with_hat_matisse.jpg",
        "./src/images/style/picasso_seated_nude_hr.jpg",
        "./src/images/style/picasso_self_portrait.jpg",
        "./src/images/style/la_muse.jpg"
    ])

    parser.add_argument("--decoder_state_dict_path", default="./logs/train_20201117_3/models/decoder_iter_021000.pth")
    parser.add_argument("--log_dir", default="./logs/infer_20201117_3")
    args = parser.parse_args()

    main(args)
