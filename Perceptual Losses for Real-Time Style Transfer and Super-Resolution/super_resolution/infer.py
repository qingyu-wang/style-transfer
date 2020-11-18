import glob
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm

from criterion import ContentLoss
from dataset import SuperResolutionDataset, PostProcessor
from model import SuperResolutionX4, VGG16Features
from utils import AverageMeter


def main(model_path, log_dir):

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
        print("mkdir => %s" % log_dir)

    log_image_dir = "%s/images" % log_dir
    if not os.path.isdir(log_image_dir):
        os.mkdir(log_image_dir)
        print("mkdir => %s" % log_image_dir)

    torch.backends.cudnn.benchmark = True

    image_paths = sorted(glob.glob("./src/images/*"))

    dataset = SuperResolutionDataset(image_paths=image_paths, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], large_size=(288, 288), small_size=(72, 72))
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    num_data = len(dataset)
    num_iters = len(dataloader)
    print("Init Data (data: %d, iters: %d)" % (num_data, num_iters))

    post_processor = PostProcessor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = SuperResolutionX4(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.cuda()
    print("Init Model")

    model.eval()
    with tqdm(desc="infer", total=num_iters) as pbar:
        data_idx = 0
        for inputs, targets in dataloader:

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
                        reference=np.array(image_target),
                        multichannel=True
                    )
                )

                image_input.save("%s/image_%02d_input.jpg" % (log_image_dir, data_idx))
                image_target.save("%s/image_%02d_target.jpg" % (log_image_dir, data_idx))
                image_output.save("%s/image_%02d_output.jpg" % (log_image_dir, data_idx))
                image_output_final.save("%s/image_%02d_output_final.jpg" % (log_image_dir, data_idx))

                data_idx += 1
            # Info
            pbar.update(1)
    return


if __name__ == "__main__":

    """

python -m infer

    """

    main(
        model_path="./log/train/models/last.pth",
        log_dir="./log/infer"
    )

    # main(
    #     model_path="./log/train_beta_method_3/models/last.pth",
    #     log_dir="./log/infer_beta_method_3"
    # )
