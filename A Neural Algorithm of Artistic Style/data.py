import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms


class PreProcessor(object):

    def __init__(self, ctype="BGR", div=1, mean=[103.939, 116.779, 123.68], std=[1, 1, 1], size=512, init_mode="random"):
        assert ctype in ["BGR", "RGB"]
        assert init_mode in ["random", "content", "style"]
        if ctype == "BGR":
            self.transformer = transforms.Compose([
                transforms.Resize(size),
                transforms.Lambda(lambda x: torch.tensor(np.array(x)).mul(1/div).permute((2, 0, 1)).float()),
                transforms.Lambda(lambda x: x.flip(0)),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.Resize(size),
                transforms.Lambda(lambda x: torch.tensor(np.array(x)).mul(1/div).permute((2, 0, 1)).float()),
                transforms.Normalize(mean=mean, std=std)
            ])
        self.init_mode = init_mode

    def __call__(self, content_image_path, style_image_path):
        # Content
        content_image = Image.open(content_image_path).convert("RGB")
        content_input = self.transformer(content_image)
        # Style
        style_image = Image.open(style_image_path).convert("RGB")
        style_input = self.transformer(style_image)
        # Transfer Input
        if self.init_mode == "random":
            transfer_input = torch.randn(content_input.shape).mul(0.001)
        elif self.init_mode == "content":
            transfer_input = content_input.clone()
        elif self.init_mode == "style":
            transfer_image = style_image.copy()
            transfer_image = transfer_image.resize(content_image.size)
            transfer_input = self.transformer(transfer_image)
        return content_input, style_input, transfer_input


class PostProcessor(object):

    def __init__(self, ctype="BGR", div=1, mean=[103.939, 116.779, 123.68], std=[1, 1, 1]):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        if ctype == "BGR":
            self.transformer = transforms.Compose([
                transforms.Lambda(lambda x: x.mul(std).add(mean)),
                transforms.Lambda(lambda x: x.mul(div).permute(1, 2, 0).clamp(0, 255)),
                transforms.Lambda(lambda x: x.flip(2)),
                transforms.Lambda(lambda x: Image.fromarray(x.numpy().astype(np.uint8)))
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.Lambda(lambda x: x.mul(std).add(mean)),
                transforms.Lambda(lambda x: x.mul(div).permute(1, 2, 0).clamp(0, 255)),
                transforms.Lambda(lambda x: Image.fromarray(x.numpy().astype(np.uint8)))
            ])

    def __call__(self, input_):
        output = self.transformer(input_)
        return output


def transfer_color(src, dst):
    """
    transfer dst color style to src

    src control content
    dst control color style
    """
    src = np.array(src)
    dst = np.array(dst)

    _H, _W, _C = src.shape
    dst = cv2.resize(dst, dsize=(_W, _H), interpolation=cv2.INTER_CUBIC)

    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)            #1 Extract the Destination"s luminance
    dst_yiq = cv2.cvtColor(dst, cv2.COLOR_RGB2YCrCb)            #2 Convert the Source from RGB to YIQ/YCbCr
    dst_yiq[..., 0] = src_gray                                  #3 Combine Destination"s luminance and Source"s IQ/CbCr
    dst_transfer = cv2.cvtColor(dst_yiq, cv2.COLOR_YCrCb2RGB)   #4 Convert new image from YIQ back to RGB

    dst_transfer = Image.fromarray(dst_transfer)
    return dst_transfer


if __name__ == "__main__":

    """

python -m data

    """

    import glob

    content_image_paths = sorted(glob.glob("./src/images/content/*"))[:3]
    print("content images: %s" % len(content_image_paths))

    style_image_paths = sorted(glob.glob("./src/images/style/*"))[:3]
    print("style   images: %s" % len(style_image_paths))

    # pre_processor = PreProcessor(init_mode="style")
    # post_processor = PostProcessor()

    pre_processor = PreProcessor(ctype="BGR", div=1, mean=[103.939, 116.779, 123.68], std=[1, 1, 1], init_mode="random")
    post_processor = PostProcessor(ctype="BGR", div=1, mean=[103.939, 116.779, 123.68], std=[1, 1, 1])

    # pre_processor = PreProcessor(ctype="RGB", div=255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], init_mode="random")
    # post_processor = PostProcessor(ctype="RGB", div=255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for idx, (content_image_path, style_image_path) in enumerate(zip(content_image_paths, style_image_paths)):
        content_input, style_input, transfer_input = pre_processor(content_image_path, style_image_path)
        post_processor(content_input).save("./tmp/data/%d_content_input.jpg" % idx)
        post_processor(style_input).save("./tmp/data/%d_style_input.jpg" % idx)
        post_processor(transfer_input).save("./tmp/data/%d_transfer_input.jpg" % idx)
