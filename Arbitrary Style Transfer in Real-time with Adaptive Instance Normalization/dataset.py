import random

import torch
import torch.utils.data as data

from PIL import Image
from torchvision import transforms


class AdaptiveInstanceNormDataset(data.Dataset):

    def __init__(self, content_image_paths, style_image_paths, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.content_image_paths = content_image_paths
        self.style_image_paths = style_image_paths
        self.image_transformer = transforms.Compose([
            transforms.Resize(512, Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.content_image_paths)

    def __getitem__(self, idx):
        # Content Input
        content_idx = idx
        content_image = Image.open(self.content_image_paths[content_idx]).convert("RGB")
        content_input = self.image_transformer(content_image)
        # Style Input
        style_idx = random.randint(0, len(self.style_image_paths)-1)
        style_image = Image.open(self.style_image_paths[style_idx]).convert("RGB")
        style_input = self.image_transformer(style_image)
        return content_input, style_input


class PreProcessor(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, content_image_path, style_image_path):
        # Content Input
        content_image = Image.open(content_image_path).convert("RGB")
        content_input = self.transformer(content_image)
        # Style Input
        style_image = Image.open(style_image_path).convert("RGB")
        style_image = style_image.resize(content_image.size, Image.BICUBIC)
        style_input = self.transformer(style_image)
        return content_input, style_input


class PostProcessor(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.transformer = transforms.Compose([
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage()
        ])

    def __call__(self, input_):
        output = input_ * self.std + self.mean
        output = self.transformer(output)
        return output


if __name__ == "__main__":

    """

python -m dataset

    """

    import glob
    import random

    content_image_paths = glob.glob("./src/images/content/*")
    print("content images: %s" % len(content_image_paths))

    style_image_paths = glob.glob("./src/images/style/*")
    print("style   images: %s" % len(style_image_paths))

    dataset = AdaptiveInstanceNormDataset(content_image_paths, style_image_paths)
    print("image_transformer")
    print(dataset.image_transformer)

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=2
    )

    post_process = PostProcessor()

    for content_inputs, style_inputs in dataloader:
        for idx, (content_input, style_input) in enumerate(zip(content_inputs, style_inputs)):
            post_process(content_input).save("./tmp/dataset/%d_content_input.jpg" % idx)
            post_process(style_input).save("./tmp/dataset/%d_style_input.jpg" % idx)
