import torch
import torch.utils.data as data

from PIL import Image, ImageFilter
from torchvision import transforms


class GaussianBlur(object):
    
    def __init__(self, radius=1):
        self.radius = radius
        self.filter = ImageFilter.GaussianBlur(radius=radius)

    def __call__(self, image):
        image = image.filter(self.filter)
        return image

    def __repr__(self):
        return self.__class__.__name__ + "(%s)" % self.radius


class SuperResolutionDataset(data.Dataset):

    def __init__(self, image_paths, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], large_size=(288, 288), small_size=(72, 72)):
        self.image_paths = image_paths
        self.large_size = large_size
        self.small_size = small_size
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.large_transformer = transforms.Compose([
            transforms.Resize(self.large_size, Image.BICUBIC)
        ])
        self.small_transformer = transforms.Compose([
            GaussianBlur(radius=1),
            transforms.Resize(self.small_size, Image.BICUBIC)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        target = Image.open(self.image_paths[idx]).convert("RGB")

        target = self.large_transformer(target)
        input_ = self.small_transformer(target)

        input_ = self.image_transformer(input_)
        target = self.image_transformer(target)

        return input_, target


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

    image_paths = glob.glob("./src/images/*")
    print("images: %s" % len(image_paths))

    dataset = SuperResolutionDataset(image_paths)
    print("image_transformer")
    print(dataset.image_transformer)
    print("large_transformer")
    print(dataset.large_transformer)
    print("small_transformer")
    print(dataset.small_transformer)

    post_process = PostProcessor()

    input_, target = dataset.__getitem__(idx=random.randint(0, len(dataset)-1))
    
    post_process(input_).save("./tmp/dataset/input_.jpg")
    post_process(target).save("./tmp/dataset/target.jpg")
