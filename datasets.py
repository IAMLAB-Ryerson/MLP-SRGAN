"""Module data loader and preprocessing for training and inference."""
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    '''
    Denormalizes already normalized tensor(s).

            Parameters:
                    tensors (int, float16, float32, float64): A PyTorch tensor

            Returns:
                   torch.clamp(tensors, 0, 255): A clamped denormalized tensor
    '''
    for item in range(3):
        tensors[:, item].mul_(std[item]).add_(mean[item])
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
    """
    A class to encapsulte the image dataset for trianing.
    """
    def __init__(self, root, hr_shape):
        hr_height, _ = hr_shape
        # Transforms for lr ans hr images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
