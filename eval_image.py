"""Module eval image runs inference on a 2D image."""
import time
import os

import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from torch.autograd import Variable

from model import GeneratorMixer
from datasets import denormalize


# Model Path
MODEL_PATH = 'proposed_method_depth_1/models/generator_199.pth'
IMAGE_PATH = 'MSSEG2/Slices/Sagittal/LR/'
OUT_PATH = 'MSSEG2/Holdout Test/proposed_method2_depth_1/'

HR_HEIGHT = 256

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
tform = Compose([Resize((HR_HEIGHT//4,HR_HEIGHT), Image.BICUBIC), ToTensor(), Normalize(mean,std)])

# Check for CUDA
cuda = torch.cuda.is_available()

# Load Model
model = GeneratorMixer().eval()
if cuda:
    model = model.cuda()

model.load_state_dict(torch.load(MODEL_PATH))

# Load Images
images = os.listdir(IMAGE_PATH)

for image in images:
    print('Evaluating Image: ' + image)
    img = Image.open(IMAGE_PATH + image)
    img = Variable(tform(img)).unsqueeze(0)
    if cuda:
        img = img.cuda()

    start = time.time()
    out = denormalize(model(img))
    end = time.time() - start
    print(end)
    save_image(out, OUT_PATH + image)
