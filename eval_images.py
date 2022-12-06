import argparse
import time
import os
import numpy as np

import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import *
from datasets import *


# Model Path
model_path = '/media/samir/Primary/Deep Learning Models/Super Resolution/proposed_method_depth_1/models/generator_199.pth'
image_path = '/media/samir/Primary/Data/Super Resolution/MSSEG2/Slices/Sagittal/LR/'
ref_path = '/media/samir/Primary/Data/Super Resolution/MSSEG2/Holdout Test/reference/'
out_path = '/media/samir/Primary/Data/Super Resolution/MSSEG2/Holdout Test/proposed_method2_depth_1/'

hr_height = 256

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize((hr_height // 4, hr_height), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean, std)])

# Check for CUDA
cuda = torch.cuda.is_available()

# Load Model
model = GeneratorMixer().eval()
if cuda:
	model = model.cuda()

model.load_state_dict(torch.load(model_path))

# Load Images
images = os.listdir(ref_path)

for image in images:
	print('Evaluating Image: ' + image)
	img = Image.open(ref_path + image)
	img = Variable(transform(img)).unsqueeze(0)
	if cuda:
		img = img.cuda()

	start = time.time()
	out = denormalize(model(img))
	end = time.time() - start
	print(end)
	save_image(out, out_path + image)