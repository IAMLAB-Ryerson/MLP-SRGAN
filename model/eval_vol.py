import argparse
import scipy.io as sio
import skimage.transform as st
import numpy as np
import time
import os

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageOps
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import *
from datasets import *


# Model Path
model_path = '/media/samir/Primary/Deep Learning Models/Super Resolution/proposed_method_depth_1/models/generator_199.pth'
mat_path = '/media/samir/Primary/Data/Super Resolution/CAIN2/Holdout Set/Test/'
out_path = '/media/samir/Primary/Data/Super Resolution/CAIN2/Holdout Test/proposed_method2_depth_1_bicubic/'

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
images = os.listdir(mat_path)

for image in images:
	print('Evaluating Image: ' + image)
	moving = sio.loadmat(mat_path + image)
	moving = moving.get('im')
	moving = moving[0, 0]
	moving = moving['final'].astype('float32')

	# Convert to Saggital View
	sag = np.rot90(moving, k=1, axes=(0, 2))
	sag = np.rot90(sag, k=1, axes=(2, 1))

	# Convert to Compatible Intensity Range
	m = np.amax(moving)
	sag = ((sag/m)*255).astype('uint8')

	# Create Upscaled Array
	moving_SR = np.zeros((4*sag.shape[0], sag.shape[1], sag.shape[2]))

	# Evalulate each slice of the volume
	for j in range(sag.shape[2]):
		print('Evaluating Slice: ' + str(j))
		img = sag[:, :, j]
		img = np.stack((img, img, img), axis=2)
		sag_img = Image.fromarray(img)
		sag_img = Variable(transform(sag_img), volatile=True).unsqueeze(0)
		image1 = sag_img
		if cuda:
			image1 = image1.cuda()
		out = denormalize(model(image1))
		out_img = out.detach().cpu().numpy().squeeze()
		out_img = out_img[0, :, :]
		out_img = st.resize(out_img, (4*sag.shape[0], sag.shape[1]))
	moving_SR[:, :, j] = out_img

	moving_SR = np.rot90(moving_SR, k=-1, axes=(2, 1))
	moving_SR = np.rot90(moving_SR, k=-1, axes=(0, 2))
	moving_SR = ((moving_SR/255)*m).astype('float32')

	sio.savemat(out_path + image, {'SRvol': moving_SR}, do_compression = True)