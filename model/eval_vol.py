"""Module eval vol runs inference on a 3D image."""
import os

import scipy.io as sio
import skimage.transform as st
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from model import GeneratorMixer
from datasets import denormalize


# Model Path
MODEL_PATH = 'proposed_method_depth_1/models/generator_199.pth'
MAT_PATH = 'CAIN2/Holdout Set/Test/'
OUT_PATH = 'CAIN2/Holdout Test/proposed_method2_depth_1_bicubic/'

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
images = os.listdir(MAT_PATH)

for image in images:
    print('Evaluating Image: ' + image)
    moving = sio.loadmat(MAT_PATH + image)
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
        sag_img = Variable(tform(sag_img), volatile=True).unsqueeze(0)
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

    sio.savemat(OUT_PATH + image, {'SRvol': moving_SR}, do_compression = True)
