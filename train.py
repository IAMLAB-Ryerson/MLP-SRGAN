import os
import numpy as np
import math
import itertools
import sys
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import pandas as pd

import matplotlib.pyplot as plt

# Defaults
n_channels = 3
lr_size = 64
hr_size = 256
batch_size = 8
epochs = 200
upscale_factor = 4
learning_rate = 2e-4
b1 = 0.5
b2 = 0.999
decay_epoch = 100
model_loss = 'perceptual_loss'
train_split = 0.8
validation_split = 0.0
test_split = 0.1
sample_interval = 100
checkpoint_interval = 10
warmup_batches=500

# Directories
hr_train_path = '/media/samir/Primary/Data/Super Resolution/MSSEG2/Slices/Sagittal/HR/Train/'
model_path = '/media/samir/Primary/Deep Learning Models/Super Resolution/proposed_method_depth_1'
model_name = 'proposed_method2_depth_1_MSSEG2'

# Check for CUDA
cuda = torch.cuda.is_available()

# set HR size
hr_shape = (hr_size, hr_size)

# Create output directories
if(not os.path.isdir(model_path) or not os.listdir(model_path)):
	os.makedirs(model_path + '/models')
	os.makedirs(model_path + '/history')
	os.makedirs(model_path + '/figures')
	os.makedirs(model_path + '/params')
	os.makedirs(model_path + '/samples')

# Model Parameters
params = dict()
params['Number of channels'] = n_channels
params['Low Resolution Dim'] = lr_size
params['High Resolution Dim'] = hr_size
params['Batch Size'] = batch_size
params['Epochs'] = epochs
params['Upscale Factor'] = upscale_factor
params['Learning rate'] = learning_rate
params['Training split'] = train_split
params['Validation split'] = validation_split
params['Testing split'] = test_split

print(['Model Parameters'])
print('------------')
for key in params.keys():
	print(key + ':', params[key])

# Initialize Networks
generator = GeneratorMixer()
discriminator = Discriminator(input_shape=(n_channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set Feature Extractor to Inference Mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

if cuda:
	generator = generator.cuda()
	discriminator = discriminator.cuda()
	feature_extractor = feature_extractor.cuda()
	criterion_GAN = criterion_GAN.cuda()
	criterion_content = criterion_content.cuda()
	criterion_pixel = criterion_pixel.cuda()

# Set Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
	ImageDataset(hr_train_path, hr_shape=hr_shape),
	batch_size=batch_size,
	shuffle=True,
	num_workers=8
)

# Pandas Variables
epoch_array = []
discriminator_loss_array = []
generator_loss_array = []

# Start Time
start_time = time.time()

# Training Loop
for epoch in range(epochs):
	for i, imgs in enumerate(dataloader):

		batches_done = epoch * len(dataloader) + i

		# Configure Model Input
		imgs_lr = Variable(imgs["lr"].type(Tensor))
		imgs_hr = Variable(imgs["hr"].type(Tensor))

		# Adversarial Ground Truths
		valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
		fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

		## Train Generator
		optimizer_G.zero_grad()

		# Generate SR Image
		gen_hr = generator(imgs_lr)

		# Pixel Loss
		loss_pixel = criterion_pixel(gen_hr, imgs_hr)

		# Content Loss v1
		if batches_done < warmup_batches:
			# Warm-up (pixel-wise loss only)
			loss_pixel.backward()
			optimizer_G.step()
			print(
				"[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
				% (epoch, epochs, i, len(dataloader), loss_pixel.item())
			)
			continue

		# Extract validity predictions from discriminator
		pred_real = discriminator(imgs_hr).detach()
		pred_fake = discriminator(gen_hr)

		# Adversarial Loss
		loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

		# Content Loss v2
		gen_features = feature_extractor(gen_hr)
		real_features = feature_extractor(imgs_hr).detach()
		loss_content = criterion_content(gen_features, real_features)

		# Total Loss
		loss_G = loss_content + 5e-3 * loss_GAN + 1e-2 * loss_pixel

		loss_G.backward()
		optimizer_G.step()

		## Train Discriminator
		optimizer_D.zero_grad()

		pred_real = discriminator(imgs_hr)
		pred_fake = discriminator(gen_hr.detach())

		# Loss of Real and Fake Images
		loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
		loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

		# Total loss
		loss_D = (loss_real + loss_fake) / 2

		loss_D.backward()
		optimizer_D.step()

		## Log Progress
		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
			% (
				epoch,
				epochs,
				i,
				len(dataloader),
				loss_D.item(),
				loss_G.item(),
				loss_content.item(),
				loss_GAN.item(),
				loss_pixel.item(),
			)
		)

		if batches_done % sample_interval == 0:
			# Save image grid with upsampled inputs
			imgs_lr = nn.functional.interpolate(imgs_lr, size=(256, 256), mode='bicubic')
			img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
			save_image(img_grid, model_path + '/samples/' + "%d.png" % batches_done, nrow=1, normalize=False)

	# Append to Pandas Array
	epoch_array.append(epoch)
	discriminator_loss_array.append(loss_D.item())
	generator_loss_array.append(loss_G.item())

	if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
		# Save model checkpoints
		torch.save(generator.state_dict(), model_path + '/models/generator_%d.pth' % epoch)
		torch.save(discriminator.state_dict(), model_path + '/models/discriminator_%d.pth' % epoch)

# Save Final Models
torch.save(generator.state_dict(), model_path + '/models/generator_%d.pth' % epoch)
torch.save(discriminator.state_dict(), model_path + '/models/discriminator_%d.pth' % epoch)

# Finish Time
elapsed_time = time.time() - start_time

# Write to Spreadsheet
df = pd.DataFrame(
		data={'Epoch': epoch, 'D loss': discriminator_loss_array, 'G loss': generator_loss_array}
	)
df.to_csv(model_path + '/history/' + model_name + '.csv')

# Save parameters
params['Training Times'] = elapsed_time
f = open(model_path + '/params/' + model_name + '.txt', 'w')
f.write('[Model Parameters]' + '\n')
f.write('------------' + '\n')
for k, v in params.items():
	f.write(str(k) + ': '+ str(v) + '\n')
f.close()

# Display loss curves
fig, ax = plt.subplots(1, 1)
ax.plot(generator_loss_array, color='blue', label='Generator Loss')
ax.plot(discriminator_loss_array, color='orange', label='Discriminator Loss')
ax.set_title('Loss Curves')
ax.set_ylabel('Losses')
ax.set_xlabel('Epochs')
plt.legend()

# Plot Loss
plt.savefig(model_path + '/figures/' + model_name + '.png')
print('Loss figure saved successfully.')