"""Module train is a script to train the MLP-SRGAN."""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image

from model import GeneratorMixer, Discriminator, FeatureExtractor
from datasets import ImageDataset, denormalize


# Defaults
N_CHANNELS = 3
LR_SIZE = 64
HR_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
UPSCALE_FACTOR = 4
LEARNING_RATE = 2e-4
B1 = 0.5
B2 = 0.999
DECAY_EPOCH = 100
MODEL_LOSS = 'perceptual_loss'
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.0
TEST_SPLIT = 0.1
SAMPLE_INTERVAL = 100
CHECKPOINT_INTERVAL = 10
WARMUP_BATCHES = 500

# Directories
HR_TRAIN_PATH = 'MSSEG2/Slices/Sagittal/HR/Train/'
MODEL_PATH = 'proposed_method_depth_1'
MODEL_NAME = 'proposed_method2_depth_1_MSSEG2'

# Check for CUDA
cuda = torch.cuda.is_available()

# set HR size
hr_shape = (HR_SIZE, HR_SIZE)

# Create output directories
if(not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH)):
    os.makedirs(MODEL_PATH + '/models')
    os.makedirs(MODEL_PATH + '/history')
    os.makedirs(MODEL_PATH + '/figures')
    os.makedirs(MODEL_PATH + '/params')
    os.makedirs(MODEL_PATH + '/samples')

# Model Parameters
params = {}
params['Number of channels'] = N_CHANNELS
params['Low Resolution Dim'] = LR_SIZE
params['High Resolution Dim'] = HR_SIZE
params['Batch Size'] = BATCH_SIZE
params['Epochs'] = EPOCHS
params['Upscale Factor'] = UPSCALE_FACTOR
params['Learning rate'] = LEARNING_RATE
params['Training split'] = TRAIN_SPLIT
params['Validation split'] = VALIDATION_SPLIT
params['Testing split'] = TEST_SPLIT

print(['Model Parameters'])
print('------------')
for key in params.items():
    print(key + ':', params[key])

# Initialize Networks
generator = GeneratorMixer()
discriminator = Discriminator(input_shape=(N_CHANNELS, *hr_shape))
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
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset(HR_TRAIN_PATH, hr_shape=hr_shape),
    batch_size=BATCH_SIZE,
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
for epoch in range(EPOCHS):
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
        if batches_done < WARMUP_BATCHES:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            warmup_info = (epoch, EPOCHS, i, len(dataloader), loss_pixel.item())
            print(
                f"[Epoch {warmup_info[0]}/{warmup_info[1]}] [Batch {warmup_info[2]}/{warmup_info[3]}] [G pixel: {warmup_info[4]}]"
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
        train_info = (epoch, EPOCHS, i, len(dataloader), loss_D.item(), loss_G.item(), loss_content.item(), loss_GAN.item(), loss_pixel.item())
        print(
            f"[Epoch {train_info[0]}/{train_info[1]}] [Batch {train_info[2]}/{train_info[3]}] [D loss: {train_info[4]}] [G loss: {train_info[5]}, content: {train_info[6]}, adv: {train_info[7]}, pixel: {train_info[8]}]"
        )

        if batches_done % SAMPLE_INTERVAL == 0:
            # Save image grid with upsampled inputs
            imgs_lr = nn.functional.interpolate(imgs_lr, size=(256, 256), mode='bicubic')
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            save_image(img_grid, MODEL_PATH + '/samples/' + f"{batches_done}.png", nrow=1, normalize=False)

    # Append to Pandas Array
    epoch_array.append(epoch)
    discriminator_loss_array.append(loss_D.item())
    generator_loss_array.append(loss_G.item())

    if CHECKPOINT_INTERVAL != -1 and epoch % CHECKPOINT_INTERVAL == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), MODEL_PATH + f'/models/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), MODEL_PATH + f'/models/discriminator_{epoch}.pth')

# Save Final Models
torch.save(generator.state_dict(), MODEL_PATH + f'/models/generator_{epoch}.pth')
torch.save(discriminator.state_dict(), MODEL_PATH + f'/models/discriminator_{epoch}.pth')

# Finish Time
elapsed_time = time.time() - start_time

# Write to Spreadsheet
df = pd.DataFrame(
        data={'Epoch': epoch, 'D loss': discriminator_loss_array, 'G loss': generator_loss_array}
    )
df.to_csv(MODEL_PATH + '/history/' + MODEL_NAME + '.csv')

# Save parameters
params['Training Times'] = elapsed_time
with open(MODEL_PATH + '/params/' + MODEL_NAME + '.txt', encoding="utf-8") as f:
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
plt.savefig(MODEL_PATH + '/figures/' + MODEL_NAME + '.png')
print('Loss figure saved successfully.')
