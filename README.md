# MLP-SRGAN

![issues](https://img.shields.io/github/issues-raw/IAMLAB-Ryerson/MLP-SRGAN)

## Contents
This repository contains:
* MLP-SRGAN PyTorch model
* MLP-SRGAN PyTorch training script
* MLP-SRGAN PyTorch data generator
* MLP-SRGAN PyTorch inference script
* No-reference image metrics

The MLP-SRGAN scripts are preconfigured to run MLP-SRGAN (D-1). To reconfigure the scripts to use MLP-SRGAN (D-3) or MLP-SRGAN (D-5) set n_residual_blocks in GeneratorMixer in model.py to either 3 or 5 respectively.

## Pretrained Models
Pretrained models are available on Google Drive at the following link:
https://drive.google.com/drive/folders/1q4f1Yzraqtgdplw9dAtbdtWLGSm7vzHx?usp=sharing

## Training System Configuration
|  CPU | GPU | RAM |
| :---: | :---: | :---: |
|  AMD Threadripper 3990x | Nvidia RTX 3090 24 GB | 256 GB |

## Model Diagrams
![Generator](images/generator.png)
![Discriminator](images/discriminator.png)
