<p align="center">
  <img src="assets/IAMLab-Logo.jpg" height=120>
</p>

# MLP-SRGAN: A Single-Dimension Super Resolution GAN using MLP-Mixer

[![issues](https://img.shields.io/github/issues-raw/IAMLAB-Ryerson/MLP-SRGAN)](https://github.com/IAMLAB-Ryerson/MLP-SRGAN/issues)

## Dependencies
* Python 3
* Python packages: ```pip install numpy scikit-image scipy PyWavelets pandas torch torchvision einops matplotlib nibabel basicsr```

## Command Line Usage
Ideal input image size is 256 x 64, tiling will be used if images exceed these dimensions.

```console
Usage: python inference_mlpsrgan.py -n mlp-srgan-d-1 -i infile -o outfile [options]...

  -h                   show this help
  -i --input           Input image or folder | for 3D medical images use axial plane. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image (only 4 is available at the moment). Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --fp16               Use fp16 precision during inference. Default: fp32 (single precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

## Pretrained Models
Pretrained models are available on Google Drive at the following link:
https://drive.google.com/drive/folders/1q4f1Yzraqtgdplw9dAtbdtWLGSm7vzHx?usp=sharing

## Model Diagrams
![Generator](assets/generator.png)
![Discriminator](assets/discriminator.png)

## Image Samples
![MSSEG2](assets/msseg2_superres.png)

## Contact
If you have any questions please email `samir.mitha@torontometu.ca`.

## License
[GPL v3.0](https://github.com/IAMLAB-Ryerson/MLP-SRGAN/blob/main/LICENSE)

## arXiv Paper

## Citations

## See Also
This repository uses the [PyTorch MLP-Mixer](https://github.com/lucidrains/mlp-mixer-pytorch).

This repository uses the format provided by [BasicSR](https://github.com/XPixelGroup/BasicSR). Please check out the repository!

This work is inspired by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for natural images.
