import argparse
import cv2
import glob
import os
import scipy.io as sio
import nibabel as nib
import numpy as np
from mlpsrgan.models.rmrdb_net import GeneratorMixer

from mlpsrgan import MLPSRGANer, IAMLAB_mat_loader, IAMLAB_mat_writer, IAMLAB_nii_loader, IAMLAB_nii_writer


def main():
    """Inference demo for MLP-SRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder | for 3D medical images use axial plane')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='mlp-srgan-d-1',
        help=('Model names: mlp-srgan-d-1 | mlp-srgan-d-3 | mlp-srgan-d-5'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image (only 4 is available at the moment)')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=0, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 precision during inference. Default: fp32 (single precision).')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'mlp-srgan-d-1':  # x4 RMRDBNet model with 1 block
        model = GeneratorMixer(n_residual_blocks=1)
        netscale = 4
    elif args.model_name == 'mlp-srgan-d-3':  # x4 RMRDBNet model with 3 blocks
        model = GeneratorMixer(n_residual_blocks=3)
        netscale = 4
    elif args.model_name == 'mlp-srgan-d-5':  # x4 RMRDBNet model with 5 blocks
        model = GeneratorMixer(n_residual_blocks=5)
        netscale = 4

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            print("[ERROR] Please download models from: https://drive.google.com/drive/folders/1q4f1Yzraqtgdplw9dAtbdtWLGSm7vzHx?usp=sharing and place them in the 'weights' folder")

    # restorer
    upsampler = MLPSRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.fp16,
        gpu_id=args.gpu_id)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)
        if ((extension == ".mat") or (extension == ".nii") or (extension == ".gz")):
            if (extension == ".mat"):
                img, m = IAMLAB_mat_loader(path)
                # try:
                #     output_container = np.zeros((4*img.shape[0], img.shape[1], img.shape[2]))
                #     slices = img.shape[2]
                #     for s in range(slices):
                #         print('Testing Slice: ' + str(s))
                #         sag_img = img[:, :, s]
                #         output, _ = upsampler.enhance(sag_img, outscale=args.outscale)
                #         output_container[:, :, s] = output
                # except RuntimeError as error:
                #     print('Error', error)
                #     print('[ERROR] If you encounter CUDA out of memory try to decrease the batch size.')
                # else:
                #     if args.ext == 'auto':
                #         extension = extension[1:]
                #     else:
                #         extension = args.ext
                #     if args.suffix == '':
                #         save_path = os.path.join(args.output, f'{imgname}.{extension}')
                #     else:
                #         save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
                #     vol = IAMLAB_mat_writer(output_container, m)
                #     sio.savemat(save_path, {'SRvol': vol}, do_compression = True)
            if ((extension == ".nii") or (extension == ".gz")):
                img, m = IAMLAB_nii_loader(path)
                try:
                    output_container = np.zeros((4*img.shape[0], img.shape[1], img.shape[2]))
                    slices = img.shape[2]
                    for s in range(slices):
                        print('Testing Slice: ' + str(s))
                        sag_img = img[:, :, s]
                        output, _ = upsampler.enhance(sag_img, outscale=args.outscale)
                        output_container[:, :, s] = output
                except RuntimeError as error:
                    print('Error', error)
                    print('[ERROR] If you encounter CUDA out of memory try to decrease the batch size.')
                else:
                    if args.ext == 'auto':
                        extension = extension[1:]
                    else:
                        extension = args.ext
                    if args.suffix == '':
                        save_path = os.path.join(args.output, f'{imgname}.{extension}')
                    else:
                        if (extension == ".nii"):
                            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
                        else:
                            save_path = os.path.join(args.output, f'{imgname}.{extension}')
                    vol = IAMLAB_nii_writer(output_container, m)
                    img = nib.Nifti1Image(vol, affine=np.eye(4))
                    nib.save(img, save_path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            else:
                img_mode = None
            try:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('[ERROR] If you encounter CUDA out of memory try to decrease the batch size.')
            else:
                if args.ext == 'auto':
                    extension = extension[1:]
                else:
                    extension = args.ext
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                if args.suffix == '':
                    save_path = os.path.join(args.output, f'{imgname}.{extension}')
                else:
                    save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
                cv2.imwrite(save_path, output)


if __name__ == '__main__':
    main()