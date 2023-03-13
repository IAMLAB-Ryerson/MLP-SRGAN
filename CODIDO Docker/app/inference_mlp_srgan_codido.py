import argparse
import zipfile
import cv2
import glob
import os
import scipy.io as sio
import nibabel as nib
import numpy as np
from mlpsrgan.models.rmrdb_net import GeneratorMixer

from mlpsrgan import MLPSRGANer, IAMLAB_mat_loader, IAMLAB_mat_writer, IAMLAB_nii_loader, IAMLAB_nii_writer


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input image | for 3D medical images use axial plane')
parser.add_argument(
    '-n',
    '--model_name',
    type=str,
    default='mlp-srgan-d-1',
    help=('Model names: mlp-srgan-d-1 | mlp-srgan-d-3 | mlp-srgan-d-5'))
parser.add_argument('--output', default='outputs', help='Output folder')
parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image (only 4 is available at the moment)')
parser.add_argument(
    '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
parser.add_argument('--codido', help='running on codido')


args = parser.parse_args()

input_folder_path = os.path.join(os.sep, 'app', 'inputs')
output_folder_path = os.path.join(os.sep, 'app', 'outputs')
os.makedirs(input_folder_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)

if args.codido == 'True':
    import boto3
    s3 = boto3.client('s3')

    # downloads codido input file into the folder specified by input_folder_path
    input_file_path = os.path.join(input_folder_path, args.input.split('_SPLIT_')[-1])
    s3.download_file(os.environ['S3_BUCKET'], args.input, input_file_path)
else:
    input_file_path = glob.glob(os.path.join(input_folder_path, '*'))[0]

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

args.ext = 'auto'
args.suffix = ''

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
    tile=0,
    tile_pad=0,
    pre_pad=0,
    half=False,
    gpu_id=0)

if os.path.isfile(input_folder_path):
    paths = [input_folder_path]
else:
    paths = sorted(glob.glob(os.path.join(input_folder_path, '*')))

for idx, path in enumerate(paths):
    imgname, extension = os.path.splitext(os.path.basename(path))
    print('Testing', idx, imgname)
    if ((extension == ".mat") or (extension == ".nii") or (extension == ".gz")):
        if (extension == ".mat"):
            img, m = IAMLAB_mat_loader(path)
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
                    save_path = os.path.join(output_folder_path, f'{imgname}.{extension}')
                else:
                    save_path = os.path.join(output_folder_path, f'{imgname}_{args.suffix}.{extension}')
                vol = IAMLAB_mat_writer(output_container, m)
                sio.savemat(save_path, {'SRvol': vol}, do_compression = True)
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
                    save_path = os.path.join(output_folder_path, f'{imgname}.{extension}')
                else:
                    if (extension == ".nii"):
                        save_path = os.path.join(output_folder_path, f'{imgname}_{args.suffix}.{extension}')
                    else:
                        save_path = os.path.join(output_folder_path, f'{imgname}.{extension}')
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
                save_path = os.path.join(output_folder_path, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(output_folder_path, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)

if args.codido == 'True':
    # create zip with all the saved outputs
    zip_name = output_folder_path + '.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(output_folder_path):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, output_folder_path))

    # upload
    s3.upload_file(zip_name, os.environ['S3_BUCKET'], args.output)
