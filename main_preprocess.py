import argparse
import os
import glob
# import numpy as np

print(os.getcwd())
from utils import *
from scipy.ndimage import zoom
import nibabel as nib
# import scipy.ndimage as spimg

name = "unpaired_ct_abdomen"
# name = "unpaired_ct_lung"

parser = argparse.ArgumentParser()

#=======================================================================================================================
parser.add_argument(
        "--data_name",
        "-dn",
        help="data name for training."
        "-dn for ",
        type=str,
        default=name,
        required=False,
    )

parser.add_argument(
    "--proc_type",
    help="Training/testing type",
    default="train",
    type=str,
    required=False,
)

args = parser.parse_args()
#=======================================================================================================================

rescale_factor=0.5
process_suffix='_x2'
# from_suffix='_x4'
process_type=args.proc_type


data_name=args.data_name

crop_sz = np.array([0, 0, 0])

pair_type= "unpaired" if "unpaired" in data_name else "paired"


data_path=os.path.join('.','data',data_name,'dataset')

print(os.path.abspath(data_path))

to_process_paths = os.path.join(data_path,args.proc_type)
processed_paths = os.path.join(data_path,args.proc_type+'_proc')

if 'unpair' in name:
    # suffix = '*.nii'
    suffix = '*.nii.gz'
    image_pths = glob.glob(os.path.join(to_process_paths, 'images', suffix))
    label_pths = glob.glob(os.path.join(to_process_paths, 'labels', suffix))
else:
    # suffix = '*.nii'
    suffix = '*.nii.gz'
    image_pths = glob.glob(os.path.join(to_process_paths, '*_images', suffix))
    label_pths = glob.glob(os.path.join(to_process_paths, '*_labels', suffix))



ndims=3
if data_name in ["unpaired_ct_lung"]:
    select = [0]
else:
    select = [0, 1, 2, 4, 5, 7, 8, 9, 10]   # label for abdomen CT


def preprocess(pths, rescale_factor=rescale_factor,select=[0]):
    x=[]
    if len(select)<=1:
        for pth in pths:
            x.append(zoom(np.array(nib.load(pth).dataobj), rescale_factor, mode='nearest'))
    else:
        for pth in pths:
            x.append(np.stack([zoom(np.array(nib.load(pth).dataobj)[..., i], rescale_factor, mode='nearest').astype(float) for i in select], -1))
    return x

img_proc=preprocess(image_pths)
[write_image(img,file_path=os.path.join(processed_paths,os.path.basename(os.path.dirname(img_pth)))+str(process_suffix),file_prefix=os.path.basename(img_pth)) for img,img_pth in zip(img_proc,image_pths)]
if len(label_pths)>0:
    lab_proc=preprocess(label_pths,select=select)
    [write_image(img,file_path=os.path.join(processed_paths,os.path.basename(os.path.dirname(img_pth)))+str(process_suffix),file_prefix=os.path.basename(img_pth)) for img,img_pth in zip(lab_proc,label_pths)]

