"""
Preprocesses gives images using Gaussian Blur and Gray Crop.
"""

import argparse
import os
import sys
from tqdm import tqdm

import cv2
from scipy import ndimage
import numpy as np
import pandas as pd

from utils import remove_ext


def crop_image_from_gray(img, tol=10):
    """
    Crops retina area from fundus image using grayscale thresholding.

    tol controls the threshold for grayscale cropping.
    """

    dims = img.ndim

    if dims == 3:
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif dims == 2:
        gray_image = img

    mask = gray_image > tol

    # find start crop point
    objs = ndimage.find_objects(mask)
    height_offset = objs[0][0].start
    width_offset = objs[0][1].start

    ix = np.ix_(mask.any(1),mask.any(0))

    if dims == 2:
        return img[ix], height_offset, width_offset
    
    check_shape = img[:,:,0][ix].shape[0]
    if (check_shape == 0): # image is too dark so that we crop out everything,
        return img, 0, 0 # return original image

    img1 = img[:,:,0][ix]
    img2 = img[:,:,1][ix]
    img3 = img[:,:,2][ix]
    img = np.stack([img1, img2, img3],axis=-1)
    
    return img, height_offset, width_offset


def cropnblur(img, sigmaX=30):   
    """
    Highlights retina image using gaussian blur.

    sigmaX controls the color and contrast of the output image
    """

    img, height_offset, width_offset = crop_image_from_gray(img)
    img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)

    return img, height_offset, width_offset


def crop(img, filename, offsets):
    """
    Crops image using offsets
    """

    coordinates = offsets.loc[remove_ext(filename)]
    img_t = img[coordinates['height_offset']: coordinates['height_offset']+coordinates['new_height'], 
                coordinates['width_offset']: coordinates['width_offset']+coordinates['new_width']]
                
    return img_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesses gives images using Gaussian Blur and Gray Crop.",
                                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--offsets', '-o', type=str, required=True,
                            help='directory OR full CSV path to load existing offsets or to save generated offsets.')
    parser.add_argument('--src', '-s', type=str,
                            help='source directory containing fundus images.\nOptional when --offsets path contains valid offsets.')
    parser.add_argument('--img_dst', type=str,
                            help='directory to save cropped images.\nRequired when --src is set.')
    parser.add_argument('--masks', '-m', type=str,
                            help='source directory containing masks.\nMasks should have same name as original images.')
    parser.add_argument('--mask_dst', type=str,
                            help='directory to save cropped images.\nRequired when --masks is set.')
    parser.add_argument('--mode', type=str, choices=["train", "test"],
                            help="to append to --offsets / -o, --src / -s, --img_dst, --masks, --mask_dst.\nUseful when you have `train` and `test` splits in separate directories.")
    parser.add_argument('--force', action='store_true',
                            help='force preprocessing when offsets are already present.\nIgnored when --src is not set.')
    parser.add_argument('--skip', type=str,
                            help="comma separated list of directory names (not path) to skip\ne.g., --skip class_1,class_2.\nUsed only when --mask / -m is given.")
    
    args = parser.parse_args()


    """
    =====================================
    Checking conditions and setting paths
    =====================================
    """
    offsets_path = args.offsets
    src = os.path.join(args.src, args.mode) if args.mode and args.src else args.src
    img_dst = os.path.join(args.img_dst, args.mode) if args.mode and args.img_dst else args.img_dst
    masks_src = os.path.join(args.masks, args.mode) if args.mode and args.masks else args.masks
    mask_dst = os.path.join(args.mask_dst, args.mode) if args.mode and args.mask_dst else args.mask_dst
    if args.skip:
        to_ignore = [name.strip() for name in args.skip.split(',')]
    else:
        to_ignore = []

    if src and not img_dst:
        parser.print_help()
        print("\n'--img_dst' must be given when setting '--src / -s' path.\n")
        sys.exit(-1)

    if masks_src and not mask_dst:
        parser.print_help()
        print("\n'--mask_dst' must be given when setting '--mask / -m' path.\n")
        sys.exit(-1)

    # check for existing offsets
    skip_preprocessing = offsets_path.endswith('csv') and os.path.isfile(offsets_path)

    if not src and not skip_preprocessing:
        parser.print_help()
        print("\nValid '--offsets' path must be given omitting '--src / -s' path.")
        print("--offsets:", offsets_path)
        sys.exit(-1)
    
    if src:    # force preprocessing
        skip_preprocessing = not args.force and skip_preprocessing

    # create directories
    if img_dst and src:
        os.makedirs(img_dst, exist_ok=True)

    if mask_dst:
        os.makedirs(mask_dst, exist_ok=True)

    # load data and sets variables
    if skip_preprocessing:
        print(f'Found offsets in {offsets_path}, skipping preprocessing.')
        offsets = pd.read_csv(offsets_path, index_col='filename')
        offsets = offsets.astype('int32')

    elif not offsets_path.endswith('csv'):
        if args.mode:
            offsets_path = os.path.join(args.offsets, args.mode)

        base_path = os.path.dirname(offsets_path)
        os.makedirs(base_path, exist_ok=True)

        name, ext = os.path.splitext(os.path.basename(offsets_path))
        if not ext:
            name = os.path.basename(src)+'_offsets.csv'
        else:
            name = name + '.csv'
        offsets_out = os.path.join(base_path, name)

        print("Offsets path is not a CSV file. Offsets will be saved in", offsets_out)
    else:
        offsets_out = offsets_path

    if not skip_preprocessing:
        offsets = pd.DataFrame()


    """
    =========================
    Processsing Fundus Images
    =========================
    """
    if src and not skip_preprocessing:
        print("\nProcessing images...")

        for filename in tqdm(os.listdir(src)):
            img = cv2.imread(os.path.join(src, filename))
            height, width, _ = img.shape
            img_t, height_offset, width_offset = cropnblur(img, sigmaX=35)
            new_height, new_width, _ = img_t.shape

            offsets = offsets.append({'filename': remove_ext(filename), 'original_height': height, 'original_width': width,
                                        'height_offset': height_offset, 'width_offset': width_offset,
                                        'new_height': new_height, 'new_width': new_width}, ignore_index=True)

            cv2.imwrite(os.path.join(img_dst, filename), img_t)
        
        # save offsets to disk
        offsets = offsets.set_index('filename')
        offsets = offsets.astype('int32')
        offsets.to_csv(offsets_out)
    

    """
    =========================
    Processsing Anomaly Masks
    =========================
    """
    if masks_src:
        print("\nProcessing masks...")

        for _type in os.listdir(masks_src):
            path = os.path.join(masks_src, _type)

            if os.path.isfile(path):
                print(_type)
                img = cv2.imread(path)
                img_t = crop(img, _type, offsets)
                cv2.imwrite(os.path.join(mask_dst, _type), img_t)

            elif os.path.isdir(path) and _type not in to_ignore:   # process recursively
                for filename in tqdm(os.listdir(path), desc=_type):
                    img = cv2.imread(os.path.join(path, filename), 0)
                    img_t = crop(img, filename, offsets)

                    out_path = os.path.join(mask_dst, _type)
                    os.makedirs(out_path, exist_ok=True)

                    cv2.imwrite(os.path.join(out_path, filename), img_t)
