"""
Splits images in given directory into square patches.
"""

import argparse
import os
import cv2
from tqdm import tqdm
import multiprocessing as mp

from utils import roundmultiple


def create_patches(image, size, out_dir, filename):
    """
    Creates square SIZEd patches from IMAGE and saves them in OUT_DIR
    """

    count = 0
    for i in range(0, image.shape[0], size):
        for j in range(0, image.shape[1], size):
            count += 1
            
            patch = image[i:i+size, j:j+size]
            cv2.imwrite(os.path.join(out_dir, str(count).zfill(5)+"_"+filename), patch)


def process_image(image_path, patch_size, destination, gray):
    image = cv2.imread(image_path)

    height, width, dim = image.shape
    new_height = roundmultiple(height, patch_size)
    new_width = roundmultiple(width, patch_size)

    image = cv2.resize(image, (new_width, new_height))
    if gray and dim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    create_patches(image, patch_size, destination, os.path.basename(image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splits images in given directory into square patches. \
                                                Use preprocess_cropnblur.py for croping masks and images to make them same size before creating patches.",
                                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--src', '-s', type=str, required=True,
                            help='source directory containing images')
    parser.add_argument('--dst', '-d', type=str, required=True,
                            help='directory to save patches')
    parser.add_argument('--size', '-p', type=int, default=64,
                            help='size of the square patch')
    parser.add_argument('--mode', type=str, choices=["train", "test"],
                            help="to append to --src / -s and --dst / -d.\nUseful when you have `train` and `test` splits in separate directories.")
    parser.add_argument('--gray', action='store_true',
                            help='convert image to grayscale before creating patches.\nRECOMMENDED for masks.')
    parser.add_argument('--recursive', action='store_true',
                            help='recursively check for immediate directories in source')
    parser.add_argument('--skip', type=str,
                            help="comma separated list of directory names (not path) to skip\ne.g., --skip class_1,class_2.\nUsed only when --recursive is set.")

    args = parser.parse_args()

    # retrieving data
    dst = os.path.join(args.dst, args.mode) if args.mode else args.dst
    src = os.path.join(args.src, args.mode) if args.mode else args.src
    if args.skip:
        to_ignore = [name.strip() for name in args.skip.split(',')]
    else:
        to_ignore = []

    os.makedirs(dst, exist_ok=True)

    files = []
    print("Loading files....")
    for item in tqdm(os.listdir(src)):
        path = os.path.join(src, item)

        if os.path.isfile(path) and item.endswith(("jpg", "png", "tif")):
            files.append((path, args.size, dst, args.gray))

        elif os.path.isdir(path) and args.recursive and item not in to_ignore:
            for file in tqdm(os.listdir(path), desc=item):
                file_path = os.path.join(path, file)

                if os.path.isfile(file_path) and file.endswith(("jpg", "png", "tif")):
                    out_path = os.path.join(dst, item)
                    os.makedirs(out_path, exist_ok=True)
                    files.append((file_path, args.size, out_path, args.gray))

    print("Processing files....")
    with mp.Pool(mp.cpu_count()) as p:
        p.starmap(process_image, files)