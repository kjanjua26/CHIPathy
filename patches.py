"""
Splits images in given directory into square patches.
"""

import argparse
import os
import cv2
from tqdm import tqdm
import multiprocessing as mp


def roundmultiple(x, base):
    """
    Rounds X to the nearest multiple of BASE.
    """

    return base * round(x/base)


def create_patches(image, size, out_dir, filename):
    """
    Creates square SIZEd patches from IMAGE and saves them in OUT_DIR
    """

    count = 0
    for i in range(0, image.shape[0], size):
        for j in range(0, image.shape[1], size):
            count += 1
            
            patch = image[i:i+size, j:j+size]
            cv2.countNonZero(image) != 0
            cv2.imwrite(os.path.join(out_dir, str(count).zfill(5)+"_"+filename), patch)


def process_image(image_path, destination, gray, patch_size):
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
    parser.add_argument('--patch', '-p', type=int, default=64,
                            help='size of the square patch')
    parser.add_argument('--gray', action='store_true',
                            help='convert image to grayscale before creating patches.\nRECOMMENDED for masks.')
    parser.add_argument('--recursive', action='store_true',
                            help='recursively check for immediate directories in source')

    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    files = []
    print("Loading files....")
    for img in tqdm(os.listdir(args.src)):
        path = os.path.join(args.src, img)

        if os.path.isfile(path) and img.endswith(("jpg", "png", "tif")):
            files.append((path, args.dst, args.gray, args.patch))
        elif os.path.isdir(path) and args.recursive:
            for file in tqdm(os.listdir(path), desc=img):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path) and file.endswith(("jpg", "png", "tif")):
                    out_path = os.path.join(args.dst, img)
                    os.makedirs(out_path, exist_ok=True)
                    files.append((file_path, out_path, args.gray, args.patch))

    print("Processing files....")
    with mp.Pool(mp.cpu_count()) as p:
        p.starmap(process_image, files)