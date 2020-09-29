"""
Splits images in given directory into square patches.
"""

import argparse
import os
import cv2
from tqdm import tqdm


def roundmultiple(x, base):
    """
    Rounds X to the nearest multiple of BASE.
    """

    return base * round(x/base)


def create_patches(image, size, out_dir):
    """
    Creates square SIZEd patches from IMAGE and saves them in OUT_DIR
    """

    count = 0
    for i in range(0, image.shape[0], size):
        for j in range(0, image.shape[1], size):
            count += 1
            
            patch = image[i:i+size, j:j+size]
            cv2.imwrite(os.path.join(out_dir, str(count)+".jpg"), patch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splits images in given directory into square patches. \
                                                Use preprocess_cropnblur.py for croping masks and images to make them same size before creating patches.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src', '-s', type=str, required=True,
                            help='source directory containing images')
    parser.add_argument('--dst', '-d', type=str, required=True,
                            help='directory to save patches')
    parser.add_argument('--patch', '-p', type=int, default=64,
                            help='size of the square patch')
    parser.add_argument('--gray', action='store_true',
                            help='convert image to grayscale before creating patches. RECOMMENDED for masks.')

    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    for image_path in tqdm(os.listdir(args.src)):
        image = cv2.imread(os.path.join(args.src, image_path))

        height, width, dim = image.shape
        new_height = roundmultiple(height, args.patch)
        new_width = roundmultiple(width, args.patch)

        image = cv2.resize(image, (new_width, new_height))
        if args.gray and dim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        patches_out_dir = os.path.join(args.dst, os.path.basename(image_path)[:-4])
        if not os.path.isdir(patches_out_dir):
            os.makedirs(patches_out_dir)
        
        create_patches(image, args.patch, patches_out_dir)