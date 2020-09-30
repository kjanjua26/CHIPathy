"""
Generates bounding boxes from grayscale masks and dumps them in csv in following format:
image_path, class_id, x1, y1, x2, y2, height, width -> one record/row for each object
"""

import argparse
import os
import sys
import cv2
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops      # ref: https://scikit-image.org/docs/dev/api/skimage.measure.html

from preprocess_cropnblur import remove_ext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates bounding boxes from grayscale masks and dumps them in CSV as \
                                                    image_path, class_id, x1, y1, x2, y2, height, width",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--src', '-s', type=str, required=True,
                            help='source directory containing sub directories for each mask type')
    parser.add_argument('--base', '-b', type=str, required=True,
                            help='directory path to be used as base path for original images')
    parser.add_argument('--dst', '-d', type=str, default='.',
                            help='directory OR full CSV path to save generated offsets')
    parser.add_argument('--ext', '-e', type=str, default='jpg',
                            help='extension of original images in --base / -b')
    parser.add_argument('--mode', '-m', type=str, choices=["train", "test"],
                            help="to be added at the end of --src / -s. --base / -b, and --dst / -d. Useful when you have `train` and `test` splits in separate directories")
    parser.add_argument('--skip', type=str,
                            help="comma separated list of directory names (not path) to skip e.g., --skip class_1,class_2")
    
    args = parser.parse_args()

    # retrieving data
    dst = os.path.join(args.dst, args.mode) if args.mode else args.dst
    base = os.path.join(args.base, args.mode) if args.mode else args.base
    src = os.path.join(args.src, args.mode) if args.mode else args.src
    if args.skip:
        to_ignore = [name.strip() for name in args.skip.split(',')]
    else:
        to_ignore = []

    # fixing paths
    base_path = os.path.dirname(dst)
    os.makedirs(base_path, exist_ok=True)

    name, ext = os.path.splitext(os.path.basename(dst))
    if not ext:
        name = os.path.basename(src)+'_bboxes.csv'
    else:
        name = name + '.csv'
    bboxes_out = os.path.join(base_path, name)

    print("Destination path is not a CSV file. Output will be saved in", bboxes_out)

    records = pd.DataFrame()
    
    files = []
    for i, item in enumerate(os.listdir(src)):
        path = os.path.join(src, item)

        if not os.path.isdir(path) or item in to_ignore: continue

        for file in tqdm(os.listdir(path), desc=item):

            image_path = os.path.join(base, remove_ext(file)+'.'+args.ext)

            if not os.path.isfile(image_path):
                print("\nOriginal image not found at ", image_path)
                print("Check --base and --ext and try again. Skipping...")
                continue

            file_path = os.path.join(path, file)

            if os.path.isfile(file_path) and file.endswith(("jpg", "png", "tif")):
                image = cv2.imread(file_path, 0)

                height, width = image.shape

                dim = min(height, width)
                size = dim if dim % 2 else dim - 1 # choose odd value

                # create binary mask
                mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, 0)

                # convert segmentation masks to bounding boxes
                label_mask = label(mask)
                props = regionprops(label_mask)

                for i, prop in enumerate(props):
                    if prop.bbox == (0, 0, height, width) or \
                        prop.bbox[2] - prop.bbox[0] == prop.bbox[3] - prop.bbox[1] == 1:
                        
                        continue
                    
                    record = {'image_path': image_path, 'class': item,
                                'x1': prop.bbox[1], 'y1': prop.bbox[0],     # [0] -> min_row, [1] -> min_col. See ref on top.
                                'x2': prop.bbox[3], 'y2': prop.bbox[2],     # [2] -> max_row, [3] -> max_col. See ref on top.
                                'height': height, 'width': width}
                    records = records.append(record, ignore_index=True)

    if len(records.index) != 0:
        # save to disk
        records = records.astype(dtype={"x1":"int32", "y1":"int32",
                                        "x2":"int32", "y2":"int32",
                                        "height":"int32", "width":"int32"})
        records.to_csv(bboxes_out, columns=['image_path', 'class', 'x1', 'y1', 'x2', 'y2', 'height', 'width'], index=False)