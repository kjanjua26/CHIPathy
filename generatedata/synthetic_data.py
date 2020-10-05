"""
Generate the synthetic dataset.
"""
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate the synthetic dataset.", 
                                formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--src', '-s', type=str, required=True,
                    help='The source directory containing the perfect fundus images.')
parser.add_argument('--dst', '-d', type=str, required=True,
                    help='The destination directory to save the corrupted/affected images.')
args = parser.parse_args()

class SyntheticGenerator:
    def __init__(self):
        self.input_dir = args.src
        self.output_dir = args.dest
        self.thres = 50
    
    def get_candidate_row_from_df(cls):
        """
        Gets the patch from the image of a given class.
        """
        recs = df.groupby('class')
        recs = recs.filter(lambda t: t['class'].tolist()[0] == cls)
        list_of_candidate_rows = []
        for i, row in recs.iterrows():
            width = row['x2'] - row['x1']
            height = row['y2'] - row['y1']
            if width < self.thres or height < self.thres: continue
            list_of_candidate_rows.append(row)
        candidate = random.choice(list_of_candidate_rows)
        return candidate
    
    def get_patch_from_image(cls):
        """
        Return the patch required to paste onto the perfect images of fundus.
        """
        path, _, x1, y1, x2, y2, height, width = get_candidate_row_from_df(cls)
        img = cv2.imread(path)
        patch = img[y1: y2, x1: x2]
        return patch
    
    def apply_patch_on_the_image(img, patch, count=5, offset=150):
        """
        Applies the patch on the main image and returns the mask and bounding box as well.
        """
        mask = np.zeros(shape=img.shape)
        boxes = []
        prev = (0, 0)
        gen = gencoordinates(img.shape[0], img.shape[1])
        for i in range(count):
            rnd = random.choice([x for x in range(100)])
            x_offset = rnd + patch.shape[0]
            y_offset = rnd + patch.shape[1]
            x_offset += prev[0]
            y_offset += prev[1]
            if y_offset < patch.shape[1]:
                y_offset = patch.shape[1]
            if x_offset < patch.shape[0]:
                x_offset = patch.shape[0]
            img[y_offset:y_offset+patch.shape[0], x_offset:x_offset+patch.shape[1]] = patch
            mask[y_offset:y_offset+patch.shape[0], x_offset:x_offset+patch.shape[1]] = 1
            boxes.append((y_offset, patch.shape[0], x_offset, patch.shape[1]))
            prev = (x_offset, y_offset)
        return img, mask, boxes