"""
Converts segmentation masks of different types from different folders to YOLO format bounding boxes.
"""

import os
import sys
import cv2
from tqdm import tqdm

from skimage.measure import label, regionprops


if __name__ == "__main__":
    GT_PATH = sys.argv[1]

    labels = os.listdir(GT_PATH)

    # conversion
    for class_id, class_label in enumerate(labels):
        class_path = os.path.join(GT_PATH, class_label)

        if not os.path.isdir(class_path): continue

        for file in tqdm(os.listdir(class_path), desc=class_label):
            mask_path = os.path.join(class_path, file)
            mask = cv2.imread(mask_path)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            height, width, _ = mask.shape

            # convert segmentation masks to bounding boxes
            label_mask = label(mask)
            props = regionprops(label_mask)

            with open(os.path.join(GT_PATH, file[:8]+".txt"), 'a') as f:
                for prop in props:
                    # write annotation in YOLO format
                    f.write("{_id} {x} {y} {width} {height}\n".format(_id=class_id, 
                                                                        x=prop.bbox[0] / width, y=prop.bbox[1] / height,
                                                                        width=(prop.bbox[3] - prop.bbox[0]) / width, 
                                                                        height=(prop.bbox[4] - prop.bbox[1]) / height))