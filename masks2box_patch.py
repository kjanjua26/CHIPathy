"""
Converts segmentation masks of different types from different folders to YOLO format bounding boxes.
"""

import os
import sys
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import label, regionprops

class_name = "Hard_Exudates"

if __name__ == "__main__":
    MASKS_PATH = sys.argv[1]
    OUT_PATH = sys.argv[2]

    out = open(OUT_PATH, "w")

    for image in os.listdir(MASKS_PATH):
        image_path = os.path.join(MASKS_PATH, image)
        if not os.path.isdir(image_path): continue

        for patch in tqdm(os.listdir(image_path), desc=image):
            patch_path = os.path.join(image_path, patch)

            orig_mask = cv2.imread(patch_path, 0)
            mask = cv2.adaptiveThreshold(orig_mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,63,0)

            # convert segmentation masks to bounding boxes
            label_mask = label(mask)
            props = regionprops(label_mask)

            show = False
            for i, prop in enumerate(props):
                if prop.bbox == (0, 0, 64, 64) or \
                    prop.bbox[2] - prop.bbox[0] == prop.bbox[3] - prop.bbox[1] == 1: continue
                # print(patch, prop.bbox)
                # if patch == "404.jpg" and "IDRiD_55" in image:
                #     f, ax = plt.subplots(2)
                #     ax[0].imshow(mask, cmap="gray")
                #     ax[1].imshow(orig_mask, cmap="gray")
                #     rect = Rectangle((prop.bbox[1], prop.bbox[0]), prop.bbox[3]-prop.bbox[1], prop.bbox[2]-prop.bbox[0], facecolor='none', edgecolor='r', linewidth=2)
                #     ax[0].add_patch(rect)
                #     plt.show()
                
                out.write("{path},{x1},{y1},{x2},{y2},{class_name}\n".format(path=patch_path,
                                                                        x1=prop.bbox[0], y1=prop.bbox[1], x2=prop.bbox[2], y2=prop.bbox[3],
                                                                        class_name=class_name))

                
    out.close()