"""
Converts master bounding boxes to YOLO format.
"""

import argparse
import os
import cv2
import json
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts master bounding boxes to YOLO format.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src', '-s', type=str, required=True,
                            help='CSV file containing labels')
    parser.add_argument('--dst', '-d', type=str, required=True,
                            help='directory to save COCO annotations.')
    parser.add_argument('--mode', type=str, choices=["train", "test"], required=True,
                            help="to append to --dst / -d.")
    parser.add_argument('--skip', type=str,
                            help="comma separated list of directory names (not path) to skip\ne.g., --skip class_1,class_2.")
    parser.add_argument('--thresh', '-t', type=int, default=20,
                            help='bounding box threshold')
    parser.add_argument('--base', '-b', type=str, default='',
                            help='directory to containing images paths in CSV')

    args = parser.parse_args()

    # retrieving data

    if os.path.basename(args.dst) == "annotations" or os.path.dirname(args.dst) == "annotations":
        dst = args.dst
    else:
        dst = os.path.join(args.dst, "annotations")
    
    print("Output Directory:", dst)
    src = args.src
    if args.skip:
        to_ignore = [name.strip() for name in args.skip.split(',')]
    else:
        to_ignore = []

    # Annotations JSON
    annotations = dict()       # using format: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format

    # standard stuff
    annotations["info"] = {
        "description": "IDRiD Dataset",
        "url": "https://idrid.grand-challenge.org/",
        "version": "1.0",
        "year": 2018,
        "contributor": "EEE International Symposium on Biomedical Imaging",
        "date_created": "2017/10/25"
    }

    annotations["licenses"] = [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ]
    
    os.makedirs(dst, exist_ok=True)

    # Processing
    records = pd.read_csv(src)

    recs = records.groupby('class')
    classes = [col for col in list(recs.groups.keys()) if col not in to_ignore]

    # Adding categories to JSON
    annotations["categories"] = list()
    for i, cls in enumerate(classes):
        annotations["categories"].append({"supercategory": "", "id": i+1, "name": cls})
    
    recs = recs.filter(lambda t: t['class'].tolist()[0] not in to_ignore)
    recs = recs.groupby('image_path')

    images = list(recs.groups.keys())     # get all image paths

    annotations["images"] = list()
    annotations["annotations"] = list()

    for _id, img in tqdm(enumerate(images)):      # iterate over images
        grp = recs.get_group(img)     # get all bounding boxes for that image

        count = 0
        for i, row in grp.iterrows():   # iterate all bounding boxes
            class_id = classes.index(row['class'])
            
            width = row['x2'] - row['x1']
            height = row['y2'] - row['y1']

            if width < args.thresh or height < args.thresh:
                continue
                
            count += 1

            # Adding annotations to JSON
            annotations["annotations"].append({"segmentation": [],
                                                "area": width*height,
                                                "iscrowd": 0,
                                                "image_id": _id + 1,
                                                "bbox": [row['x1'],row['y1'],width,height], #  [top left x position, top left y position, width, height]
                                                "category_id": classes.index(row['class']) + 1,
                                                "id": _id*len(images) + i})
    
        if count:
            image = cv2.imread(os.path.join(args.base, img))
            # Adding image to JSON
            annotations["images"].append({
                "license": 4,
                "file_name": os.path.basename(img),
                "coco_url": "",
                "height": image.shape[0],
                "width": image.shape[1],
                "date_captured": "2013-11-14 17:02:52",
                "coco_url": "",
                "id": _id + 1
            })
    
    # saving to disk
    with open(os.path.join(dst, "instances_"+args.mode+".json"), 'w') as out:
        json.dump(annotations, out)