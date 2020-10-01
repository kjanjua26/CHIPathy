"""
Converts master bounding boxes to YOLO format.
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts master bounding boxes to YOLO format.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src', '-s', type=str, required=True,
                            help='CSV file containing labels')
    parser.add_argument('--dst', '-d', type=str, required=True,
                            help='directory to save bounding boxes')
    parser.add_argument('--mode', type=str, choices=["train", "test"],
                            help="to append to --dst / -d.\nUseful when you have `train` and `test` splits in separate directories.")
    parser.add_argument('--skip', type=str,
                            help="comma separated list of directory names (not path) to skip\ne.g., --skip class_1,class_2.")
    parser.add_argument('--thresh', '-t', type=int, default=20,
                            help='bounding box threshold')

    args = parser.parse_args()

    # retrieving data
    file_path = os.path.join(args.dst, args.mode) if args.mode else os.path.join(args.dst, os.path.basename(src))
    dst = os.path.join(args.dst, args.mode) if args.mode else args.dst
    src = args.src
    if args.skip:
        to_ignore = [name.strip() for name in args.skip.split(',')]
    else:
        to_ignore = []
    
    os.makedirs(dst, exist_ok=True)

    # Processing
    records = pd.read_csv(src)

    recs = records.groupby('class')
    classes = [col for col in list(recs.groups.keys()) if col not in to_ignore]

    with open(file_path+'.names', 'w') as out:
        out.write('\n'.join(classes))
    
    recs = recs.filter(lambda t: t['class'].tolist()[0] not in to_ignore)
    recs = recs.groupby('image_path')

    images = list(recs.groups.keys())     # get all image paths

    out = open(file_path+'.txt', 'w')

    for img in tqdm(images):      # iterate over images
        grp = recs.get_group(img)     # get all bounding boxes for that image

        count = 0
        with open(os.path.join(dst, os.path.splitext(os.path.basename(img))[0] + '.txt'), 'w') as labels_out:
            for i, row in grp.iterrows():   # iterate all bounding boxes
                class_id = classes.index(row['class'])
                
                width = row['x2'] - row['x1']
                height = row['y2'] - row['y1']

                if width < args.thresh or height < args.thresh:
                    continue
                    
                count += 1

                x_center = (row['x1'] + width / 2) / row['width']
                y_center = (row['y1'] + height / 2) / row['height']

                width =  width / row['width']
                height = height / row['height']

                labels_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        if count:
            out.write(os.path.abspath(img) + '\n')