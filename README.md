# CHIPathy

Diabetic Retinopathy Detection using PatchCNN Networks.

## CHIPathyDB - The Retinopathy Detection Database

### Steps to Re-produce the Train/Test Dataset
> Since the available data is highly limited and contains segmentation masks, we convert those to bounding boxes for detection and localization.

1. Download diabetic retinopathy dataset from either of these databases:
    - https://idrid.grand-challenge.org/
    - http://www2.it.lut.fi/project/imageret/diaretdb1/
    
2. Pre-processed image using Gray Scale Cropping and Weighted Gaussian Blur. To reproduce the results, run ```PreProcessing.ipynb```.
3. Use the offsets obtained from Step # 02 to crop the masks as well in the similar manner, run the file ```crop_masks.ipynb``` to reproduce results.
4. Divide the masks and images into patches of size 64x64, run ```patches.ipynb``` to get the results.
5. Convert the segmentation masks to bounding boxes and prepare the dataset for EfficientDet. To obtain the results run ```masks2boxes_patches.py``` file.
6. To get the data for Yolo, run the file ```masks2boxes_patches_yolo.py``` instead.
