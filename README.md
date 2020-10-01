# CHIPathy

Diabetic Retinopathy Detection using PatchCNN Networks.

## CHIPathyDB - The Retinopathy Detection Database

### Steps to Re-produce the Train/Test Dataset
> Since the available data is highly limited and contains segmentation masks, we convert those to bounding boxes for detection and localization.

1. Download diabetic retinopathy dataset from either of these databases:
    - https://idrid.grand-challenge.org/
    - http://www2.it.lut.fi/project/imageret/diaretdb1/
    
2. Pre-processed image using Gray Scale Cropping and Weighted Gaussian Blur. To reproduce the results, run ```dataprep/preprocess_cropnblur.py``` file.
3. Divide the masks and images into patches of size 64x64, run ```dataprep/patches.py``` to get the results.
4. Convert the segmentation masks to bounding boxes. To obtain the results run ```dataprep/masks2boxes.py``` file.
5. Convert these bounding boxes to different formats as required by the model using specific conversion scripts. e.g., for YOLO use ```dataprep/prepdata_yolo.py``` file. 
