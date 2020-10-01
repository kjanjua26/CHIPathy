"""
Augment the images in the directory and generate multiple images from that \
    to increase the numnber of available data.
"""
import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Augment the images in the directory and generate multiple images.", 
                                formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--src', '-s', type=str, required=True,
                    help='The source directory containing the images.')
parser.add_argument('--dst', '-d', type=str, required=True,
                    help='The destination directory to save the images.')
parser.add_argument('--techniques', '-t', type=str,
                    help='The techniques to apply to augment the images.\nBy default all the techniques will be applied.')
args = parser.parse_args()

class Augment:
    def __init__(self):
        self.input_dir = args.src
        self.output_dir = args.dst
        self.techniques = args.techniques
        self.zoom_factors = [1.15, 1.25, 1.35, 1.45, 1.50, 1.60, 1.70, 1.80, 1.90]
        self.severity = [1, 2, 3, 4, 5]
        self.count = 0

        if self.techniques is not None:
            print(f'The augmentations to apply are: {[x for x in self.techniques]}')
        else:
            print("Applying all the augmentations.")
        

    def crop_image_from_gray(self, img, tol=10):
        dims = img.ndim

        if dims == 3:
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif dims == 2:
            gray_image = img

        mask = gray_image > tol

        # find start crop point
        objs = ndimage.find_objects(mask)
        height_offset = objs[0][0].start
        width_offset = objs[0][1].start
        ix = np.ix_(mask.any(1),mask.any(0))

        if dims == 2:
            return img[ix], height_offset, width_offset
        
        check_shape = img[:,:,0][ix].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img, 0, 0 # return original image

        img1 = img[:,:,0][ix]
        img2 = img[:,:,1][ix]
        img3 = img[:,:,2][ix]
        img = np.stack([img1, img2, img3],axis=-1)

        return img, height_offset, width_offset
    
    def circle_crop(self, img, sigmaX=50):   
        """
        Create circular crop around image centre    
        """
        img, height_offset, width_offset = self.crop_image_from_gray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)

        return img, height_offset, width_offset
    
    def flip_image(self, img, mode='h'):
        """
        Flips the image horizontally or vertically depending on the mode.
        """

        if mode == 'h':
            aug = iaa.Fliplr()
        else:
            aug = iaa.Flipud()
            
        flipped_img = aug(image=img)
        return flipped_img
    
    def compression_blur(self, img, severity):
        """
        Blur in an image caused due to the JPEG compression.
        """
        aug = iaa.imgcorruptlike.JpegCompression(severity=severity)
        return aug(image=img)
    
    def cv2_clipped_zoom(self, img, zoom_factor):
        """
        Center zoom in/out of the given image and returning an enlarged/shrinked view of 
        the image without changing dimensions
        Args:
            img : Image array
            zoom_factor : amount of zoom as a ratio (0 to Inf)

        Ftn Taken From: https://stackoverflow.com/a/48097478/6735773
        """
        height, width = img.shape[:2] # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

        ### Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1,x1,y2,x2])
        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

        result = cv2.resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode='constant')
        assert result.shape[0] == height and result.shape[1] == width
        return result
    
    def main(self):
        """
        The main caller function of the augmentation file.
        """

        exts = glob(self.input_dir + "*.jpeg") + glob(self.input_dir + "*.png") + \
                glob(self.input_dir + "*.jpg") + glob(self.input_dir + "*.tif")
        list_of_images = [x for x in exts]
        list_of_transformed_images = []

        for i, img_path in enumerate(list_of_images):
            print(f"Processing {i}/{len(list_of_images)}")
            img = cv2.imread(img_path)
            
            print("Applying the base transformation: CropnBlur.")
            img_t, height_offset, width_offset = self.circle_crop(img, 50)
            list_of_transformed_images.append(img_t)
            
            print("Applying the horizontal/vertical flips.")
            img_t_fliplr = self.flip_image(img_t)
            list_of_transformed_images.append(img_t_fliplr)

            img_t_flipud = self.flip_image(img_t, mode='v')
            list_of_transformed_images.append(img_t_flipud)
            
            print("Applying zoom transformation.")
            for zoom_factor in self.zoom_factors:
                zoomed_img = self.cv2_clipped_zoom(img_t, zoom_factor)
                list_of_transformed_images.append(zoomed_img)
            
            print("Applying JPEG compression blur.")
            for sev in self.severity:
                compressed_img_t = self.compression_blur(img_t, sev)
                compressed_img_fliplr = self.compression_blur(img_t_fliplr, sev)
                compressed_img_flipud = self.compression_blur(img_t_flipud, sev)

                list_of_transformed_images.append(compressed_img_t)
                list_of_transformed_images.append(compressed_img_fliplr)
                list_of_transformed_images.append(compressed_img_flipud)

        for transformed_img in tqdm(list_of_transformed_images):
            self.count += 1
            transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.output_dir + f"{self.count}.jpg", transformed_img)

if __name__ == "__main__":
    ag = Augment()
    ag.main()