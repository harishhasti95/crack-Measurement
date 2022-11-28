from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import torchvision.transforms.functional as TF
from random import random


class ClassificationDataset(Dataset):
    def __init__(self, img_dir, label_name, image_transformation_function):
        self.img_dir = img_dir
        self.label_name = label_name
        self.image_transformation_function = image_transformation_function

    def __getitem__(self, i):
        img = Image.open(self.img_dir[i])
        
        if self.image_transformation_function:
            img = self.image_transformation_function(**{"image": np.array(img)})["image"]
        return img, self.label_name[i]

    def __len__(self):
        return len(self.img_dir)

class SegmentationDataset(Dataset):
    def __init__(self, imgages_dir, masks_dir, transform_function_image, transform_function_mask):
        self.imgages_dir = imgages_dir
        self.masks_dir = masks_dir
        self.transform_function_image = transform_function_image
        self.transform_function_mask = transform_function_mask
        self.images = os.listdir(imgages_dir)
        self.masks = os.listdir(masks_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        image_dir = os.path.join(self.imgages_dir, self.images[i])
        mask_dir = os.path.join(self.masks_dir, self.masks[i])
            
        image = np.array(Image.open(image_dir).convert("RGB"))
        mask = np.array(Image.open(mask_dir).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform_function_image is not None and self.transform_function_mask is not None:
            image = self.transform_function_image(image=image)["image"]
            mask = self.transform_function_mask(image=mask)["image"]
            

        return image, mask