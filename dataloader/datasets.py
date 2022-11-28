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
    def __init__(self, imgages_dir, masks_dir, transform_function = None):
        self.imgages_dir = imgages_dir
        self.masks_dir = masks_dir
        self.transform_function = transform_function
        self.images = os.listdir(imgages_dir)
        
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_dir = os.path.join(self.imgages_dir, self.images[index])
        mask_dir = os.path.join(self.masks_dir, self)