from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os, cv2
import torchvision.transforms.functional as TF
from random import random
import torch
from skimage.io import imread
from torch.utils import data

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
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        image_dir = os.path.join(self.imgages_dir, self.images[i])
        mask_dir = os.path.join(self.masks_dir, self.masks[i])
        image = np.array(Image.open(image_dir).convert("RGB"))
        mask = np.array(Image.open(mask_dir).convert("L"), dtype=np.float32)
        # print(mask)
        mask[mask == 255.0] = 1.0
        if self.transform_function_image is not None and self.transform_function_mask is not None:
            image = self.transform_function_image(image=image)["image"]
            mask = self.transform_function_mask(image=mask)["image"]
        return image, mask

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
def get_transform(phase,mean,std):
    list_trans=[]
    if phase=='train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Resize(height=256, width=256), Normalize(mean=mean,std=std, p=1), ToTensor()])  #normalizing the data & then converting to tensors
    list_trans=Compose(list_trans)
    return list_trans

class SMPDataset(Dataset):
    def __init__(self, imgages_dir, masks_dir, mean, std, phase):
        self.imgages_dir = imgages_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(imgages_dir)
        self.masks = os.listdir(masks_dir)
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.trasnform=get_transform(phase,mean,std)
    def __getitem__(self, i):
        image_dir = os.path.join(self.imgages_dir, self.images[i])
        mask_dir = os.path.join(self.masks_dir, self.masks[i])
        img=cv2.imread(image_dir)
        mask=cv2.imread(mask_dir,cv2.IMREAD_GRAYSCALE)
        augmentation=self.trasnform(image=img, mask=mask)
        img_aug=augmentation['image']                           #[3,128,128] type:Tensor
        mask_aug=augmentation['mask']                           #[1,128,128] type:Tensor
        return img_aug, mask_aug

    def __len__(self):
        return len(self.images)
