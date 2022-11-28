import os, shutil
import glob
from pathlib import Path

train_files = []
val_files = []
train_masks = []
val_masks = []

train = open('data/segmentation/train.txt').readlines()
test = open('data/segmentation/test.txt').readlines()
val = open('data/segmentation/val.txt').readlines()


for i in train:
    train_files.append(i.split(' ')[0])
    train_masks.append(i.split(' ')[1][:-1])
for i in test:
    train_files.append(i.split(' ')[0])
    train_masks.append(i.split(' ')[1][:-1])
for i in val:
    val_files.append(i.split(' ')[0])
    val_masks.append(i.split(' ')[1][:-1])




for i in train_files:
    path = os.path.join('data/segmentation/', i)
    shutil.copy2(path, 'dataSegmentation/trainImage/')
for i in train_masks:
    path = os.path.join('data/segmentation/', i)
    shutil.copy2(path, 'dataSegmentation/trainMask/')
for i in val_files:
    path = os.path.join('data/segmentation/', i)
    shutil.copy2(path, 'dataSegmentation/valImage/')
for i in val_masks:
    path = os.path.join('data/segmentation/', i)
    shutil.copy2(path, 'dataSegmentation/valMask/')