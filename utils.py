import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from dataloader.datasets import ClassificationDataset
from dataloader.dataloader import get_loader
from PIL import Image
from numpy import asarray
import pickle
import albumentations as A
from albumentations.pytorch.transforms import ToTensor


def prepare_train_test_val(dir):
    if 'dataset.txt' in os.listdir(os.getcwd()):
        with open('dataset.txt', 'rb') as f:
            loaded = pickle.load(f)
            return loaded['X_train'], loaded['X_val'], loaded['Y_train'], loaded['Y_val']
    class_names = os.listdir(dir)
    num_class = len(class_names)
    image_files = [[os.path.join(dir, class_name, x) for x in os.listdir(os.path.join(dir, class_name))] for class_name in class_names]

    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    num_total = len(image_label_list)

    valid_frac = 0.1
    X_train,Y_train = [],[]
    X_val,Y_val = [],[]

    for i in range(num_total):
        rann = np.random.random()
        if rann < valid_frac:
            X_val.append(image_file_list[i])
            Y_val.append(image_label_list[i])
        else:
            X_train.append(image_file_list[i])
            Y_train.append(image_label_list[i])

    def convert_to_numpy_array(arr):
        return np.array(arr)


    X_train = convert_to_numpy_array(X_train)
    X_val = convert_to_numpy_array(X_val)
    Y_train = convert_to_numpy_array(Y_train)
    Y_val = convert_to_numpy_array(Y_val)
    
    
    d = {'X_train':X_train,'X_val':X_val,'Y_train': Y_train, 'Y_val': Y_val}
    with open('dataset.txt', 'wb') as f:
        pickle.dump(d,f)  
        
    
    return X_train, X_val, Y_train, Y_val

def file_loader_for_testing(file_name):
    img = Image.open(file_name)
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    val_transformation = A.Compose([
        A.Cutout(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        ToTensor(normalize=imagenet_stats)
            ])
    if val_transformation:
        img = val_transformation(**{"image": np.array(img)})["image"]
    img = img.to('cuda')
    temp = img.size()
    img = img.reshape(1, temp[0], temp[1], temp[2])

    return img
    

def get_loaders(X_train, Y_train, X_val, Y_val, batch):
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    train_transformation = A.Compose([
        A.Cutout(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        ToTensor(normalize=imagenet_stats)
            ])
        
    val_transformation = A.Compose([
            ToTensor(normalize=imagenet_stats)
            ])  
    train_dataset = ClassificationDataset(X_train, Y_train, train_transformation)
    train_loader = get_loader(train_dataset, batch_size=batch, shuffle=True)

    val_dataset = ClassificationDataset(X_val, Y_val, val_transformation)
    val_loader = get_loader(val_dataset, batch_size=batch, shuffle=True)
    
    return train_loader, val_loader

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False