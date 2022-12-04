import os
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from dataloader.datasets import ClassificationDataset, SegmentationDataset, SMPDataset
from dataloader.dataloader import get_loader, get_loader_seg, get_loader_smp
from PIL import Image
from numpy import asarray
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.transforms import ToTensor
import scipy.ndimage as ndimage
from pathlib import Path

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

def file_loader_for_testing_segmentation(file_name):
    img = Image.open(file_name)
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    val_transformation = A.Compose([
        A.Resize(height=256, width=256),
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

def get_loaders_segmentation(train_files, train_masks, val_files, val_masks, height, width, batch):
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    train_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Cutout(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        ToTensor(normalize=imagenet_stats)
            ])
        
    val_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Cutout(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        ToTensor(normalize=imagenet_stats)
            ])  
    # train_transform = A.Compose([
    #     A.Resize(height=height, width=width),
    #     ToTensor(normalize=imagenet_stats)
    #         ])
        
    # val_transform = A.Compose([
    #     A.Resize(height=height, width=width),
    #     ToTensor(normalize=imagenet_stats)
    #         ])  
    mask_transform = A.Compose([A.Resize(height=height, width=width), ToTensor()])
    
    train_dataset = SegmentationDataset(train_files, train_masks, train_transform, mask_transform)
    train_loader = get_loader_seg(dataset=train_dataset,batch_size=batch,shuffle=True)

    val_dataset = SegmentationDataset(val_files, val_masks, val_transform, mask_transform)
    val_loader = get_loader_seg(dataset=train_dataset,batch_size=batch,shuffle=True)
    
    return train_loader, val_loader

def get_loaders_smp(train_files, train_masks, val_files, val_masks, batch, height, width):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_dataset = SMPDataset(train_files, train_masks, mean, std, 'train', height, width)
    train_loader = get_loader_smp(dataset=train_dataset,batch_size=batch,shuffle=True)

    val_dataset = SMPDataset(val_files, val_masks, mean, std, 'val', height, width)
    val_loader = get_loader_smp(dataset=val_dataset,batch_size=batch,shuffle=True)
    
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
            
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    
    with open("testingResultsSegmentation/segmentationResults.txt", "w") as f:
        f.writelines(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
        f.writelines(f"Dice score: {dice_score/len(loader)}")
        f.writelines('*' * 100)
    return (num_correct/num_pixels)*100, dice_score/len(loader)

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    folder_pred = folder + 'pred/'
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder_pred}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)
