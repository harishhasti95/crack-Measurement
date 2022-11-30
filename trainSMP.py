# visualization library
import cv2
from matplotlib import pyplot as plt
# data storing library
import numpy as np
import pandas as pd
# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
# architecture and data split library
from utils import get_loaders_smp
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
# augmenation library
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
# others
import os
import pdb
import time
import warnings
import random
from tqdm import tqdm_notebook as tqdm
import concurrent.futures
import concurrent.futures
# warning print supression
warnings.filterwarnings("ignore")

# *****************to reproduce same results fixing the seed and hash*******************
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_loader, val_loader = get_loaders_smp('dataSegmentation/trainImage', 'dataSegmentation/trainMask', 'dataSegmentation/valImage', 'dataSegmentation/valMask', 1)


def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

''' initialize a empty list when Scores is called, append the list with dice scores
for every batch, at the end of epoch calculates mean of the dice scores'''
class Scores:
    def __init__(self, phase, epoch):
        self.base_dice_scores = []

    def update(self, targets, outputs):
        probs = outputs
        dice= dice_score(probs, targets)
        self.base_dice_scores.append(dice)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)         
        return dice

def epoch_log(epoch_loss, measure):
    '''logging the metrics at the end of an epoch'''
    dices= measure.get_metrics()    
    dice= dices                       
    print("Loss: %0.4f |dice: %0.4f" % (epoch_loss, dice))
    return dice



class Trainer(object):
    def __init__(self,model):
        self.num_workers=4
        self.batch_size={'train':1, 'val':1}
        self.accumulation_steps=4//self.batch_size['train']
        self.lr=5e-4
        self.num_epochs=10
        self.phases=['train','val']
        self.best_loss=float('inf')
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net=model.to(self.device)
        cudnn.benchmark= True
        self.criterion=torch.nn.BCEWithLogitsLoss()
        self.optimizer=optim.Adam(self.net.parameters(),lr=self.lr)
        self.scheduler=ReduceLROnPlateau(self.optimizer,mode='min',patience=3, verbose=True)
        self.dataloaders={'train': train_loader, 'val': val_loader}

        self.losses={phase:[] for phase in self.phases}
        self.dice_score={phase:[] for phase in self.phases}

    def forward(self, inp_images, tar_mask):
        inp_images=inp_images.to(self.device)
        tar_mask=tar_mask.to(self.device)
        pred_mask=self.net(inp_images)
        loss=self.criterion(pred_mask,tar_mask)
        return loss, pred_mask

    def iterate(self, epoch, phase):
        measure=Scores(phase, epoch)
        start=time.strftime("%H:%M:%S")
        print (f"Starting epoch: {epoch} | phase:{phase} | 🙊':{start}")
        batch_size=self.batch_size[phase]
        self.net.train(phase=="train")
        dataloader=self.dataloaders[phase]
        running_loss=0.0
        total_batches=len(dataloader)
        self.optimizer.zero_grad()
        for itr,batch in enumerate(dataloader):
            images,mask_target=batch
            loss, pred_mask=self.forward(images,mask_target)
            loss=loss/self.accumulation_steps
            if phase=='train':
                loss.backward()
                if (itr+1) % self.accumulation_steps ==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss+=loss.item()
            pred_mask=pred_mask.detach().cpu()
            measure.update(mask_target,pred_mask)
        epoch_loss=(running_loss*self.accumulation_steps)/total_batches
        dice=epoch_log(epoch_loss, measure)
        self.losses[phase].append(epoch_loss)
        self.dice_score[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss
    def start(self):
        for epoch in range (self.num_epochs):
            self.iterate(epoch,"train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss=self.iterate(epoch,"val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model_office.pth")
            print ()



model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)

temp = Trainer(model)
temp.start()