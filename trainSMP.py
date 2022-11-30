# data storing library
import numpy as np
# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
# architecture and data split library
from utils import get_loaders_smp
import segmentation_models_pytorch as smp

# others
import os
import argparse
import time
import warnings
import random
from tqdm import tqdm_notebook as tqdm
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


def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

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
    dices= measure.get_metrics()    
    dice= dices                       
    print("Loss: %0.4f |dice: %0.4f" % (epoch_loss, dice))
    return dice

def forward(model, criterion, input, mask, device):
    input=input.to(device)
    mask=mask.to(device)
    predicted_mask=model(input)
    loss=criterion(predicted_mask,mask)
    return loss, predicted_mask

def iterate(args, model, epoch, phase, dataloaders, optimizer, criterion, device, accumulation_steps, losses, dice_scores):
        measure=Scores(phase, epoch)
        start=time.strftime("%H:%M:%S")
        print (f"Starting epoch: {epoch} | phase:{phase} | 🙊':{start}")
        model.train(phase=="train")
        dataloader=dataloaders[phase]
        running_loss=0.0
        total_batches=len(dataloader)
        optimizer.zero_grad()
        for itr,batch in enumerate(dataloader):
            images,mask_target=batch
            loss, pred_mask=forward(model, criterion, images,mask_target, device)
            loss=loss/accumulation_steps
            if phase=='train':
                loss.backward()
                if (itr+1) % accumulation_steps ==0:
                    optimizer.step()
                    optimizer.zero_grad()
            running_loss+=loss.item()
            pred_mask=pred_mask.detach().cpu()
            measure.update(mask_target,pred_mask)
        epoch_loss=(running_loss*accumulation_steps)/total_batches
        dice=epoch_log(epoch_loss, measure)
        losses[phase].append(epoch_loss)
        dice_scores[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss

def main():
    parser = argparse.ArgumentParser(description='PyTorch Crack Classification')
    parser.add_argument('--load', type=bool, default=True, metavar='N',
                        help='Load Pretrained model from checkpoint')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                        help='Learning rate for training (default: 5e-4)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--num_epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--height', type=int, default=256, metavar='N',
                        help='Height of the image to resize (default: 240)')
    parser.add_argument('--width', type=int, default=256, metavar='N',
                        help='Width of the image to resize (default: 240)')
    parser.add_argument('--train_files', type=str, default='dataSegmentation/trainImage', metavar='N',
                        help='Path for training images')
    parser.add_argument('--train_masks', type=str, default='dataSegmentation/trainMask', metavar='N',
                        help='Path for training masks')
    parser.add_argument('--val_files', type=str, default='dataSegmentation/valImage', metavar='N',
                        help='Path for validation images')
    parser.add_argument('--val_masks', type=str, default='dataSegmentation/valMask', metavar='N',
                        help='Path for validation masks')
    parser.add_argument('--feature_extract', type=bool, default=False, metavar='N',
                        help='finetuning the last layer')
    parser.add_argument('--model_path', default='models/',
                        help='For Saving the current Model')
    args = parser.parse_args()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    model.to(device)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    cudnn.benchmark= True
    if os.path.exists("model_office.pth"):
        temp = torch.load("model_office.pth")
        model.load_state_dict(temp['state_dict'])
    criterion=torch.nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(),lr=args.lr)
    scheduler=ReduceLROnPlateau(optimizer,mode='min',patience=3, verbose=True)
    train_loader, val_loader = get_loaders_smp(args.train_files, args.train_masks, 
                                            args.val_files, args.val_masks, args.batch_size, args.height, args.width)

    best_loss=float('inf')
    phases=['train','val']
    losses={phase:[] for phase in phases}
    dice_scores={phase:[] for phase in phases}
    dataloaders = {'train': train_loader, 'val': val_loader}
    accumulation_steps=4//args.batch_size
    
    for epoch in range (args.num_epochs):
        iterate(args, model, epoch, phases[0], dataloaders, optimizer, criterion, device, accumulation_steps, losses, dice_scores)
        state = {
            "epoch": epoch,
            "best_loss": best_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        with torch.no_grad():
            val_loss=iterate(args, model, epoch, phases[1], dataloaders, optimizer, criterion, device, accumulation_steps, losses, dice_scores)
            scheduler.step(val_loss)
        if val_loss < best_loss:
            print("******** New optimal found, saving state ********")
            state["best_loss"] = best_loss = val_loss
            torch.save(state, "./model_office.pth")
        print ()
    
    
if __name__ == "__main__":
    main()