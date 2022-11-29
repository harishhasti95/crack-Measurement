import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import prepare_train_test_val, get_loaders, initialize_model, set_parameter_requires_grad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse


def train_model(args, model, dataloaders, criterion, optimizer, is_inception=False):
    num_epochs = args.epochs
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # labels = labels.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_path + str(args.model_name))
    
    return model, val_acc_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Crack Classification')
    parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--feature_extract', type=bool, default=False, metavar='N',
                        help='finetuning the last layer')
    parser.add_argument('--model_name', default='resnet',
                        help='For Saving the current Model', choices = ['resnet', 'squeezenet', 'densenet', 'inception'])
    parser.add_argument('--model_path', default='models/',
                        help='For Saving the current Model')
    args = parser.parse_args()
    data_dir = 'dataClassification' 
    X_train, X_val, Y_train, Y_val = prepare_train_test_val(data_dir)
    train_loader, val_loader = get_loaders(X_train, Y_train, X_val, Y_val, args.batchsize)
    dataloaders_dict = {'train': train_loader, 'val': val_loader}
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    
    num_classes = len(np.unique(X_train))
    feature_extract = False
    model_ft, input_size = initialize_model(args.model_name, num_classes, args.feature_extract, use_pretrained=True)
    
    if os.path.exists('models/resnet'):
        model_ft = torch.load('models/resnet')
    # Send the model to GPU
    model_ft = model_ft.to(device)
    

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(args, model_ft, dataloaders_dict, criterion, optimizer_ft, is_inception=(args.model_name=="inception"))