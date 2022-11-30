import torch, os, numpy as np, shutil
import tqdm
import torchvision
import torch.nn as nn
import torch.optim as optim
from uNet import UNet16
from utils import get_loaders_segmentation, calc_crack_pixel_weight, get_loaders_smp
import argparse
from torch.autograd import Variable
import scipy.ndimage as ndimage
from pathlib import Path

class LossTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
        model_paths.append(path)
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)

    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = np.argmax(epochs)
        return model_paths[max_idx]
    else:
        return None

def train(train_loader, model, criterion, optimizer, validation, args, valid_loader):
    latest_model_path = find_latest_model_path(args.model_dir)

    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])
    if latest_model_path is not None:
        state = torch.load(best_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        epoch = epoch

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 0
        min_val_los = 9999

    valid_losses = []
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = LossTracker()

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            
            losses.update(loss)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)

def validate(model, val_loader, criterion):
    losses = LossTracker()
    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
            
            
            masks_pred = model(input_var)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            losses.update(loss.item(), input_var.size(0))
            

    return {'valid_loss': losses.avg}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')



def main():
    parser = argparse.ArgumentParser(description='PyTorch Crack Segmentation')
    parser.add_argument('--load', type=bool, default=False, metavar='N', help='Load Pretrained model from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-45, metavar='N', help='Learning rate for training (default: 128)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--height', type=int, default=128, metavar='N', help='Height of the image to resize (default: 240)')
    parser.add_argument('--weight', type=int, default=128, metavar='N', help='Width of the image to resize (default: 240)')
    parser.add_argument('--train_files', type=str, default='dataSegmentation/trainImage', metavar='N', help='Path for training images')
    parser.add_argument('--train_masks', type=str, default='dataSegmentation/trainMask', metavar='N', help='Path for training masks')
    parser.add_argument('--val_files', type=str, default='dataSegmentation/valImage', metavar='N', help='Path for validation images')
    parser.add_argument('--val_masks', type=str, default='dataSegmentation/valMask', metavar='N', help='Path for validation masks')
    parser.add_argument('--feature_extract', type=bool, default=False, metavar='N', help='finetuning the last layer')
    parser.add_argument('--model_dir', default='models/', help='For Saving the current Model')
    parser.add_argument('-n_epoch', default=25, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = UNet16()
    # model.eval().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    # #                             momentum=args.momentum,
    # #                             weight_decay=args.weight_decay)
    # criterion = nn.BCEWithLogitsLoss().to('cuda')

    # train_loader, valid_loader = get_loaders_segmentation(args.train_files, args.train_masks, args.val_files, args.val_masks, args.height, args.weight, args.batch_size)
    # model.cuda()

    # train(train_loader, model, criterion, optimizer, validate, args, valid_loader)
    

if __name__ == "__main__":
    main()