import torch
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch.optim as optim
from uNet import UNet16
from utils import get_loaders_segmentation, save_predictions_as_imgs, check_accuracy, load_checkpoint, save_checkpoint
import argparse

def train_fn(loader, model, optimizer, loss_fn, scaler, device, val_loader, metrics, epoch):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    # check accuracy
    epoch_acc, epoch_dice = check_accuracy(val_loader, model, device=device)
    if epoch_acc > metrics.acc:
        metrics.acc = epoch_acc
        metrics.dice = epoch_dice
        save_checkpoint(checkpoint)



def main():
    parser = argparse.ArgumentParser(description='PyTorch Crack Classification')
    parser.add_argument('--load', type=bool, default=True, metavar='N',
                        help='Load Pretrained model from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='Learning rate for training (default: 128)')
    parser.add_argument('--batchsize', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--height', type=int, default=256, metavar='N',
                        help='Height of the image to resize (default: 240)')
    parser.add_argument('--weight', type=int, default=256, metavar='N',
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet16(pretrained=True)
    model.eval().to(device)
    
    loss_fn = nn.BCEWithLogitsLoss().to('cuda')
    train_loader, val_loader = get_loaders_segmentation(args.train_files, args.train_masks, args.val_files, args.val_masks, args.height, args.weight, args.batchsize) 
    
    
    
    if args.load:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        current_accuracy, current_dice = check_accuracy(val_loader, model, device=device)
        metrics = {'acc': current_accuracy, 'dice':current_dice}
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    # check_accuracy(val_loader, model, device=device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device, val_loader, metrics, epoch)
        
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=device
        )


if __name__ == "__main__":
    main()