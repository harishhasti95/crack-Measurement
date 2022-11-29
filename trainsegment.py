import torch
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch.optim as optim
from uNet import UNET
from utils import get_loaders_segmentation, save_predictions_as_imgs, check_accuracy, load_checkpoint, save_checkpoint
import argparse

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
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


def main():
    parser = argparse.ArgumentParser(description='PyTorch Crack Classification')
    parser.add_argument('--load', type=bool, default=False, metavar='N',
                        help='Load Pretrained model from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='Learning rate for training (default: 128)')
    parser.add_argument('--batchsize', type=int, default=4, metavar='N',
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

    

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    if not args.load:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, val_loader = get_loaders_segmentation(args.train_files, args.train_masks, args.val_files, args.val_masks, args.height, args.weight, args.batchsize)

    # check_accuracy(val_loader, model, device=device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        # train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=device)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=device
        )


if __name__ == "__main__":
    main()