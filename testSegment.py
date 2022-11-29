import argparse, os, torch, torchvision
from uNet import UNet16
from utils import file_loader_for_testing_segmentation, load_checkpoint
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Crack segmentation')
    parser.add_argument('--testing_dir', default='testing/segmentation/',
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    # model = UNET(in_channels=3, out_channels=1).to(device)
    # load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    model = UNet16(pretrained=True)
    model.eval().to(device)
    
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    for image_name in os.listdir(args.testing_dir):
        test_file = args.testing_dir + image_name
        test_file_input = file_loader_for_testing_segmentation(test_file)
        with torch.no_grad():
            preds = torch.sigmoid(model(test_file_input))
            preds = (preds > 0.45).float()
        torchvision.utils.save_image(
            preds, f"{args.testing_dir}/${image_name}_predicted.png"
        )
        
    
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    