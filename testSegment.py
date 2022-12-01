import argparse, os, torch, torchvision
import segmentation_models_pytorch as smp
import torch.backends.cudnn as cudnn
from utils import file_loader_for_testing_segmentation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Crack segmentation')
    parser.add_argument('--testing_dir', default='testing/segmentation/',
                        help='For Saving the current Model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    model.to(device)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    cudnn.benchmark= True
    if os.path.exists("model_office.pth"):
        temp = torch.load("model_office.pth")
        model.load_state_dict(temp['state_dict'])
        
    for image_name in os.listdir(args.testing_dir):
        test_file = args.testing_dir + image_name
        test_file_input = file_loader_for_testing_segmentation(test_file)
        with torch.no_grad():
            preds = torch.sigmoid(model(test_file_input))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{args.testing_dir}/${image_name}_predicted.png"
        )

        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")