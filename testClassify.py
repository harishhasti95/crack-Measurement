import argparse, os, torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from utils import initialize_model, file_loader_for_testing, set_parameter_requires_grad
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Crack Classification')
    parser.add_argument('--model_name', default='resnet',
                        help='For Saving the current Model', choices = ['resnet', 'squeezenet', 'densenet', 'inception'])
    parser.add_argument('--pretrained', default='models/resnet',
                        help='For Saving the current Model')
    parser.add_argument('--testing_dir', default='testing/classification/',
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    model = torch.load(args.pretrained)
    
    for image_name in os.listdir(args.testing_dir):
        test_file = args.testing_dir + image_name
        test_file_input = file_loader_for_testing(test_file)
        outputs = model(test_file_input)
        _, preds = torch.max(outputs, 1)
        # 0 for cracked and 1 for non cracked 
        print(preds)
    