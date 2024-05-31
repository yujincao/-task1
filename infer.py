import torch
import os
import numpy as np
from argparse import ArgumentParser
from data import BirdDataset, test_transform
from torch.utils.data import DataLoader
import timm

def parse_args():
    parser = ArgumentParser(description='Image classification inference.')
    parser.add_argument('--model', type=str, default='resnet50', help='Model name.')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file path.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = timm.create_model(args.model, pretrained=False, num_classes=200)
    model.load_state_dict(torch.load(args.ckpt))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load the dataset
    test_data = BirdDataset(root_dir="dataset/CUB_200_2011/images", mode="test", transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    #compute the accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

