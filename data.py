import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import torch.nn as nn
import cv2
from pathlib import Path



#定义训练集和测试集的transform
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

strong_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BirdDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_path = self.get_imgpath(self.root_dir)
        self.train_dataset, self.test_dataset = self.train_test_split(dataset = self.images_path) 
        if mode == "train":
            self.dataset = self.train_dataset
        else:
            self.dataset = self.test_dataset
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_name = str(self.dataset[idx])
        label = img_name.split("/")[-2].split(".")[0]
        label = torch.tensor(int(label), dtype=torch.int64) - 1
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label.long()

    #获取所有图片的路径
    def get_imgpath(self, root):
        img_path = []
        root = Path(root)
        for file in root.glob("**/*.jpg"):
            img_path.append(file)
        return img_path

    def train_test_split(self, dataset, test_size=0.1, seed=42):
        #划分训练集和测试集
        np.random.seed(seed)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_indices = np.random.choice(indices, size=int(test_size*dataset_size), replace=False)
        train_indices = list(set(indices) - set(test_indices))
        train_dataset = [dataset[i] for i in train_indices]
        test_dataset = [dataset[i] for i in test_indices]
        return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset = BrainDataset(root_dir="dataset/CUB_200_2011", mode="train", transform=train_transform)
    test_dataset = BrainDataset(root_dir="dataset/CUB_200_2011", mode="test", transform=test_transform)
    print("train dataset size: ", len(train_dataset))
    print(train_dataset[0])
    print("test dataset size: ", len(test_dataset))
    print(test_dataset[0])
